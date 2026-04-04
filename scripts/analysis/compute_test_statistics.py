#!/usr/bin/env python3
"""
Compute statistics for selected test episodes.
Generates metrics similar to multi_feasible_action_statistics.json:
- Device type counts
- Action affordance counts
- Per-home and average overlap metrics (complete and partial)
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import argparse

from scripts.common import PROJECT_ROOT


def extract_device_type(device_name: str) -> str:
    """Extract device type from device name like 'livingRoomLight' -> 'Light'

    Examples:
        livingRoomLight -> Light
        masterBedroomAirConditioner -> AirConditioner
        guestBedroomDehumidifiers -> Dehumidifiers
        kitchenWaterHeater -> WaterHeater
    """
    # Known room prefixes to strip
    room_prefixes = [
        'livingRoom', 'masterBedroom', 'guestBedroom', 'studyRoom', 'diningRoom',
        'bathroom', 'kitchen', 'balcony', 'corridor', 'foyer', 'garage',
        'storeRoom', 'laundryRoom'
    ]

    for prefix in room_prefixes:
        if device_name.startswith(prefix):
            device_type = device_name[len(prefix):]
            if device_type:
                return device_type

    # Fallback: find all capitalized words and take from the second one onward
    # This handles cases like "bathroomLight" -> "Light"
    parts = re.findall(r'[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$', device_name)
    if len(parts) >= 2:
        return ''.join(parts[1:])
    elif parts:
        return ''.join(parts)

    return device_name


def extract_affordance_info(output: dict) -> Optional[Tuple[str, str, str]]:
    """Extract (device, affordance_action, full_affordance) from output entry.

    Returns None for error_input entries.
    """
    if output.get('execution') == 'error_input':
        return None

    affordance = output.get('affordance', '')
    if not affordance:
        return None

    # Parse: http://localhost:8080/workspaces/home86/balcony/artifacts/balconyLight/set_brightness
    match = re.search(r'/artifacts/([^/]+)/([^/]+)$', affordance)
    if match:
        device = match.group(1)
        action = match.group(2)
        return (device, action, f"{device}/{action}")
    return None


def extract_device_room_from_input(input_text: str) -> Set[Tuple[str, str]]:
    """Extract (device, room) pairs from input text for unfeasible tests."""
    input_text = input_text.lower()
    pairs = set()

    # Common device keywords
    devices = [
        'light', 'lamp', 'heating', 'air conditioner', 'ac', 'fan', 'humidifier',
        'dehumidifier', 'curtain', 'blinds', 'aromatherapy', 'tv', 'television',
        'air purifier', 'media player', 'water heater'
    ]

    # Common room keywords (order matters - longer matches first)
    rooms = [
        'master bedroom', 'guest bedroom', 'living room', 'dining room',
        'study room', 'store room', 'laundry room',
        'bathroom', 'bedroom', 'kitchen', 'balcony', 'corridor', 'foyer',
        'garage', 'basement', 'attic'
    ]

    for device in devices:
        if device in input_text:
            for room in rooms:
                if room in input_text:
                    room_normalized = room.replace(' ', '_')
                    pairs.add((device, room_normalized))

    return pairs


def get_test_affordances(test: dict, from_input: bool = False) -> Set[str]:
    """Get set of affordances (device/action or device/room) for a test.

    Args:
        from_input: If True, extract device/room pairs from input text (for unfeasible tests)
    """
    if from_input:
        # For unfeasible tests, return device/room pairs as "affordances"
        pairs = extract_device_room_from_input(test.get('input', ''))
        return {f"{device}/{room}" for device, room in pairs}

    affordances = set()
    for output in test.get('output', []):
        info = extract_affordance_info(output)
        if info:
            affordances.add(info[2])  # device/action
    return affordances


def compute_overlap_metrics(tests: List[dict], from_input: bool = False) -> dict:
    """Compute overlap metrics for a list of tests.

    Args:
        from_input: If True, extract device/room from input text (for unfeasible tests)

    Returns:
        - complete_overlap_count: number of tests with exact same affordances as a previous test
        - partial_overlap_count: number of tests sharing at least one affordance with a previous test
        - complete_overlap_pairs: list of (test_id, overlapping_test_id) pairs
        - partial_overlap_pairs: list of (test_id, overlapping_test_id, shared_affordances) tuples
    """
    seen_affordance_sets = {}  # affordance_tuple -> test_id
    seen_individual_affordances = defaultdict(list)  # affordance -> [test_ids]

    complete_overlap_count = 0
    partial_overlap_count = 0
    complete_overlap_pairs = []
    partial_overlap_pairs = []

    for test in tests:
        test_id = test.get('id', '')
        affordances = get_test_affordances(test, from_input=from_input)

        if not affordances:
            continue

        affordance_tuple = tuple(sorted(affordances))

        # Check complete overlap
        if affordance_tuple in seen_affordance_sets:
            complete_overlap_count += 1
            complete_overlap_pairs.append((test_id, seen_affordance_sets[affordance_tuple]))
        else:
            seen_affordance_sets[affordance_tuple] = test_id

        # Check partial overlap (at least one shared affordance)
        overlapping_tests = set()
        shared_affordances_map = defaultdict(set)
        for aff in affordances:
            for prev_test_id in seen_individual_affordances[aff]:
                overlapping_tests.add(prev_test_id)
                shared_affordances_map[prev_test_id].add(aff)

        if overlapping_tests:
            partial_overlap_count += 1
            # Record the first overlapping test with its shared affordances
            first_overlap = list(overlapping_tests)[0]
            partial_overlap_pairs.append((
                test_id,
                first_overlap,
                list(shared_affordances_map[first_overlap])
            ))

        # Add this test's affordances to seen
        for aff in affordances:
            seen_individual_affordances[aff].append(test_id)

    return {
        'complete_overlap_count': complete_overlap_count,
        'partial_overlap_count': partial_overlap_count,
        'complete_overlap_pairs': complete_overlap_pairs,
        'partial_overlap_pairs': partial_overlap_pairs,
        'total_tests': len(tests)
    }


def compute_statistics(tests: List[dict], category_name: str, from_input: bool = False) -> dict:
    """Compute comprehensive statistics for a list of tests.

    Args:
        from_input: If True, extract device/room from input text (for unfeasible tests)
    """

    # Collect all devices and affordances
    all_devices = set()
    device_type_counts = Counter()
    device_affordance_counts = Counter()
    command_count_distribution = Counter()

    # For unfeasible tests, track device/room pairs from input
    input_device_room_counts = Counter()

    # Group tests by home
    tests_by_home = defaultdict(list)

    for test in tests:
        test_id = test.get('id', '')
        home_match = re.match(r'(home\d+)_', test_id)
        home_id = home_match.group(1) if home_match else 'unknown'
        tests_by_home[home_id].append(test)

        outputs = test.get('output', [])

        if from_input:
            # For unfeasible tests, extract from input text
            pairs = extract_device_room_from_input(test.get('input', ''))
            for device, room in pairs:
                input_device_room_counts[f"{device}/{room}"] += 1
                device_type_counts[device] += 1
        else:
            # Count successful commands from affordances
            for output in outputs:
                info = extract_affordance_info(output)
                if info:
                    device, action, full_affordance = info
                    all_devices.add(device)
                    device_type = extract_device_type(device)
                    device_type_counts[device_type] += 1
                    device_affordance_counts[full_affordance] += 1

        # Track command count distribution
        command_count_distribution[len(outputs)] += 1

    # Compute per-home overlap metrics
    per_home_metrics = {}
    total_complete_overlap = 0
    total_partial_overlap = 0

    for home_id, home_tests in tests_by_home.items():
        home_overlap = compute_overlap_metrics(home_tests, from_input=from_input)

        # Group complete overlaps by their affordance set
        # Each group contains test IDs that share the exact same affordances
        complete_overlap_groups = defaultdict(list)
        for test_id, prev_test_id in home_overlap['complete_overlap_pairs']:
            # Find the affordance tuple for this pair
            for test in home_tests:
                if test.get('id') == prev_test_id:
                    affs = tuple(sorted(get_test_affordances(test, from_input=from_input)))
                    complete_overlap_groups[affs].append(prev_test_id)
                    complete_overlap_groups[affs].append(test_id)
                    break

        # Deduplicate and create list of groups
        complete_overlap_ids = []
        for affs, ids in complete_overlap_groups.items():
            unique_ids = list(dict.fromkeys(ids))  # Preserve order, remove duplicates
            if unique_ids not in complete_overlap_ids:
                complete_overlap_ids.append(unique_ids)

        # Extract partial overlap pairs as ([test_id1, test_id2], overlap_size)
        partial_overlap_ids = [
            ([pair[0], pair[1]], len(pair[2]))  # ([current_test, overlapping_test], num shared affordances)
            for pair in home_overlap['partial_overlap_pairs']
        ]

        per_home_metrics[home_id] = {
            'test_count': len(home_tests),
            'complete_overlap_count': home_overlap['complete_overlap_count'],
            'partial_overlap_count': home_overlap['partial_overlap_count'],
            'complete_overlap_rate': home_overlap['complete_overlap_count'] / len(home_tests) if home_tests else 0,
            'partial_overlap_rate': home_overlap['partial_overlap_count'] / len(home_tests) if home_tests else 0,
            'complete_overlap_ids': complete_overlap_ids,
            'partial_overlap_ids': partial_overlap_ids,
        }
        total_complete_overlap += home_overlap['complete_overlap_count']
        total_partial_overlap += home_overlap['partial_overlap_count']

    # Compute averages
    num_homes = len(tests_by_home)
    avg_complete_overlap_rate = sum(m['complete_overlap_rate'] for m in per_home_metrics.values()) / num_homes if num_homes else 0
    avg_partial_overlap_rate = sum(m['partial_overlap_rate'] for m in per_home_metrics.values()) / num_homes if num_homes else 0

    # Build result
    if from_input:
        # For unfeasible tests
        result = {
            'category': category_name,
            'summary': {
                'total_tests': len(tests),
                'total_homes': num_homes,
                'unique_device_room_pairs': len(input_device_room_counts),
                'unique_device_types': len(device_type_counts),
                'total_complete_overlaps': total_complete_overlap,
                'total_partial_overlaps': total_partial_overlap,
                'avg_complete_overlap_rate_per_home': round(avg_complete_overlap_rate, 4),
                'avg_partial_overlap_rate_per_home': round(avg_partial_overlap_rate, 4),
            },
            'device_type_counts': dict(device_type_counts.most_common()),
            'device_room_counts': dict(input_device_room_counts.most_common()),
            'command_count_distribution': {str(k): v for k, v in sorted(command_count_distribution.items())},
            'per_home_metrics': per_home_metrics,
        }
    else:
        result = {
            'category': category_name,
            'summary': {
                'total_tests': len(tests),
                'total_homes': num_homes,
                'unique_devices': len(all_devices),
                'unique_device_types': len(device_type_counts),
                'unique_affordances': len(device_affordance_counts),
                'total_complete_overlaps': total_complete_overlap,
                'total_partial_overlaps': total_partial_overlap,
                'avg_complete_overlap_rate_per_home': round(avg_complete_overlap_rate, 4),
                'avg_partial_overlap_rate_per_home': round(avg_partial_overlap_rate, 4),
            },
            'devices': sorted(all_devices),
            'device_type_counts': dict(device_type_counts.most_common()),
            'device_affordance_counts': dict(device_affordance_counts.most_common()),
            'command_count_distribution': {str(k): v for k, v in sorted(command_count_distribution.items())},
            'per_home_metrics': per_home_metrics,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description='Compute statistics for selected test episodes')
    parser.add_argument('--input-dir', '-i', default='data/homebench/benchmarks/repeated_query',
                       help='Directory containing selected test files')
    parser.add_argument('--output-dir', '-o', default='data/homebench/benchmarks/repeated_query',
                       help='Directory for output statistics files')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Define categories and their files (with from_input flag for unfeasible tests)
    categories = [
        ('single_feasible', 'selected_test_single_feasible.json', False),
        ('single_unfeasible', 'selected_test_single_unfeasible.json', True),  # from_input=True
        ('multi_feasible', 'selected_test_multi_feasible.json', False),
        ('multi_mixed', 'selected_test_multi_mixed.json', False),
    ]

    all_stats = {}

    for category_name, filename, from_input in categories:
        input_path = input_dir / filename

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping {category_name}")
            continue

        print(f"\nProcessing {category_name}...")

        with open(input_path, 'r') as f:
            tests = json.load(f)

        stats = compute_statistics(tests, category_name, from_input=from_input)
        all_stats[category_name] = stats

        # Save individual category stats
        output_path = output_dir / f'{category_name}_statistics.json'
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to {output_path}")

        # Print summary
        s = stats['summary']
        print(f"  Tests: {s['total_tests']}, Homes: {s['total_homes']}")
        if from_input:
            print(f"  Unique device/room pairs: {s.get('unique_device_room_pairs', 0)}, Device types: {s['unique_device_types']}")
        else:
            print(f"  Unique devices: {s.get('unique_devices', 0)}, Device types: {s['unique_device_types']}")
            print(f"  Unique affordances: {s.get('unique_affordances', 0)}")
        print(f"  Complete overlaps: {s['total_complete_overlaps']} (avg rate: {s['avg_complete_overlap_rate_per_home']:.2%})")
        print(f"  Partial overlaps: {s['total_partial_overlaps']} (avg rate: {s['avg_partial_overlap_rate_per_home']:.2%})")

    # Save combined stats
    combined_output = output_dir / 'all_categories_statistics.json'
    with open(combined_output, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved combined statistics to {combined_output}")

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<20} {'Tests':>6} {'Homes':>6} {'DevTypes':>8} {'Affordances':>12} {'Complete':>10} {'Partial':>10}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")

    for cat_name, stats in all_stats.items():
        s = stats['summary']
        affordances = s.get('unique_affordances', s.get('unique_device_room_pairs', 0))
        print(f"{cat_name:<20} {s['total_tests']:>6} {s['total_homes']:>6} {s['unique_device_types']:>8} {affordances:>12} {s['total_complete_overlaps']:>10} {s['total_partial_overlaps']:>10}")


if __name__ == '__main__':
    main()
