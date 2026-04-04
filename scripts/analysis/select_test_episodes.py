#!/usr/bin/env python3
"""
Script to select 400 episodes from test_data.json according to specific criteria.
- 100 single_feasible: home<id>_one_*, single output, no error_input
- 100 single_unfeasible: home<id>_one_*, single output, error_input
- 100 multi_feasible: home<id>_multi_*, multiple outputs, no error_input
- 100 multi_mixed: home<id>_multi_*, multiple outputs, has error_input (but at most half)
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
import rdflib

from scripts.common import PROJECT_ROOT

TTL_DIR = PROJECT_ROOT / "data" / "homebench" / "hmas" / "home_description"

PROPERTY_RANGE_SPARQL = '''
PREFIX td: <https://www.w3.org/2019/wot/td#>
PREFIX hctl: <https://www.w3.org/2019/wot/hypermedia#>
PREFIX jsonschema: <https://www.w3.org/2019/wot/json-schema#>

SELECT ?target ?min ?max WHERE {
    ?thing td:hasPropertyAffordance ?prop .
    ?prop td:hasForm ?form .
    ?form hctl:hasTarget ?target .
    ?prop td:hasOutputSchema ?schema .
    ?schema jsonschema:minimum ?min .
    ?schema jsonschema:maximum ?max .
}
'''


def build_property_range_cache(ttl_dir: Path = TTL_DIR) -> Dict[str, Tuple[float, float]]:
    """Parse all home TTL files and extract jsonschema:minimum/maximum for each property URL.

    Returns a dict mapping property URL -> (min_value, max_value).
    """
    property_ranges: Dict[str, Tuple[float, float]] = {}

    for ttl_file in sorted(ttl_dir.glob("home_*.ttl")):
        g = rdflib.Graph()
        g.parse(ttl_file, format='turtle')
        for row in g.query(PROPERTY_RANGE_SPARQL):
            prop_url = str(row["target"])
            min_val = float(row["min"])
            max_val = float(row["max"])
            property_ranges[prop_url] = (min_val, max_val)

    return property_ranges


def has_out_of_range_values(test: dict, property_ranges: Dict[str, Tuple[float, float]]) -> bool:
    """Check if any feasible output in a test sets a value outside its allowed range.

    Only checks outputs that are NOT error_input (i.e., feasible commands).
    Returns True if any expected value is out of range.
    """
    for output in test.get('output', []):
        if output.get('execution') == 'error_input':
            continue
        test_info = output.get('test', {})
        prop_url = test_info.get('property', '')
        expected_val = test_info.get('expected_value')

        if prop_url in property_ranges and isinstance(expected_val, (int, float)):
            min_val, max_val = property_ranges[prop_url]
            if expected_val < min_val or expected_val > max_val:
                return True
    return False


def extract_home_id(test_id: str) -> Optional[str]:
    """Extract home ID from test ID like 'home86_multi_329' -> '86'"""
    match = re.match(r'home(\d+)_', test_id)
    return match.group(1) if match else None


def extract_device_room_pairs(test: dict, from_input: bool = False) -> Set[Tuple[str, str]]:
    """Extract (device, room) pairs from a test case.

    Args:
        test: Test case dictionary
        from_input: If True, extract from input text (for unfeasible tests).
                   If False, extract from affordance URLs.
    """
    pairs = set()

    if from_input:
        # Extract device/room from input text for unfeasible tests
        # Pattern matches phrases like "light in the bathroom", "heating in the master bedroom"
        input_text = test.get('input', '').lower()

        # Common device keywords
        devices = [
            'light', 'lamp', 'heating', 'air conditioner', 'ac', 'fan', 'humidifier',
            'dehumidifier', 'curtain', 'blinds', 'aromatherapy', 'tv', 'television'
        ]

        # Common room keywords
        rooms = [
            'bathroom', 'bedroom', 'master bedroom', 'guest bedroom', 'living room',
            'kitchen', 'dining room', 'study room', 'balcony', 'corridor', 'foyer',
            'garage', 'laundry room', 'basement', 'attic'
        ]

        for device in devices:
            for room in rooms:
                # Check if both device and room are mentioned
                if device in input_text and room in input_text:
                    # Normalize room name to match affordance format
                    room_normalized = room.replace(' ', '_')
                    pairs.add((device, room_normalized))

        return pairs

    # Extract from affordance URLs
    for output in test.get('output', []):
        if output.get('execution') == 'error_input':
            continue
        affordance = output.get('affordance', '')
        # Parse affordance URL like: http://localhost:8080/workspaces/home86/balcony/artifacts/balconyLight/set_brightness
        match = re.search(r'/workspaces/home\d+/([^/]+)/artifacts/([^/]+)/', affordance)
        if match:
            room = match.group(1)
            device = match.group(2)
            pairs.add((device, room))
    return pairs


def count_commands(test: dict) -> int:
    """Count the number of commands (outputs) in a test case."""
    return len(test.get('output', []))


def is_single_feasible(test: dict) -> bool:
    """Check if test is single_feasible: home<id>_one_*, single output, no error_input"""
    test_id = test.get('id', '')
    if '_one_' not in test_id:
        return False
    outputs = test.get('output', [])
    if len(outputs) != 1:
        return False
    return outputs[0].get('execution') != 'error_input'


def is_single_unfeasible(test: dict) -> bool:
    """Check if test is single_unfeasible: home<id>_one_*, single output, error_input"""
    test_id = test.get('id', '')
    if '_one_' not in test_id:
        return False
    outputs = test.get('output', [])
    if len(outputs) != 1:
        return False
    return outputs[0].get('execution') == 'error_input'


def is_multi_feasible(test: dict) -> bool:
    """Check if test is multi_feasible: home<id>_multi_*, multiple outputs, no error_input"""
    test_id = test.get('id', '')
    if '_multi_' not in test_id:
        return False
    outputs = test.get('output', [])
    if len(outputs) <= 1:
        return False
    return all(o.get('execution') != 'error_input' for o in outputs)


def is_multi_mixed(test: dict) -> bool:
    """Check if test is multi_mixed: home<id>_multi_*, multiple outputs, has error_input (at most half)"""
    test_id = test.get('id', '')
    if '_multi_' not in test_id:
        return False
    outputs = test.get('output', [])
    if len(outputs) <= 1:
        return False
    error_count = sum(1 for o in outputs if o.get('execution') == 'error_input')
    # Must have at least one error_input and at most half
    return error_count >= 1 and error_count <= len(outputs) // 2


def group_by_home(tests: List[dict]) -> Dict[str, List[dict]]:
    """Group tests by home ID."""
    groups = defaultdict(list)
    for test in tests:
        home_id = extract_home_id(test.get('id', ''))
        if home_id:
            groups[home_id].append(test)
    return groups


def filter_homes_with_min_instances(groups: Dict[str, List[dict]], min_count: int = 50) -> Dict[str, List[dict]]:
    """Keep only groups with at least min_count instances."""
    return {home_id: tests for home_id, tests in groups.items() if len(tests) >= min_count}


def find_tests_with_same_device_room(tests: List[dict], from_input: bool = False) -> List[Tuple[dict, dict]]:
    """Find pairs of tests that use the same device in the same room."""
    pairs = []
    for i, test1 in enumerate(tests):
        pairs1 = extract_device_room_pairs(test1, from_input=from_input)
        for test2 in tests[i+1:]:
            pairs2 = extract_device_room_pairs(test2, from_input=from_input)
            if pairs1 & pairs2:  # Intersection - same device/room pair
                pairs.append((test1, test2))
    return pairs


def find_tests_with_exact_same_devices(tests: List[dict], from_input: bool = False) -> List[Tuple[dict, dict]]:
    """Find pairs of tests that use the exact same devices in the exact same rooms."""
    pairs = []
    for i, test1 in enumerate(tests):
        pairs1 = extract_device_room_pairs(test1, from_input=from_input)
        for test2 in tests[i+1:]:
            pairs2 = extract_device_room_pairs(test2, from_input=from_input)
            if pairs1 == pairs2 and pairs1:  # Exact match
                pairs.append((test1, test2))
    return pairs


def select_single_tests(
    home_tests: List[dict],
    target_count: int = 10,
    seed: Optional[int] = None,
    from_input: bool = False
) -> List[dict]:
    """
    Select tests for single_feasible or single_unfeasible categories.
    Ensures at least 2 tests use the same device in the same room.

    Args:
        from_input: If True, extract device/room from input text (for unfeasible tests)
    """
    if seed is not None:
        random.seed(seed)

    selected = []
    remaining = list(home_tests)

    # First, try to find 2 tests with the same device/room
    same_device_pairs = find_tests_with_same_device_room(remaining, from_input=from_input)

    if same_device_pairs:
        # Pick a random pair
        pair = random.choice(same_device_pairs)
        selected.extend(pair)
        remaining = [t for t in remaining if t not in selected]

    # Fill the rest randomly
    needed = target_count - len(selected)
    if needed > 0 and remaining:
        additional = random.sample(remaining, min(needed, len(remaining)))
        selected.extend(additional)

    return selected


def select_multi_tests(
    home_tests: List[dict],
    target_count: int = 10,
    seed: Optional[int] = None
) -> List[dict]:
    """
    Select tests for multi_feasible or multi_mixed categories.
    Requirements (best effort):
    - 2 tests with exact same devices in exact same rooms
    - Rest with at least one common device/room pair
    - 5 with 2 commands, 3 with 3 commands, 2 with 4+ commands
    """
    if seed is not None:
        random.seed(seed)

    # Group by command count
    by_cmd_count = {
        2: [t for t in home_tests if count_commands(t) == 2],
        3: [t for t in home_tests if count_commands(t) == 3],
        '4+': [t for t in home_tests if count_commands(t) >= 4]
    }

    selected = []
    used_ids = set()

    def add_test(test):
        if test['id'] not in used_ids:
            selected.append(test)
            used_ids.add(test['id'])

    # Try to find 2 tests with exact same devices
    exact_pairs = find_tests_with_exact_same_devices(home_tests)
    if exact_pairs:
        pair = random.choice(exact_pairs)
        for t in pair:
            add_test(t)

    # Now try to fulfill the command count requirements
    # Need: 5 with 2 commands, 3 with 3 commands, 2 with 4+
    targets = [(2, 5), (3, 3), ('4+', 2)]

    for cmd_count, target in targets:
        available = [t for t in by_cmd_count.get(cmd_count, []) if t['id'] not in used_ids]

        # If we have some tests with common device/room pairs, prefer those
        if selected:
            common_pairs = extract_device_room_pairs(selected[0])
            for s in selected[1:]:
                common_pairs |= extract_device_room_pairs(s)

            # Sort by preference (those with common pairs first)
            def has_common(t):
                t_pairs = extract_device_room_pairs(t)
                return bool(t_pairs & common_pairs)

            with_common = [t for t in available if has_common(t)]
            without_common = [t for t in available if not has_common(t)]
            available = with_common + without_common

        current_count = sum(1 for t in selected if
                          (cmd_count == '4+' and count_commands(t) >= 4) or
                          (cmd_count != '4+' and count_commands(t) == cmd_count))
        needed = target - current_count

        if needed > 0 and available:
            to_add = available[:min(needed, len(available))]
            for t in to_add:
                add_test(t)

    # Fill remaining slots if needed
    remaining = [t for t in home_tests if t['id'] not in used_ids]
    random.shuffle(remaining)
    while len(selected) < target_count and remaining:
        add_test(remaining.pop(0))

    return selected[:target_count]


def verify_selection_constraints(selected: List[dict], is_multi: bool, category_name: str, from_input: bool = False) -> dict:
    """Verify and report on how well selection constraints were met."""
    verification = {
        'total_selected': len(selected),
        'same_device_room_pairs': 0,
        'exact_same_device_pairs': 0,
    }

    # Check for same device/room pairs
    same_pairs = find_tests_with_same_device_room(selected, from_input=from_input)
    verification['same_device_room_pairs'] = len(same_pairs)

    if is_multi:
        # Check for exact same devices
        exact_pairs = find_tests_with_exact_same_devices(selected, from_input=from_input)
        verification['exact_same_device_pairs'] = len(exact_pairs)

        # Count by command count
        cmd_counts = {2: 0, 3: 0, '4+': 0}
        for t in selected:
            n = count_commands(t)
            if n == 2:
                cmd_counts[2] += 1
            elif n == 3:
                cmd_counts[3] += 1
            elif n >= 4:
                cmd_counts['4+'] += 1
        verification['by_command_count'] = cmd_counts
        verification['target_command_counts'] = {2: 5, 3: 3, '4+': 2}

    return verification


def select_category(
    all_tests: List[dict],
    category_filter: callable,
    is_multi: bool = False,
    num_homes: int = 10,
    tests_per_home: int = 10,
    min_instances: int = 50,
    seed: Optional[int] = None,
    category_name: str = "",
    from_input: bool = False,
    target_total: int = 100
) -> Tuple[List[dict], dict]:
    """
    Select tests for a category.
    Returns selected tests and statistics.

    Args:
        from_input: If True, extract device/room from input text (for unfeasible tests)
        target_total: Target total number of tests to select
    """
    if seed is not None:
        random.seed(seed)

    # Filter tests by category
    category_tests = [t for t in all_tests if category_filter(t)]

    # Group by home
    home_groups = group_by_home(category_tests)

    # Filter homes with at least min_instances
    eligible_homes = filter_homes_with_min_instances(home_groups, min_instances)

    stats = {
        'total_category_tests': len(category_tests),
        'total_homes': len(home_groups),
        'eligible_homes': len(eligible_homes),
        'eligible_home_ids': list(eligible_homes.keys()),
    }

    # If not enough eligible homes, relax the constraint
    if len(eligible_homes) < num_homes:
        print(f"  Warning: Only {len(eligible_homes)} homes with >= {min_instances} instances, using all available homes")
        # Sort homes by test count and take top ones
        sorted_homes = sorted(home_groups.items(), key=lambda x: len(x[1]), reverse=True)
        eligible_homes = dict(sorted_homes[:max(num_homes, len(home_groups))])
        stats['relaxed_constraint'] = True

    # Calculate how many homes we need to reach target_total
    # If total available tests < target_total, use all homes
    total_available = sum(len(tests) for tests in eligible_homes.values())
    if total_available < target_total:
        print(f"  Note: Only {total_available} tests available in category (target: {target_total})")
        selected_home_ids = list(eligible_homes.keys())
    else:
        # Calculate optimal number of homes based on tests per home
        available_homes = list(eligible_homes.keys())
        # Try to select homes that together have enough tests
        sorted_home_ids = sorted(available_homes, key=lambda h: len(eligible_homes[h]), reverse=True)

        selected_home_ids = []
        total_selected_capacity = 0
        for home_id in sorted_home_ids:
            if len(selected_home_ids) >= num_homes and total_selected_capacity >= target_total:
                break
            selected_home_ids.append(home_id)
            total_selected_capacity += min(len(eligible_homes[home_id]), tests_per_home)

        # If we need more homes to reach target, add more
        if total_selected_capacity < target_total:
            for home_id in sorted_home_ids:
                if home_id not in selected_home_ids:
                    selected_home_ids.append(home_id)
                    total_selected_capacity += len(eligible_homes[home_id])
                    if total_selected_capacity >= target_total:
                        break

        # Shuffle to add randomness while keeping high-capacity homes
        random.shuffle(selected_home_ids)

    stats['selected_home_ids'] = selected_home_ids
    stats['tests_per_selected_home'] = {h: len(eligible_homes[h]) for h in selected_home_ids}

    # Select tests from each home
    all_selected = []
    for home_id in selected_home_ids:
        home_tests = eligible_homes[home_id]
        if is_multi:
            selected = select_multi_tests(home_tests, tests_per_home, seed)
        else:
            selected = select_single_tests(home_tests, tests_per_home, seed, from_input=from_input)
        all_selected.extend(selected)

    # Verify constraints
    stats['verification'] = verify_selection_constraints(all_selected, is_multi, category_name, from_input=from_input)

    return all_selected, stats


def _extract_device_affordance_info(test: dict) -> List[Tuple[str, str, str]]:
    """Extract (device_name, device_type, affordance_action) from feasible outputs.

    Returns list of tuples like ('balconyLight', 'Light', 'set_brightness').
    """
    results = []
    for output in test.get('output', []):
        if output.get('execution') == 'error_input':
            continue
        affordance = output.get('affordance', '')
        # e.g. .../artifacts/balconyLight/set_brightness
        match = re.search(r'/artifacts/([^/]+)/([^/]+)$', affordance)
        if match:
            device_name = match.group(1)
            action = match.group(2)
            # Extract device type from camelCase name: 'balconyLight' -> 'Light'
            type_match = re.search(r'[a-z]([A-Z][a-zA-Z]+)$', device_name)
            device_type = type_match.group(1) if type_match else device_name
            results.append((device_name, device_type, action))
    return results


def _compute_overlaps(tests: List[dict]) -> Tuple[List, List]:
    """Compute complete and partial device-set overlaps among tests.

    Complete overlap: two tests target the exact same set of devices.
    Partial overlap: two tests share at least one common device (with count of shared devices).
    """
    # Build device sets per test
    test_device_sets = []
    for t in tests:
        devices = set()
        for output in t.get('output', []):
            if output.get('execution') == 'error_input':
                continue
            affordance = output.get('affordance', '')
            match = re.search(r'/artifacts/([^/]+)/', affordance)
            if match:
                devices.add(match.group(1))
        test_device_sets.append((t['id'], devices))

    complete_overlaps = []
    partial_overlaps = []

    for i, (id1, devs1) in enumerate(test_device_sets):
        for id2, devs2 in test_device_sets[i+1:]:
            if not devs1 or not devs2:
                continue
            if devs1 == devs2:
                complete_overlaps.append([id1, id2])
            shared = len(devs1 & devs2)
            if shared > 0:
                partial_overlaps.append([[id1, id2], shared])

    return complete_overlaps, partial_overlaps


def compute_category_statistics(tests: List[dict], category_name: str) -> dict:
    """Compute detailed statistics for a category of selected tests.

    Matches the format used in data/homebench/benchmarks/repeated_query/*_statistics.json.
    """
    # Collect device info across all tests
    all_devices = set()
    device_type_counter: Dict[str, int] = defaultdict(int)
    device_affordance_counter: Dict[str, int] = defaultdict(int)
    command_count_dist: Dict[str, int] = defaultdict(int)

    for test in tests:
        infos = _extract_device_affordance_info(test)
        for device_name, device_type, action in infos:
            all_devices.add(device_name)
            device_type_counter[device_type] += 1
            device_affordance_counter[f"{device_name}/{action}"] += 1

        command_count_dist[str(count_commands(test))] += 1

    # Unique affordances
    unique_affordances = len(device_affordance_counter)

    # Sort device_type_counts and device_affordance_counts by count descending
    device_type_counts = dict(sorted(device_type_counter.items(), key=lambda x: -x[1]))
    device_affordance_counts = dict(sorted(device_affordance_counter.items(), key=lambda x: -x[1]))

    # Per-home metrics
    home_groups = group_by_home(tests)
    per_home_metrics = {}
    total_complete = 0
    total_partial = 0
    overlap_rates_complete = []
    overlap_rates_partial = []

    for home_id, home_tests in home_groups.items():
        complete_overlaps, partial_overlaps = _compute_overlaps(home_tests)
        n = len(home_tests)
        complete_rate = len(complete_overlaps) / n if n > 0 else 0
        partial_rate = len(partial_overlaps) / n if n > 0 else 0

        per_home_metrics[f"home{home_id}"] = {
            "test_count": n,
            "complete_overlap_count": len(complete_overlaps),
            "partial_overlap_count": len(partial_overlaps),
            "complete_overlap_rate": round(complete_rate, 2),
            "partial_overlap_rate": round(partial_rate, 2),
            "complete_overlap_ids": complete_overlaps,
            "partial_overlap_ids": partial_overlaps,
        }

        total_complete += len(complete_overlaps)
        total_partial += len(partial_overlaps)
        overlap_rates_complete.append(complete_rate)
        overlap_rates_partial.append(partial_rate)

    num_homes = len(home_groups)
    avg_complete_rate = round(sum(overlap_rates_complete) / num_homes, 2) if num_homes else 0
    avg_partial_rate = round(sum(overlap_rates_partial) / num_homes, 2) if num_homes else 0

    return {
        "category": category_name,
        "summary": {
            "total_tests": len(tests),
            "total_homes": num_homes,
            "unique_devices": len(all_devices),
            "unique_device_types": len(device_type_counts),
            "unique_affordances": unique_affordances,
            "total_complete_overlaps": total_complete,
            "total_partial_overlaps": total_partial,
            "avg_complete_overlap_rate_per_home": avg_complete_rate,
            "avg_partial_overlap_rate_per_home": avg_partial_rate,
        },
        "devices": sorted(all_devices),
        "device_type_counts": device_type_counts,
        "device_affordance_counts": device_affordance_counts,
        "command_count_distribution": dict(sorted(command_count_dist.items())),
        "per_home_metrics": per_home_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Select 400 test episodes from test_data.json')
    parser.add_argument('--input', '-i', default='data/homebench/converted/test_data.json',
                       help='Input test data file')
    parser.add_argument('--output-dir', '-o', default='data/homebench/benchmarks/repeated_query',
                       help='Output directory for per-category test and statistics files')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed statistics')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load test data
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    print(f"Loading test data from {input_path}...")
    with open(input_path, 'r') as f:
        all_tests = json.load(f)
    print(f"Loaded {len(all_tests)} tests")

    # Filter out tests with out-of-range expected values
    print("\nBuilding property range constraints from TTL files...")
    property_ranges = build_property_range_cache()
    print(f"Loaded {len(property_ranges)} property range constraints")

    original_count = len(all_tests)
    all_tests = [t for t in all_tests if not has_out_of_range_values(t, property_ranges)]
    filtered_count = original_count - len(all_tests)
    print(f"Filtered out {filtered_count} tests with out-of-range expected values")
    print(f"Remaining: {len(all_tests)} tests")

    results = {
        'metadata': {
            'seed': args.seed,
            'source_file': str(input_path),
            'total_source_tests': original_count,
            'filtered_out_of_range': filtered_count,
            'tests_after_filtering': len(all_tests)
        },
        'categories': {}
    }

    # Category 1: single_feasible (100 tests)
    print("\nSelecting single_feasible tests...")
    single_feasible, sf_stats = select_category(
        all_tests, is_single_feasible, is_multi=False,
        num_homes=10, tests_per_home=10, min_instances=50, seed=args.seed,
        category_name='single_feasible', from_input=False, target_total=100
    )
    results['categories']['single_feasible'] = {
        'tests': single_feasible,
        'count': len(single_feasible),
        'stats': sf_stats
    }
    print(f"  Selected {len(single_feasible)} tests from {len(sf_stats.get('selected_home_ids', []))} homes")

    # Category 2: single_unfeasible (100 tests)
    # Use from_input=True because unfeasible tests have no affordance URLs
    print("\nSelecting single_unfeasible tests...")
    single_unfeasible, su_stats = select_category(
        all_tests, is_single_unfeasible, is_multi=False,
        num_homes=10, tests_per_home=10, min_instances=50, seed=args.seed + 1,
        category_name='single_unfeasible', from_input=True, target_total=100
    )
    results['categories']['single_unfeasible'] = {
        'tests': single_unfeasible,
        'count': len(single_unfeasible),
        'stats': su_stats
    }
    print(f"  Selected {len(single_unfeasible)} tests from {len(su_stats.get('selected_home_ids', []))} homes")

    # Category 3: multi_feasible (100 tests)
    # Note: This category has limited data (~237 tests across 87 homes, max 8 per home)
    # We need to select from enough homes to get ~100 tests total
    # Since max per home is ~8, we need ~15-20 homes to get 100 tests
    print("\nSelecting multi_feasible tests...")
    multi_feasible, mf_stats = select_category(
        all_tests, is_multi_feasible, is_multi=True,
        num_homes=20, tests_per_home=10, min_instances=50, seed=args.seed + 2,
        category_name='multi_feasible', from_input=False, target_total=100
    )
    # Cap at 100 tests (best effort)
    if len(multi_feasible) > 100:
        random.seed(args.seed + 2)
        multi_feasible = random.sample(multi_feasible, 100)
        mf_stats['capped_at'] = 100
    results['categories']['multi_feasible'] = {
        'tests': multi_feasible,
        'count': len(multi_feasible),
        'stats': mf_stats
    }
    print(f"  Selected {len(multi_feasible)} tests from {len(mf_stats.get('selected_home_ids', []))} homes")

    # Category 4: multi_mixed (100 tests)
    print("\nSelecting multi_mixed tests...")
    multi_mixed, mm_stats = select_category(
        all_tests, is_multi_mixed, is_multi=True,
        num_homes=10, tests_per_home=10, min_instances=50, seed=args.seed + 3,
        category_name='multi_mixed', from_input=False, target_total=100
    )
    results['categories']['multi_mixed'] = {
        'tests': multi_mixed,
        'count': len(multi_mixed),
        'stats': mm_stats
    }
    print(f"  Selected {len(multi_mixed)} tests from {len(mm_stats.get('selected_home_ids', []))} homes")

    # Summary
    total_selected = (
        len(single_feasible) + len(single_unfeasible) +
        len(multi_feasible) + len(multi_mixed)
    )
    results['metadata']['total_selected'] = total_selected

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<20} {'Tests':>6} {'Homes':>6} {'Target':>8}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*8}")
    print(f"{'single_feasible':<20} {len(single_feasible):>6} {len(sf_stats.get('selected_home_ids', [])):>6} {'100/10':>8}")
    print(f"{'single_unfeasible':<20} {len(single_unfeasible):>6} {len(su_stats.get('selected_home_ids', [])):>6} {'100/10':>8}")
    print(f"{'multi_feasible':<20} {len(multi_feasible):>6} {len(mf_stats.get('selected_home_ids', [])):>6} {'100/10':>8}")
    print(f"{'multi_mixed':<20} {len(multi_mixed):>6} {len(mm_stats.get('selected_home_ids', [])):>6} {'100/10':>8}")
    print(f"{'='*60}")
    print(f"{'TOTAL':<20} {total_selected:>6}")

    # Note about constraints
    if len(mf_stats.get('selected_home_ids', [])) > 10:
        print(f"\nNote: multi_feasible required {len(mf_stats.get('selected_home_ids', []))} homes")
        print(f"      (only {mf_stats.get('total_category_tests', 0)} tests available in category)")

    if args.stats:
        print(f"\n{'='*60}")
        print("DETAILED STATISTICS")
        print(f"{'='*60}")
        for cat_name, cat_data in results['categories'].items():
            stats = cat_data['stats']
            print(f"\n{cat_name}:")
            print(f"  Total tests in category: {stats['total_category_tests']}")
            print(f"  Total homes: {stats['total_homes']}")
            print(f"  Homes with >= 50 instances: {stats['eligible_homes']}")
            print(f"  Selected homes: {stats.get('selected_home_ids', [])}")
            if stats.get('relaxed_constraint'):
                print("  (Constraint relaxed due to insufficient eligible homes)")

            # Print tests per home
            if 'tests_per_selected_home' in stats:
                print(f"  Tests available per selected home:")
                for h, c in sorted(stats['tests_per_selected_home'].items(), key=lambda x: -x[1]):
                    print(f"    home{h}: {c}")

            # Print verification
            if 'verification' in stats:
                v = stats['verification']
                print(f"  Constraint verification:")
                print(f"    Tests with same device/room pair: {v['same_device_room_pairs']} pairs found")
                if 'exact_same_device_pairs' in v:
                    print(f"    Tests with exact same devices: {v['exact_same_device_pairs']} pairs found")
                if 'by_command_count' in v:
                    print(f"    By command count (target: 2cmd=5, 3cmd=3, 4+cmd=2):")
                    for k, cnt in v['by_command_count'].items():
                        target = v['target_command_counts'].get(k, '?')
                        status = "✓" if cnt >= target else "✗"
                        print(f"      {k} commands: {cnt} (target: {target}) {status}")

    # Save results -- per-category files
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_categories_statistics = {}

    for cat_name, cat_data in results['categories'].items():
        tests = cat_data['tests']

        # Save per-category test file
        test_file = output_dir / f"selected_test_{cat_name}.json"
        with open(test_file, 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"\n  {test_file}")

        # Compute and save per-category statistics
        cat_statistics = compute_category_statistics(tests, cat_name)
        stats_file = output_dir / f"{cat_name}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(cat_statistics, f, indent=2)
        print(f"  {stats_file}")

        all_categories_statistics[cat_name] = cat_statistics

    # Save combined statistics
    all_stats_file = output_dir / "all_categories_statistics.json"
    with open(all_stats_file, 'w') as f:
        json.dump(all_categories_statistics, f, indent=2)
    print(f"\n  {all_stats_file}")

    return results


if __name__ == '__main__':
    main()
