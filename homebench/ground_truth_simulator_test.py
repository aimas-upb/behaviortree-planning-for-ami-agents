#!/usr/bin/env python3
"""
Script to test ground truth data against a running simulator.

This script:
1. Reads a ground truth JSON file
2. Finds a specific request by ID
3. For each successfully executed output, tests the property URL against expected values
4. Returns a JSON report of the validation results
"""

import argparse
import json
import sys
import requests
from typing import Dict, Any, List


def load_ground_truth(file_path: str) -> List[Dict[str, Any]]:
    """Load ground truth JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)


def find_request_by_id(data: List[Dict[str, Any]], request_id: str) -> Dict[str, Any]:
    """Find a request entry by its ID."""
    for entry in data:
        if entry.get("id") == request_id:
            return entry

    print(f"Error: Request ID '{request_id}' not found in ground truth data", file=sys.stderr)
    sys.exit(1)


def get_property_value(property_url: str) -> Any:
    """Make a GET request to the property URL and return the value."""
    try:
        response = requests.get(property_url, timeout=5)
        response.raise_for_status()

        # Try to parse as JSON
        try:
            data = response.json()
            # The property endpoint might return the value directly or in a wrapper
            # Adjust this based on your actual API response format
            if isinstance(data, dict) and 'value' in data:
                return data['value']
            return data
        except json.JSONDecodeError:
            # If not JSON, return the text content
            return response.text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to retrieve property from {property_url}: {e}", file=sys.stderr)
        return None


def test_ground_truth(file_path: str, request_id: str) -> Dict[str, Any]:
    """
    Test ground truth data against the simulator.

    Args:
        file_path: Path to the ground truth JSON file
        request_id: ID of the request to test

    Returns:
        Dictionary with test results
    """
    # Load data and find the request
    data = load_ground_truth(file_path)
    request_entry = find_request_by_id(data, request_id)

    # Initialize result structure
    result = {
        "success_overall": True,
        "detail": {}
    }

    # Process each output entry
    output_list = request_entry.get("output", [])

    for output_entry in output_list:
        # Only process successfully executed entries
        if output_entry.get("execution") != "success":
            continue

        # Check if test key exists
        test_info = output_entry.get("test")
        if not test_info:
            continue

        property_url = test_info.get("property")
        expected_value = test_info.get("expected_value")

        if not property_url:
            continue

        # Get the actual value from the simulator
        retrieved_value = get_property_value(property_url)

        # Compare values
        # Handle different types appropriately
        if retrieved_value is None:
            status = False
            result["success_overall"] = False
        else:
            # Normalize for comparison
            # Convert to same type if possible
            try:
                if isinstance(expected_value, (int, float)) and isinstance(retrieved_value, str):
                    retrieved_value = type(expected_value)(retrieved_value)
                elif isinstance(expected_value, str) and isinstance(retrieved_value, (int, float)):
                    expected_value = str(expected_value)
            except (ValueError, TypeError):
                pass

            status = (retrieved_value == expected_value)
            if not status:
                result["success_overall"] = False

        # Add to detail
        result["detail"][property_url] = {
            "status": status,
            "expected_value": expected_value,
            "retrieved_value": retrieved_value
        }

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test ground truth data against a running simulator",
        epilog="""
Examples:
  %(prog)s -f datasets/HomeBench/converted/test_data.json -i home86_multi_329
  %(prog)s --file test_data.json --id home86_multi_329 --pretty
  %(prog)s -f test_data.json -i home86_multi_329 -o results.json --pretty
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f", "--file",
        dest="ground_truth_file",
        required=True,
        help="Path to the ground truth JSON file (e.g., datasets/HomeBench/converted/test_data.json)"
    )
    parser.add_argument(
        "-i", "--id",
        dest="request_id",
        required=True,
        help="ID of the request to test (e.g., home86_multi_329)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output"
    )

    args = parser.parse_args()

    # Run the test
    result = test_ground_truth(args.ground_truth_file, args.request_id)

    # Format output
    if args.pretty:
        json_output = json.dumps(result, indent=2)
    else:
        json_output = json.dumps(result)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_output)
        print(f"Results written to {args.output}")
    else:
        print(json_output)

    # Exit with appropriate code
    sys.exit(0 if result["success_overall"] else 1)


if __name__ == "__main__":
    main()
