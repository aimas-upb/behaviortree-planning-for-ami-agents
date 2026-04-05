#!/usr/bin/env python3
"""
Test script for the /reset endpoint
"""

import json

import requests

BASE_URL = "http://localhost:8080"


def test_reset_endpoint():
    """Test the reset endpoint"""

    print("Testing /reset endpoint...\n")

    # Test 1: Reset home 0
    print("Test 1: Reset home 0")
    response = requests.post(f"{BASE_URL}/reset", json={"home": "0"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    # Test 2: Reset home 1
    print("Test 2: Reset home 1")
    response = requests.post(f"{BASE_URL}/reset", json={"home": "1"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    # Test 3: Try to reset non-existent home
    print("Test 3: Reset non-existent home (should fail)")
    response = requests.post(f"{BASE_URL}/reset", json={"home": "999"})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    # Test 4: Missing 'home' parameter
    print("Test 4: Missing 'home' parameter (should fail)")
    response = requests.post(f"{BASE_URL}/reset", json={})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

    # Test 5: Invalid JSON payload
    print("Test 5: Invalid JSON payload (should fail)")
    response = requests.post(f"{BASE_URL}/reset", data="invalid json")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_state_reset_verification():
    """Verify that state is actually reset"""

    print("\nVerifying state reset functionality...\n")

    # Step 1: Get initial state of a device
    print("Step 1: Get initial state of master bedroom light in home 0")
    response = requests.get(
        f"{BASE_URL}/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/properties/state"
    )
    initial_state = response.json()
    print(f"Initial state: {initial_state}")
    print()

    # Step 2: Turn the light on if it's off (or vice versa)
    print("Step 2: Toggle the light state")
    if initial_state == "off" or initial_state.get("value") == "off":
        action = "turnOn"
        expected_state = "on"
    else:
        action = "turnOff"
        expected_state = "off"

    response = requests.post(
        f"{BASE_URL}/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/{action}",
        json={},
    )
    print(f"Action response: {response.json()}")

    # Verify state changed
    response = requests.get(
        f"{BASE_URL}/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/properties/state"
    )
    modified_state = response.json()
    print(f"Modified state: {modified_state}")
    print()

    # Step 3: Reset the home
    print("Step 3: Reset home 0")
    response = requests.post(f"{BASE_URL}/reset", json={"home": "0"})
    print(f"Reset response: {response.json()}")
    print()

    # Step 4: Verify state is back to initial
    print("Step 4: Verify state is reset to initial")
    response = requests.get(
        f"{BASE_URL}/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/properties/state"
    )
    reset_state = response.json()
    print(f"Reset state: {reset_state}")

    # Compare
    if reset_state == initial_state:
        print("✓ State successfully reset to initial value!")
    else:
        print("✗ State was NOT reset correctly")
    print()


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("Reset Endpoint Test Suite")
        print("=" * 60)
        print()

        # Run basic endpoint tests
        test_reset_endpoint()

        # Run state verification test
        test_state_reset_verification()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to simulator at http://localhost:8080")
        print(
            "Please ensure the simulator is running before running this test."
        )
    except Exception as e:
        print(f"Error occurred: {e}")
