#!/usr/bin/env python3
"""
Test suite for Smart Home to ThingDescription Converter

Validates core conversion functionality and edge cases.
"""

import json
import sys
import os
from pathlib import Path

# Import the converter
sys.path.insert(0, os.path.dirname(__file__))
from smart_home_to_td_converter import SmartHomeToTDConverter


class TestConverter:
    """Test suite for the converter"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.converter = SmartHomeToTDConverter()
    
    def assert_true(self, condition, test_name, message=""):
        """Assert that condition is true"""
        if condition:
            print(f"✓ PASS: {test_name}")
            self.tests_passed += 1
        else:
            print(f"✗ FAIL: {test_name}")
            if message:
                print(f"  {message}")
            self.tests_failed += 1
    
    def test_camel_case_conversion(self):
        """Test camelCase naming convention"""
        result = self.converter.to_camel_case("master_bedroom", "light")
        self.assert_true(result == "masterBedroomLight", 
                        "CamelCase conversion for master_bedroom + light",
                        f"Expected 'masterBedroomLight', got '{result}'")
        
        result = self.converter.to_camel_case("living_room", "air_conditioner")
        self.assert_true(result == "livingRoomAirConditioner",
                        "CamelCase conversion for living_room + air_conditioner",
                        f"Expected 'livingRoomAirConditioner', got '{result}'")
    
    def test_operation_to_action_name(self):
        """Test operation name conversion"""
        result = self.converter.operation_to_action_name("turn_on")
        self.assert_true(result == "turnOn",
                        "Operation name conversion for turn_on",
                        f"Expected 'turnOn', got '{result}'")
        
        result = self.converter.operation_to_action_name("set_brightness")
        self.assert_true(result == "setBrightness",
                        "Operation name conversion for set_brightness",
                        f"Expected 'setBrightness', got '{result}'")
    
    def test_device_type_class(self):
        """Test device type class generation"""
        result = self.converter.get_device_type_class("light")
        self.assert_true(result == "Light",
                        "Device type class for light",
                        f"Expected 'Light', got '{result}'")
        
        result = self.converter.get_device_type_class("air_conditioner")
        self.assert_true(result == "AirConditioner",
                        "Device type class for air_conditioner",
                        f"Expected 'AirConditioner', got '{result}'")
    
    def test_operation_class(self):
        """Test operation command class generation"""
        result = self.converter.get_operation_class("turn_on")
        self.assert_true(result == "TurnOnCommand",
                        "Operation class for turn_on",
                        f"Expected 'TurnOnCommand', got '{result}'")
        
        result = self.converter.get_operation_class("set_temperature")
        self.assert_true(result == "SetTemperatureCommand",
                        "Operation class for set_temperature",
                        f"Expected 'SetTemperatureCommand', got '{result}'")
    
    def test_basic_conversion(self):
        """Test basic conversion functionality"""
        input_data = {
            "home_id": 1,
            "method": [
                {
                    "room_name": "bedroom",
                    "device_name": "light",
                    "operation": "turn_on",
                    "parameters": []
                }
            ],
            "home_status": {
                "bedroom": {
                    "room_name": "bedroom",
                    "light": {
                        "state": "on",
                        "attributes": {
                            "brightness": {
                                "value": 50,
                                "lowest": 0,
                                "highest": 100
                            }
                        }
                    }
                }
            }
        }
        
        rdf_output, json_state = self.converter.convert(input_data)
        
        # Check RDF output contains expected elements
        self.assert_true("bedroomLight" in rdf_output,
                        "RDF contains artifact name",
                        "Artifact name 'bedroomLight' not found in RDF output")
        
        self.assert_true("td:hasActionAffordance" in rdf_output,
                        "RDF contains action affordance",
                        "Action affordance not found in RDF output")
        
        self.assert_true("td:hasPropertyAffordance" in rdf_output,
                        "RDF contains property affordance",
                        "Property affordance not found in RDF output")
        
        self.assert_true("hmas:Workspace" in rdf_output,
                        "RDF contains workspace",
                        "Workspace not found in RDF output")
        
        # Check JSON state
        artifact_uri = "http://localhost:8080/workspaces/bedroom/artifacts/bedroomLight#artifact"
        self.assert_true(artifact_uri in json_state,
                        "JSON state contains artifact URI",
                        f"Artifact URI not found in JSON state")
        
        if artifact_uri in json_state:
            state = json_state[artifact_uri]
            self.assert_true(state.get("state") == "on",
                            "JSON state has correct state value",
                            f"Expected state 'on', got '{state.get('state')}'")
            
            self.assert_true(state.get("brightness") == 50,
                            "JSON state has correct brightness value",
                            f"Expected brightness 50, got {state.get('brightness')}")
    
    def test_multiple_devices(self):
        """Test conversion with multiple devices in multiple rooms"""
        input_data = {
            "home_id": 2,
            "method": [
                {
                    "room_name": "room1",
                    "device_name": "device1",
                    "operation": "action1",
                    "parameters": []
                },
                {
                    "room_name": "room2",
                    "device_name": "device2",
                    "operation": "action2",
                    "parameters": []
                }
            ],
            "home_status": {
                "room1": {
                    "room_name": "room1",
                    "device1": {
                        "state": "on",
                        "attributes": {}
                    }
                },
                "room2": {
                    "room_name": "room2",
                    "device2": {
                        "state": "off",
                        "attributes": {}
                    }
                }
            }
        }
        
        rdf_output, json_state = self.converter.convert(input_data)
        
        # Check that both artifacts are present
        self.assert_true("room1Device1" in rdf_output,
                        "RDF contains first artifact",
                        "First artifact 'room1Device1' not found")
        
        self.assert_true("room2Device2" in rdf_output,
                        "RDF contains second artifact",
                        "Second artifact 'room2Device2' not found")
        
        # Check that both workspaces are present
        self.assert_true("room1#workspace" in rdf_output,
                        "RDF contains first workspace",
                        "First workspace not found")
        
        self.assert_true("room2#workspace" in rdf_output,
                        "RDF contains second workspace",
                        "Second workspace not found")
        
        # Check JSON state has both artifacts
        self.assert_true(len(json_state) == 2,
                        "JSON state contains both artifacts",
                        f"Expected 2 artifacts, got {len(json_state)}")
    
    def test_parameters_conversion(self):
        """Test parameter conversion to input schemas"""
        input_data = {
            "home_id": 3,
            "method": [
                {
                    "room_name": "room",
                    "device_name": "device",
                    "operation": "set_value",
                    "parameters": [
                        {
                            "name": "value",
                            "type": "int"
                        }
                    ]
                }
            ],
            "home_status": {
                "room": {
                    "room_name": "room",
                    "device": {
                        "state": "on",
                        "attributes": {}
                    }
                }
            }
        }
        
        rdf_output, _ = self.converter.convert(input_data)
        
        self.assert_true("td:hasInputSchema" in rdf_output,
                        "RDF contains input schema",
                        "Input schema not found for parameterized action")
        
        self.assert_true("jsonschema:IntegerSchema" in rdf_output,
                        "RDF contains integer schema for int parameter",
                        "Integer schema not found")
        
        self.assert_true('jsonschema:propertyName "value"' in rdf_output,
                        "RDF contains parameter name",
                        "Parameter name not found in schema")
    
    def test_property_with_range(self):
        """Test property with range constraints"""
        input_data = {
            "home_id": 4,
            "method": [],
            "home_status": {
                "room": {
                    "room_name": "room",
                    "device": {
                        "state": "on",
                        "attributes": {
                            "level": {
                                "value": 50,
                                "lowest": 10,
                                "highest": 90
                            }
                        }
                    }
                }
            }
        }
        
        rdf_output, _ = self.converter.convert(input_data)
        
        self.assert_true("jsonschema:minimum 10" in rdf_output,
                        "RDF contains minimum constraint",
                        "Minimum constraint not found")
        
        self.assert_true("jsonschema:maximum 90" in rdf_output,
                        "RDF contains maximum constraint",
                        "Maximum constraint not found")
    
    def test_property_with_enum(self):
        """Test property with enumerated options"""
        input_data = {
            "home_id": 5,
            "method": [],
            "home_status": {
                "room": {
                    "room_name": "room",
                    "device": {
                        "state": "on",
                        "attributes": {
                            "mode": {
                                "value": "auto",
                                "options": ["auto", "manual", "off"]
                            }
                        }
                    }
                }
            }
        }
        
        rdf_output, _ = self.converter.convert(input_data)
        
        self.assert_true('jsonschema:enum "auto"' in rdf_output,
                        "RDF contains enum option 'auto'",
                        "Enum option 'auto' not found")
        
        self.assert_true('jsonschema:enum "manual"' in rdf_output or '"manual"' in rdf_output,
                        "RDF contains enum option 'manual'",
                        "Enum option 'manual' not found")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("Running Smart Home to TD Converter Tests")
        print("="*60 + "\n")
        
        self.test_camel_case_conversion()
        self.test_operation_to_action_name()
        self.test_device_type_class()
        self.test_operation_class()
        self.test_basic_conversion()
        self.test_multiple_devices()
        self.test_parameters_conversion()
        self.test_property_with_range()
        self.test_property_with_enum()
        
        print("\n" + "="*60)
        print(f"Test Results: {self.tests_passed} passed, {self.tests_failed} failed")
        print("="*60 + "\n")
        
        return self.tests_failed == 0


def main():
    """Main entry point"""
    tester = TestConverter()
    success = tester.run_all_tests()
    
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
