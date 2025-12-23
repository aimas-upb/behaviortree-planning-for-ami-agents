#!/usr/bin/env python3
"""
HomeBench Ground Truth Converter

Converts HomeBench ground truth from original format to ThingDescription representation.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Any
import rdflib


class TTLParser:
    """Parser for TTL files to extract affordance mappings."""

    def __init__(self, ttl_path: str):
        self.graph = rdflib.Graph()
        self.graph.parse(ttl_path, format="turtle")
        self.affordance_map = self._build_affordance_map()
        self.property_map = self._build_property_map()

    def _build_affordance_map(self) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from device calls to affordance URLs and input schemas."""
        affordance_map = {}

        # Query for all action affordances with their targets
        query = """
        PREFIX td: <https://www.w3.org/2019/wot/td#>
        PREFIX hctl: <https://www.w3.org/2019/wot/hypermedia#>
        PREFIX jsonschema: <https://www.w3.org/2019/wot/json-schema#>

        SELECT ?artifact ?actionName ?target ?schema
        WHERE {
            ?artifact td:hasActionAffordance ?action .
            ?action td:name ?actionName .
            ?action td:hasForm ?form .
            ?form hctl:hasTarget ?target .
            OPTIONAL {
                ?action td:hasInputSchema ?schema .
            }
        }
        """

        results = self.graph.query(query)

        for row in results:
            target_url = str(row.target)

            # Extract room, artifact, and action from URL
            # Format: http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight/turn_off
            pattern = r'/workspaces/home\d+/([^/]+)/artifacts/([^/]+)/(.+)$'
            match = re.search(pattern, target_url)

            if match:
                room = match.group(1)
                artifact_name = match.group(2)
                action = match.group(3)

                # Convert camelCase artifact name to understand device type
                # e.g., livingRoomLight -> light
                device_type = self._extract_device_type(artifact_name)

                # Build key: room.device.action
                key = f"{room}.{device_type}.{action}"

                # Parse input schema if exists
                params_schema = {}
                if row.schema:
                    params_schema = self._parse_input_schema(row.schema)

                affordance_map[key] = {
                    'url': target_url,
                    'params_schema': params_schema
                }

        return affordance_map

    def _extract_device_type(self, artifact_name: str) -> str:
        """Extract device type from artifact name (e.g., livingRoomLight -> light)."""
        # Convert camelCase to snake_case and extract device type
        # Remove common room prefixes
        room_prefixes = [
            'livingRoom', 'masterBedroom', 'guestBedroom', 'studyRoom',
            'storeRoom', 'diningRoom', 'balcony', 'bathroom', 'corridor',
            'foyer', 'garage', 'kitchen'
        ]

        device_name = artifact_name
        for prefix in room_prefixes:
            if artifact_name.startswith(prefix):
                device_name = artifact_name[len(prefix):]
                break

        # Convert from camelCase to snake_case
        # Insert underscore before capitals and convert to lowercase
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', device_name).lower()

        return snake_case

    def _parse_input_schema(self, schema_node) -> Dict[str, str]:
        """Parse JSON schema to extract parameter names and types."""
        params_schema = {}

        query = """
        PREFIX jsonschema: <https://www.w3.org/2019/wot/json-schema#>

        SELECT ?propName ?propType ?min ?max ?required
        WHERE {
            ?schema jsonschema:properties ?prop .
            ?prop jsonschema:propertyName ?propName .
            OPTIONAL { ?prop a ?propType }
            OPTIONAL { ?prop jsonschema:minimum ?min }
            OPTIONAL { ?prop jsonschema:maximum ?max }
            OPTIONAL { ?schema jsonschema:required ?required }
        }
        """

        # Create a temporary graph for this schema
        temp_graph = rdflib.Graph()
        for s, p, o in self.graph.triples((schema_node, None, None)):
            temp_graph.add((s, p, o))
            for s2, p2, o2 in self.graph.triples((o, None, None)):
                temp_graph.add((s2, p2, o2))

        results = temp_graph.query(query, initBindings={'schema': schema_node})

        for row in results:
            if row.propName:
                param_info = {
                    'name': str(row.propName),
                    'type': str(row.propType) if row.propType else 'unknown'
                }
                if row.min is not None:
                    param_info['min'] = int(row.min)
                if row.max is not None:
                    param_info['max'] = int(row.max)
                params_schema[str(row.propName)] = param_info

        return params_schema

    def _build_property_map(self) -> Dict[str, str]:
        """Build a mapping from artifacts to their property URLs."""
        property_map = {}

        # Query for all property affordances with their targets
        query = """
        PREFIX td: <https://www.w3.org/2019/wot/td#>
        PREFIX hctl: <https://www.w3.org/2019/wot/hypermedia#>

        SELECT ?artifact ?propName ?target
        WHERE {
            ?artifact td:hasPropertyAffordance ?property .
            ?property td:name ?propName .
            ?property td:hasForm ?form .
            ?form hctl:hasTarget ?target .
        }
        """

        results = self.graph.query(query)

        for row in results:
            target_url = str(row.target)

            # Extract artifact URL and property name from target
            # Format: http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight/properties/state
            pattern = r'(/workspaces/home\d+/[^/]+/artifacts/[^/]+)/properties/(.+)$'
            match = re.search(pattern, target_url)

            if match:
                artifact_base = match.group(1)
                property_name = match.group(2)

                # Store mapping: artifact_base.property_name -> property_url
                key = f"{artifact_base}.{property_name}"
                property_map[key] = target_url

        return property_map

    def find_affordance(self, room: str, device: str, action: str) -> Optional[Dict[str, Any]]:
        """Find affordance URL and schema for a given room.device.action."""
        key = f"{room}.{device}.{action}"
        return self.affordance_map.get(key)

    def find_property_url(self, artifact_url: str, property_name: str) -> Optional[str]:
        """Find property URL for a given artifact and property name."""
        key = f"{artifact_url}.{property_name}"
        return self.property_map.get(key)

    def get_artifact_base_url(self, affordance_url: str) -> Optional[str]:
        """Extract artifact base URL from affordance URL."""
        # Format: http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight/turn_off
        # Extract: http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight
        pattern = r'(/workspaces/home\d+/[^/]+/artifacts/[^/]+)/[^/]+$'
        match = re.search(pattern, affordance_url)
        if match:
            return match.group(1)
        return None


class GroundTruthConverter:
    """Converts HomeBench ground truth to ThingDescription format."""

    def __init__(self, hmas_format_dir: str):
        self.hmas_format_dir = Path(hmas_format_dir)
        self.ttl_parsers: Dict[int, TTLParser] = {}

    def _get_ttl_parser(self, home_id: int) -> TTLParser:
        """Get or create TTL parser for a home."""
        if home_id not in self.ttl_parsers:
            ttl_path = self.hmas_format_dir / f"home_{home_id}.ttl"
            if not ttl_path.exists():
                raise FileNotFoundError(f"TTL file not found: {ttl_path}")
            self.ttl_parsers[home_id] = TTLParser(str(ttl_path))
        return self.ttl_parsers[home_id]

    def _parse_action_call(self, call_str: str) -> Optional[Dict[str, Any]]:
        """Parse an action call string like 'living_room.light.turn_off()'."""
        # Pattern: room.device.action(params)
        pattern = r'([^.]+)\.([^.]+)\.([^(]+)\(([^)]*)\)'
        match = re.match(pattern, call_str.strip())

        if not match:
            return None

        room = match.group(1)
        device = match.group(2)
        action = match.group(3)
        params_str = match.group(4)

        # Parse parameters
        params = {}
        if params_str:
            # Handle simple parameters like "60" or "auto" or "intensity=60"
            if '=' in params_str:
                # Named parameter
                for param in params_str.split(','):
                    key, value = param.strip().split('=')
                    params[key.strip()] = self._parse_value(value.strip())
            else:
                # Positional parameter - we need to infer the name
                # This will be handled later when we have the schema
                params['_positional'] = self._parse_value(params_str.strip())

        return {
            'room': room,
            'device': device,
            'action': action,
            'params': params
        }

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate Python type."""
        # Try to parse as int
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string, removing quotes if present
        return value_str.strip('\'"')

    def _extract_param_name_from_schema(self, schema: Dict[str, Any]) -> Optional[str]:
        """Extract the first parameter name from schema."""
        if schema and len(schema) > 0:
            return list(schema.keys())[0]
        return None

    def _determine_test_info(
        self,
        parser: TTLParser,
        affordance_url: str,
        action: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Determine which property to test and what value to expect."""
        # Get artifact base URL
        artifact_url = parser.get_artifact_base_url(affordance_url)
        if not artifact_url:
            return None

        # Map actions to properties and expected values
        # Common action patterns:
        # - turn_on -> state: "on"
        # - turn_off -> state: "off"
        # - set_X(value) -> X: value
        # - open -> state: "on"
        # - close -> state: "off"

        property_name = None
        expected_value = None

        if action == 'turn_on':
            property_name = 'state'
            expected_value = 'on'
        elif action == 'turn_off':
            property_name = 'state'
            expected_value = 'off'
        elif action == 'open':
            property_name = 'state'
            expected_value = 'on'
        elif action == 'close':
            property_name = 'state'
            expected_value = 'off'
        elif action.startswith('set_'):
            # Extract property name from action (e.g., set_temperature -> temperature)
            property_name = action[4:]  # Remove 'set_' prefix
            # The expected value is the parameter value
            if params:
                # Get the first (and usually only) parameter value
                expected_value = list(params.values())[0] if params else None

        if property_name and expected_value is not None:
            # Find the property URL
            property_url = parser.find_property_url(artifact_url, property_name)
            if property_url:
                return {
                    'property': property_url,
                    'expected_value': expected_value
                }

        return None

    def convert_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single ground truth entry."""
        entry_id = entry['id']
        input_text = entry['input']
        output_text = entry['output']

        # Extract home_id from entry id (e.g., "home76_multi_201" -> 76)
        home_id_match = re.match(r'home(\d+)_', entry_id)
        if not home_id_match:
            raise ValueError(f"Cannot extract home_id from entry id: {entry_id}")

        home_id = int(home_id_match.group(1))

        # Get TTL parser for this home
        parser = self._get_ttl_parser(home_id)

        # Parse output text
        # Format: "'''error_input,living_room.light.turn_off(),error_input,'''"
        output_text = output_text.strip().strip("'")

        # Split by comma and process each action
        actions = [action.strip() for action in output_text.split(',') if action.strip()]

        converted_output = []

        for action in actions:
            if action == 'error_input':
                converted_output.append({
                    'execution': 'error_input'
                })
            else:
                # Parse the action call
                parsed = self._parse_action_call(action)

                if parsed:
                    # Find the affordance
                    affordance_info = parser.find_affordance(
                        parsed['room'],
                        parsed['device'],
                        parsed['action']
                    )

                    if affordance_info:
                        # Handle positional parameters
                        params = parsed['params']
                        if '_positional' in params and affordance_info['params_schema']:
                            param_name = self._extract_param_name_from_schema(
                                affordance_info['params_schema']
                            )
                            if param_name:
                                params = {param_name: params['_positional']}
                            else:
                                params = {}
                        elif '_positional' in params:
                            params = {}

                        # Build the output entry
                        output_entry = {
                            'execution': 'success',
                            'affordance': affordance_info['url'],
                            'params': params
                        }

                        # Determine test information
                        test_info = self._determine_test_info(
                            parser,
                            affordance_info['url'],
                            parsed['action'],
                            params
                        )

                        if test_info:
                            output_entry['test'] = test_info

                        converted_output.append(output_entry)
                    else:
                        # Affordance not found, treat as error
                        converted_output.append({
                            'execution': 'error_input'
                        })
                else:
                    # Failed to parse action
                    converted_output.append({
                        'execution': 'error_input'
                    })

        return {
            'id': entry_id,
            'input': input_text,
            'output': converted_output
        }

    def convert_file(self, input_file: str, output_file: str):
        """Convert a JSONL file to JSON format."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read JSONL file
        entries = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Convert each entry
        converted_entries = []
        for i, entry in enumerate(entries):
            try:
                converted = self.convert_entry(entry)
                converted_entries.append(converted)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(entries)} entries...")
            except Exception as e:
                print(f"Error processing entry {entry.get('id', 'unknown')}: {e}")
                # Add entry with error marker
                converted_entries.append({
                    'id': entry.get('id', 'unknown'),
                    'input': entry.get('input', ''),
                    'output': [{'execution': 'error_input'}],
                    'error': str(e)
                })

        # Write output JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(converted_entries, f, indent=2)

        print(f"Conversion complete. Wrote {len(converted_entries)} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HomeBench ground truth to ThingDescription format'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input JSONL file (e.g., datasets/HomeBench/original/train_data_part1.jsonl)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '-t', '--ttl-dir',
        default='datasets/HomeBench/hmas_format',
        help='Directory containing TTL files (default: datasets/HomeBench/hmas_format)'
    )

    args = parser.parse_args()

    converter = GroundTruthConverter(args.ttl_dir)
    converter.convert_file(args.input, args.output)


if __name__ == '__main__':
    main()
