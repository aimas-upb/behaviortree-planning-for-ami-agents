#!/usr/bin/env python3
"""
Smart Home to ThingDescription Artifact Converter

Converts smart home state JSON into:
1. TD artifact-based RDF/Turtle representation (Hypermedia Environment)
2. Hierarchical workspace structure: Home -> Rooms -> Artifacts

Author: Dataset Preparation Script
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal, URIRef, BNode


class SmartHomeToTDConverter:
    """Converts smart home JSON to TD artifact format using RDFLib"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

        # Define namespaces
        self.EX = Namespace("http://example.org/")
        self.HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
        self.HMAS = Namespace("https://purl.org/hmas/")
        self.HTTP = Namespace("http://www.w3.org/2011/http#")
        self.JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")
        self.TD = Namespace("https://www.w3.org/2019/wot/td#")

    def sanitize_name(self, name: str) -> str:
        """Sanitize a name by removing/replacing invalid URI characters"""
        # Strip leading/trailing whitespace
        name = name.strip()
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Remove any other problematic characters
        name = re.sub(r'[^\w\-_]', '', name)
        return name

    def to_camel_case(self, room_name: str, device_name: str) -> str:
        """Convert room and device names to camelCase artifact name"""
        # Sanitize inputs first
        room_name = self.sanitize_name(room_name)
        device_name = self.sanitize_name(device_name)

        # Convert room_name: master_bedroom -> masterBedroom
        room_parts = room_name.split('_')
        room_camel = room_parts[0].lower() + ''.join(word.capitalize() for word in room_parts[1:])

        # Convert device_name: air_conditioner -> AirConditioner
        device_parts = device_name.split('_')
        device_camel = ''.join(word.capitalize() for word in device_parts)

        return f"{room_camel}{device_camel}"

    def operation_to_action_name(self, operation: str) -> str:
        """Convert operation name to action affordance name"""
        # Sanitize first
        operation = self.sanitize_name(operation)
        # turn_on -> turnOn, set_brightness -> setBrightness
        parts = operation.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    def get_device_type_class(self, device_name: str) -> str:
        """Get the device type class (capitalized)"""
        device_name = self.sanitize_name(device_name)
        parts = device_name.split('_')
        return ''.join(word.capitalize() for word in parts)

    def get_operation_class(self, operation: str) -> str:
        """Get the operation command class"""
        operation = self.sanitize_name(operation)
        parts = operation.split('_')
        class_name = ''.join(word.capitalize() for word in parts)
        return f"{class_name}Command"

    def add_input_schema(self, g: Graph, action_node: BNode, parameters: List[Dict],
                         property_constraints: Dict[str, Dict] = None):
        """Add JSON Schema for input parameters to the graph with property constraints"""
        if not parameters:
            return

        if property_constraints is None:
            property_constraints = {}

        # Create input schema blank node
        input_schema = BNode()
        g.add((action_node, self.TD.hasInputSchema, input_schema))
        g.add((input_schema, RDF.type, self.JSONSCHEMA.ObjectSchema))

        for param in parameters:
            param_name = param['name']
            param_type = param['type']

            # Determine schema type
            if param_type == 'int':
                schema_type = self.JSONSCHEMA.IntegerSchema
            elif param_type == 'str':
                schema_type = self.JSONSCHEMA.StringSchema
            elif param_type == 'bool':
                schema_type = self.JSONSCHEMA.BooleanSchema
            else:
                schema_type = self.JSONSCHEMA.StringSchema

            # Create property blank node
            prop_node = BNode()
            g.add((input_schema, self.JSONSCHEMA.properties, prop_node))
            g.add((prop_node, RDF.type, schema_type))
            # Use original name for the property name value
            g.add((prop_node, self.JSONSCHEMA.propertyName, Literal(param_name)))
            g.add((input_schema, self.JSONSCHEMA.required, Literal(param_name)))

            # Apply property constraints if they exist for this parameter
            if param_name in property_constraints:
                constraints = property_constraints[param_name]

                # Handle array types
                if constraints.get('is_array'):
                    # Change schema type to ArraySchema
                    # Remove the previously added type
                    g.remove((prop_node, RDF.type, schema_type))
                    g.add((prop_node, RDF.type, self.JSONSCHEMA.ArraySchema))

                    # Add items schema based on item_type
                    item_type = constraints.get('item_type', 'int')
                    item_schema = BNode()
                    g.add((prop_node, self.JSONSCHEMA.items, item_schema))

                    if item_type == 'int':
                        g.add((item_schema, RDF.type, self.JSONSCHEMA.IntegerSchema))
                    elif item_type == 'str':
                        g.add((item_schema, RDF.type, self.JSONSCHEMA.StringSchema))
                    elif item_type == 'bool':
                        g.add((item_schema, RDF.type, self.JSONSCHEMA.BooleanSchema))
                else:
                    # Add enum constraint
                    if 'enum' in constraints:
                        for enum_value in constraints['enum']:
                            g.add((prop_node, self.JSONSCHEMA['enum'], Literal(enum_value)))

                    # Add min/max constraints for numeric types
                    if 'minimum' in constraints and param_type == 'int':
                        g.add((prop_node, self.JSONSCHEMA.minimum, Literal(constraints['minimum'])))

                    if 'maximum' in constraints and param_type == 'int':
                        g.add((prop_node, self.JSONSCHEMA.maximum, Literal(constraints['maximum'])))

    def add_property_affordance(self, g: Graph, artifact_uri: URIRef, property_name: str,
                                property_data: Dict, workspace_id: str, home_id: str,
                                artifact_name: str):
        """Add PropertyAffordance to the graph"""
        # Sanitize property name for use in URIs
        property_name_sanitized = self.sanitize_name(property_name)

        # Create property affordance blank node
        prop_node = BNode()
        g.add((artifact_uri, self.TD.hasPropertyAffordance, prop_node))
        g.add((prop_node, RDF.type, self.TD.PropertyAffordance))
        # Use original name in literals
        g.add((prop_node, RDFS.comment, Literal(f"{property_name} of {artifact_name}")))
        g.add((prop_node, self.TD.name, Literal(property_name)))
        g.add((prop_node, self.TD.title, Literal(property_name)))
        g.add((prop_node, self.TD.isObservable, Literal(True)))

        # Property read form (use sanitized name in URL)
        property_url = f"{self.base_url}/workspaces/home{home_id}/{workspace_id}/artifacts/{artifact_name}/properties/{property_name_sanitized}"
        form_node = BNode()
        g.add((prop_node, self.TD.hasForm, form_node))
        g.add((form_node, self.HTTP.methodName, Literal("GET")))
        g.add((form_node, self.HCTL.forContentType, Literal("application/json")))
        g.add((form_node, self.HCTL.hasOperationType, self.TD.readProperty))
        g.add((form_node, self.HCTL.hasTarget, URIRef(property_url)))

        # Output schema based on value type and constraints
        output_schema = BNode()
        g.add((prop_node, self.TD.hasOutputSchema, output_schema))

        value = property_data.get('value')

        # Check if it has options (enum)
        if 'options' in property_data:
            g.add((output_schema, RDF.type, self.JSONSCHEMA.StringSchema))
            for option in property_data['options']:
                g.add((output_schema, self.JSONSCHEMA['enum'], Literal(option)))
        # Check if it has range (lowest/highest)
        elif 'lowest' in property_data and 'highest' in property_data:
            g.add((output_schema, RDF.type, self.JSONSCHEMA.IntegerSchema))
            g.add((output_schema, self.JSONSCHEMA.minimum, Literal(property_data['lowest'])))
            g.add((output_schema, self.JSONSCHEMA.maximum, Literal(property_data['highest'])))
        # Check if value is an array (list)
        elif isinstance(value, list):
            g.add((output_schema, RDF.type, self.JSONSCHEMA.ArraySchema))
            # Determine item type from first element if available
            if value:
                if isinstance(value[0], int):
                    item_schema = BNode()
                    g.add((output_schema, self.JSONSCHEMA.items, item_schema))
                    g.add((item_schema, RDF.type, self.JSONSCHEMA.IntegerSchema))
                elif isinstance(value[0], str):
                    item_schema = BNode()
                    g.add((output_schema, self.JSONSCHEMA.items, item_schema))
                    g.add((item_schema, RDF.type, self.JSONSCHEMA.StringSchema))
                elif isinstance(value[0], bool):
                    item_schema = BNode()
                    g.add((output_schema, self.JSONSCHEMA.items, item_schema))
                    g.add((item_schema, RDF.type, self.JSONSCHEMA.BooleanSchema))
        # Infer type from value
        elif isinstance(value, int):
            g.add((output_schema, RDF.type, self.JSONSCHEMA.IntegerSchema))
        elif isinstance(value, bool):
            g.add((output_schema, RDF.type, self.JSONSCHEMA.BooleanSchema))
        elif isinstance(value, str):
            g.add((output_schema, RDF.type, self.JSONSCHEMA.StringSchema))
        else:
            g.add((output_schema, RDF.type, self.JSONSCHEMA.StringSchema))

    def add_action_affordance(self, g: Graph, artifact_uri: URIRef, operation: str,
                             parameters: List[Dict], workspace_id: str, home_id: str,
                             artifact_name: str, property_constraints: Dict[str, Dict] = None):
        """Add ActionAffordance to the graph"""
        action_name = self.operation_to_action_name(operation)
        operation_class = self.get_operation_class(operation)
        operation_sanitized = self.sanitize_name(operation)

        # Create action affordance blank node
        action_node = BNode()
        g.add((artifact_uri, self.TD.hasActionAffordance, action_node))
        g.add((action_node, RDF.type, self.EX[operation_class]))
        g.add((action_node, RDF.type, self.TD.ActionAffordance))
        g.add((action_node, self.TD.name, Literal(action_name)))
        g.add((action_node, self.TD.title, Literal(action_name)))

        # Action form (use sanitized operation name in URL)
        action_url = f"{self.base_url}/workspaces/home{home_id}/{workspace_id}/artifacts/{artifact_name}/{operation_sanitized}"
        form_node = BNode()
        g.add((action_node, self.TD.hasForm, form_node))
        g.add((form_node, self.HTTP.methodName, Literal("POST")))
        g.add((form_node, self.HCTL.forContentType, Literal("application/json")))
        g.add((form_node, self.HCTL.hasOperationType, self.TD.invokeAction))
        g.add((form_node, self.HCTL.hasTarget, URIRef(action_url)))

        # Input schema if parameters exist
        if parameters:
            self.add_input_schema(g, action_node, parameters, property_constraints or {})

    def add_artifact(self, g: Graph, workspace_id: str, home_id: str, artifact_name: str,
                    device_name: str, methods: List[Dict], device_state: Dict) -> URIRef:
        """Add a TD artifact to the graph"""
        artifact_uri = URIRef(f"{self.base_url}/workspaces/home{home_id}/{workspace_id}/artifacts/{artifact_name}#artifact")
        room_workspace_uri = URIRef(f"{self.base_url}/workspaces/home{home_id}/{workspace_id}#workspace")
        device_class = self.get_device_type_class(device_name)

        # Add artifact triples
        g.add((artifact_uri, RDF.type, self.EX[device_class]))
        g.add((artifact_uri, RDF.type, self.HMAS.Artifact))
        g.add((artifact_uri, RDF.type, self.TD.Thing))
        g.add((artifact_uri, self.HMAS.isContainedIn, room_workspace_uri))
        g.add((artifact_uri, self.TD.title, Literal(artifact_name.capitalize())))

        # Build property constraints map for action input schema validation
        property_constraints = {}

        # Collect constraints from device state attributes
        if 'attributes' in device_state:
            for prop_name, prop_data in device_state['attributes'].items():
                constraints = {}
                value = prop_data.get('value')

                if 'options' in prop_data:
                    constraints['enum'] = prop_data['options']
                if 'lowest' in prop_data and 'highest' in prop_data:
                    constraints['minimum'] = prop_data['lowest']
                    constraints['maximum'] = prop_data['highest']
                # Track if it's an array type
                if isinstance(value, list):
                    constraints['is_array'] = True
                    if value:
                        # Determine item type from first element
                        if isinstance(value[0], int):
                            constraints['item_type'] = 'int'
                        elif isinstance(value[0], str):
                            constraints['item_type'] = 'str'
                        elif isinstance(value[0], bool):
                            constraints['item_type'] = 'bool'

                if constraints:
                    property_constraints[prop_name] = constraints

        # Add state constraint
        if 'state' in device_state:
            property_constraints['state'] = {'enum': ['on', 'off']}

        # Add action affordances with property constraints
        for method in methods:
            self.add_action_affordance(
                g, artifact_uri, method['operation'], method['parameters'],
                workspace_id, home_id, artifact_name, property_constraints
            )

        # Add property affordances from device state
        if 'attributes' in device_state:
            for prop_name, prop_data in device_state['attributes'].items():
                self.add_property_affordance(
                    g, artifact_uri, prop_name, prop_data,
                    workspace_id, home_id, artifact_name
                )

        # Add state property if exists
        if 'state' in device_state:
            state_data = {
                'value': device_state['state'],
                'options': ['on', 'off']
            }
            self.add_property_affordance(
                g, artifact_uri, 'state', state_data,
                workspace_id, home_id, artifact_name
            )

        return artifact_uri

    def add_room_workspace(self, g: Graph, workspace_id: str, home_id: str,
                          artifact_uris: List[URIRef]) -> URIRef:
        """Add a room workspace to the graph"""
        workspace_uri = URIRef(f"{self.base_url}/workspaces/home{home_id}/{workspace_id}#workspace")

        g.add((workspace_uri, RDF.type, self.HMAS.Workspace))
        g.add((workspace_uri, RDF.type, self.TD.Thing))

        for artifact_uri in artifact_uris:
            g.add((workspace_uri, self.HMAS.contains, artifact_uri))

        return workspace_uri

    def add_home_workspace(self, g: Graph, home_id: str, room_workspace_uris: List[URIRef]):
        """Add a home workspace to the graph"""
        home_workspace_uri = URIRef(f"{self.base_url}/workspaces/home{home_id}#workspace")

        g.add((home_workspace_uri, RDF.type, self.HMAS.Workspace))
        g.add((home_workspace_uri, RDF.type, self.TD.Thing))
        g.add((home_workspace_uri, self.TD.title, Literal(f"Home {home_id}")))

        for room_workspace_uri in room_workspace_uris:
            g.add((home_workspace_uri, self.HMAS.contains, room_workspace_uri))

    def extract_json_state(self, artifact_uri: str, device_state: Dict) -> Dict:
        """Extract JSON state representation using PropertyAffordance names"""
        state = {}

        # Add state if exists
        if 'state' in device_state:
            state['state'] = device_state['state']

        # Add attributes (sanitize property names)
        if 'attributes' in device_state:
            for prop_name, prop_data in device_state['attributes'].items():
                # Use sanitized property name as key
                sanitized_prop_name = self.sanitize_name(prop_name)
                state[sanitized_prop_name] = prop_data['value']

        return {artifact_uri: state}

    def convert_home(self, input_data: Dict) -> tuple[Graph, Dict]:
        """
        Convert a single home's data to TD format

        Returns:
            tuple: (RDF Graph, JSON state dict)
        """
        home_id = input_data.get('home_id')
        if home_id is None:
            raise ValueError("Input data must contain 'home_id'")

        methods = input_data.get('method', [])
        home_status = input_data.get('home_status', {})

        # Create RDF graph
        g = Graph()

        # Bind namespaces
        g.bind("ex", self.EX)
        g.bind("hctl", self.HCTL)
        g.bind("hmas", self.HMAS)
        g.bind("http", self.HTTP)
        g.bind("jsonschema", self.JSONSCHEMA)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("td", self.TD)
        g.bind("xsd", XSD)
        g.bind("", Namespace(f"{self.base_url}/workspaces/"))

        # Group methods by room and device
        methods_by_room_device = {}
        for method in methods:
            room = method['room_name']
            device = method['device_name']
            key = (room, device)
            if key not in methods_by_room_device:
                methods_by_room_device[key] = []
            methods_by_room_device[key].append(method)

        # Track room workspaces
        room_workspace_uris = []
        json_state = {}

        # Process each room
        for room_name, room_data in home_status.items():
            # Sanitize room name for use in URIs
            workspace_id = self.sanitize_name(room_name)
            artifact_uris = []

            for device_name, device_state in room_data.items():
                if device_name == 'room_name':
                    continue

                # Generate artifact name
                artifact_name = self.to_camel_case(room_name, device_name)

                # Get methods for this device
                device_methods = methods_by_room_device.get((room_name, device_name), [])

                # Add artifact to graph
                artifact_uri = self.add_artifact(
                    g, workspace_id, home_id, artifact_name,
                    device_name, device_methods, device_state
                )
                artifact_uris.append(artifact_uri)

                # Extract JSON state
                artifact_state = self.extract_json_state(str(artifact_uri), device_state)
                json_state.update(artifact_state)

            # Add room workspace
            if artifact_uris:
                room_workspace_uri = self.add_room_workspace(
                    g, workspace_id, home_id, artifact_uris
                )
                room_workspace_uris.append(room_workspace_uri)

        # Add home workspace
        self.add_home_workspace(g, home_id, room_workspace_uris)

        return g, json_state

    def convert(self, input_data: Any) -> Dict[str, tuple[Graph, Dict]]:
        """
        Convert smart home JSON to TD format
        Handles both single home and multiple homes

        Returns:
            dict: {home_id: (RDF Graph, JSON state dict)}
        """
        results = {}

        # Check if input is a list of homes or a single home
        if isinstance(input_data, list):
            # Multiple homes
            for home_data in input_data:
                home_id = home_data.get('home_id')
                if home_id is None:
                    print(f"Warning: Skipping home without home_id", file=sys.stderr)
                    continue
                g, state = self.convert_home(home_data)
                results[str(home_id)] = (g, state)
        elif isinstance(input_data, dict):
            # Single home
            home_id = input_data.get('home_id')
            if home_id is None:
                raise ValueError("Input data must contain 'home_id'")
            g, state = self.convert_home(input_data)
            results[str(home_id)] = (g, state)
        else:
            raise ValueError("Input data must be a dict or list of dicts")

        return results


def main():
    """Main entry point for the converter script"""
    parser = argparse.ArgumentParser(
        description="Convert smart home state JSON into TD artifact-based RDF/Turtle and JSON state representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smart_home_to_td_converter.py -i input.json -o output_dir/
  python smart_home_to_td_converter.py -i input.json -o output_dir/ --base-url http://192.168.1.100:8080
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='FILE',
        help='Input JSON file containing smart home configuration (single home or list of homes)'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        metavar='DIR',
        help='Output directory for .ttl and .json files (default: output/)'
    )
    parser.add_argument(
        '-u', '--base-url',
        default='http://localhost:8080',
        metavar='URL',
        help='Base URL for the hypermedia environment (default: http://localhost:8080)'
    )

    args = parser.parse_args()

    input_file = args.input
    output_dir = Path(args.output)
    base_url = args.base_url

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input JSON
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)

    # Convert
    converter = SmartHomeToTDConverter(base_url=base_url)

    try:
        results = converter.convert(input_data)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Write outputs for each home
    total_artifacts = 0
    for home_id, (graph, json_state) in results.items():
        # Write RDF output
        rdf_file = output_dir / f"home_{home_id}.ttl"
        graph.serialize(destination=rdf_file, format='turtle')

        # Write JSON state output
        state_file = output_dir / f"home_{home_id}_state.json"
        with open(state_file, 'w') as f:
            json.dump(json_state, f, indent=4)

        num_artifacts = len(json_state)
        total_artifacts += num_artifacts

        print(f"Home {home_id}:")
        print(f"  RDF output written to: {rdf_file}")
        print(f"  JSON state written to: {state_file}")
        print(f"  Generated {num_artifacts} artifacts")
        print()

    print(f"Conversion complete!")
    print(f"Total: {len(results)} homes, {total_artifacts} artifacts")


if __name__ == "__main__":
    main()
