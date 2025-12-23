#!/usr/bin/env python3
"""
Smart Home Simulator - FastAPI-based implementation

Simulates smart home devices based on TD artifact descriptions
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from rdflib import Graph, Namespace, URIRef, RDF, Literal
import uvicorn


# Namespaces for RDF parsing
TD = Namespace("https://www.w3.org/2019/wot/td#")
HMAS = Namespace("https://purl.org/hmas/")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
HTTP = Namespace("http://www.w3.org/2011/http#")
EX = Namespace("http://example.org/")


class Device(ABC):
    """Base class for all smart home devices"""

    def __init__(self, artifact_uri: str, initial_state: Dict[str, Any], available_actions: set):
        self.artifact_uri = artifact_uri
        self.state = initial_state.copy()
        self.available_actions = available_actions  # Set of action names that are available on this instance

    @abstractmethod
    def get_device_type(self) -> str:
        """Return the device type name"""
        pass

    def is_action_available(self, action_name: str) -> bool:
        """Check if an action is available on this device instance"""
        return action_name in self.available_actions

    def get_property(self, property_name: str) -> Any:
        """Get a property value"""
        if property_name not in self.state:
            raise KeyError(f"Property '{property_name}' not found")
        return self.state[property_name]

    def set_property(self, property_name: str, value: Any):
        """Set a property value"""
        self.state[property_name] = value

    def get_all_properties(self) -> Dict[str, Any]:
        """Get all properties"""
        return self.state.copy()


class LightDevice(Device):
    """Light device - Properties: brightness, color, state"""

    def get_device_type(self) -> str:
        return "light"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_brightness(self, brightness: int):
        self.state['brightness'] = brightness

    def set_color(self, color: str):
        self.state['color'] = color


class HeatingDevice(Device):
    """Heating device - Properties: fan_speed, mode, state, temperature"""

    def get_device_type(self) -> str:
        return "heating"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_temperature(self, temperature: int):
        self.state['temperature'] = temperature

    def set_mode(self, mode: str):
        self.state['mode'] = mode

    def set_fan_speed(self, fan_speed: str):
        self.state['fan_speed'] = fan_speed


class FanDevice(Device):
    """Fan device - Properties: speed, state, swing"""

    def get_device_type(self) -> str:
        return "fan"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_speed(self, speed: str):
        self.state['speed'] = speed

    def set_swing(self, swing: str):
        self.state['swing'] = swing


class AirConditionerDevice(Device):
    """Air conditioner device - Properties: fan_speed, mode, state, swing, temperature"""

    def get_device_type(self) -> str:
        return "air_conditioner"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_temperature(self, temperature: int):
        self.state['temperature'] = temperature

    def set_mode(self, mode: str):
        self.state['mode'] = mode

    def set_fan_speed(self, fan_speed: str):
        self.state['fan_speed'] = fan_speed

    def set_swing(self, swing: str):
        self.state['swing'] = swing


class GarageDoorDevice(Device):
    """Garage door device"""

    def get_device_type(self) -> str:
        return "garage_door"

    def open(self):
        self.state['state'] = 'open'

    def close(self):
        self.state['state'] = 'closed'


class BlindsDevice(Device):
    """Blinds device"""

    def get_device_type(self) -> str:
        return "blinds"

    def open(self):
        self.state['state'] = 'open'

    def close(self):
        self.state['state'] = 'closed'

    def set_degree(self, degree: int):
        if 'degree' in self.state:
            self.state['degree'] = degree


class CurtainDevice(Device):
    """Curtain device"""

    def get_device_type(self) -> str:
        return "curtain"

    def open(self):
        self.state['state'] = 'open'

    def close(self):
        self.state['state'] = 'closed'

    def set_degree(self, degree: int):
        if 'degree' in self.state:
            self.state['degree'] = degree


class MediaPlayerDevice(Device):
    """Media player device - Properties: state, volume"""

    def get_device_type(self) -> str:
        return "media_player"

    def play(self):
        self.state['state'] = 'playing'

    def pause(self):
        self.state['state'] = 'paused'

    def stop(self):
        self.state['state'] = 'stopped'

    def set_volume(self, volume: int):
        self.state['volume'] = volume

    def set_artist(self, artist: str):
        self.state['artist'] = artist

    def set_song(self, song: str):
        self.state['song'] = song

    def set_style(self, style: str):
        self.state['style'] = style


class VacuumRobotDevice(Device):
    """Vacuum robot device"""

    def get_device_type(self) -> str:
        return "vacuum_robot"

    def start(self):
        self.state['state'] = 'cleaning'

    def stop(self):
        self.state['state'] = 'idle'

    def return_to_dock(self):
        self.state['state'] = 'docked'


class TrashDevice(Device):
    """Trash device - Properties: state"""

    def get_device_type(self) -> str:
        return "trash"

    def pack(self):
        self.state['state'] = 'packed'


class HumidifierDevice(Device):
    """Humidifier device - Properties: intensity, mode, state, tank"""

    def get_device_type(self) -> str:
        return "humidifier"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_intensity(self, intensity: int):
        self.state['intensity'] = intensity

    def set_mode(self, mode: str):
        self.state['mode'] = mode


class DehumidifierDevice(Device):
    """Dehumidifier device - Properties: intensity, mode, state, tank"""

    def get_device_type(self) -> str:
        return "dehumidifiers"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_intensity(self, intensity: int):
        self.state['intensity'] = intensity

    def set_mode(self, mode: str):
        self.state['mode'] = mode


class AromatherapyDevice(Device):
    """Aromatherapy device - Properties: intensity, interval, state"""

    def get_device_type(self) -> str:
        return "aromatherapy"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_interval(self, interval: int):
        self.state['interval'] = interval

    def set_intensity(self, intensity: int):
        self.state['intensity'] = intensity


class WaterHeaterDevice(Device):
    """Water heater device - Properties: mode, state, temperature"""

    def get_device_type(self) -> str:
        return "water_heater"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_temperature(self, temperature: int):
        self.state['temperature'] = temperature

    def set_mode(self, mode: str):
        self.state['mode'] = mode


class AirPurifierDevice(Device):
    """Air purifier device - Properties: fan_speed, mode, state"""

    def get_device_type(self) -> str:
        return "air_purifiers"

    def turn_on(self):
        self.state['state'] = 'on'

    def turn_off(self):
        self.state['state'] = 'off'

    def set_fan_speed(self, fan_speed: str):
        self.state['fan_speed'] = fan_speed

    def set_mode(self, mode: str):
        self.state['mode'] = mode


class PetFeederDevice(Device):
    """Pet feeder device"""

    def get_device_type(self) -> str:
        return "pet_feeder"

    def feed(self):
        if 'last_feed_time' in self.state:
            from datetime import datetime
            self.state['last_feed_time'] = datetime.now().isoformat()

    def set_schedule(self, schedule: str):
        if 'schedule' in self.state:
            self.state['schedule'] = schedule


# Device mapping
DEVICE_MAP = {
    "Light": LightDevice,
    "Heating": HeatingDevice,
    "Fan": FanDevice,
    "AirConditioner": AirConditionerDevice,
    "GarageDoor": GarageDoorDevice,
    "Blinds": BlindsDevice,
    "Curtain": CurtainDevice,
    "MediaPlayer": MediaPlayerDevice,
    "VacuumRobot": VacuumRobotDevice,
    "Trash": TrashDevice,
    "Humidifier": HumidifierDevice,
    "Dehumidifiers": DehumidifierDevice,
    "Aromatherapy": AromatherapyDevice,
    "WaterHeater": WaterHeaterDevice,
    "AirPurifiers": AirPurifierDevice,
    "PetFeeder": PetFeederDevice,
}


class SmartHomeSimulator:
    """Smart home simulator that manages devices and handles HTTP requests"""

    def __init__(self, home_description_dir: Path):
        self.home_description_dir = Path(home_description_dir)
        self.devices: Dict[str, Device] = {}
        self.property_routes: Dict[str, tuple] = {}  # path -> (artifact_uri, prop_name, output_schema_type)
        self.action_routes: Dict[str, tuple] = {}  # path -> (artifact_uri, action_name, params, input_schema_data)
        self.graphs: Dict[str, Graph] = {}  # home_id -> RDF graph
        self.home_workspaces: Dict[str, set] = {}  # home_id -> set of workspace URIs
        self.workspace_contains: Dict[str, set] = {}  # workspace_uri -> set of contained URIs (artifacts or sub-workspaces)
        self.artifact_graphs: Dict[str, Graph] = {}  # artifact_uri -> subgraph with TD description

    def load_homes(self):
        """Load all home descriptions from the directory"""
        ttl_files = sorted(self.home_description_dir.glob("home_*.ttl"))

        for ttl_file in ttl_files:
            home_id = ttl_file.stem.replace("home_", "")
            state_file = self.home_description_dir / f"home_{home_id}_state.json"

            if state_file.exists():
                print(f"Loading home {home_id}...")
                self.load_home(ttl_file, state_file)

    def load_home(self, ttl_file: Path, state_file: Path):
        """Load a single home from TTL and state files"""
        # Extract home_id from filename
        home_id = ttl_file.stem.replace("home_", "")

        # Load state
        with open(state_file, 'r') as f:
            states = json.load(f)

        # Parse TTL file
        g = Graph()
        g.parse(ttl_file, format='turtle')

        # Store the graph for this home
        self.graphs[home_id] = g

        # Track workspaces for this home
        self.home_workspaces[home_id] = set()

        # First pass: find all workspaces and their containment relationships
        for workspace_uri in g.subjects(predicate=RDF.type, object=HMAS.Workspace):
            workspace_uri_str = str(workspace_uri)

            # Track workspace
            self.home_workspaces[home_id].add(workspace_uri_str)

            # Track what this workspace contains
            if workspace_uri_str not in self.workspace_contains:
                self.workspace_contains[workspace_uri_str] = set()

            for contained_uri in g.objects(workspace_uri, HMAS.contains):
                self.workspace_contains[workspace_uri_str].add(str(contained_uri))

        # Find all artifacts
        for artifact_uri in g.subjects(predicate=HMAS.isContainedIn, object=None):
            artifact_uri_str = str(artifact_uri)

            # Get device type from rdf:type
            device_type = self._get_device_type(g, artifact_uri)
            if not device_type:
                continue

            # Get initial state
            initial_state = states.get(artifact_uri_str, {})

            # Get available actions for this device instance
            available_actions = self._get_available_actions(g, artifact_uri)

            # Create device instance
            device_class = DEVICE_MAP.get(device_type)
            if device_class:
                device = device_class(artifact_uri_str, initial_state, available_actions)
                self.devices[artifact_uri_str] = device

                # Register routes
                self._register_routes(g, artifact_uri, artifact_uri_str)

                # Store artifact subgraph (all triples with this artifact as subject)
                artifact_graph = Graph()

                # Recursive function to add all related triples including blank nodes
                def add_triples_recursive(node, visited=None):
                    if visited is None:
                        visited = set()

                    node_id = str(node)
                    if node_id in visited:
                        return
                    visited.add(node_id)

                    for s, p, o in g.triples((node, None, None)):
                        artifact_graph.add((s, p, o))
                        # Recursively process blank nodes and URIs, but not certain predicates
                        # Don't follow hmas:contains (to avoid pulling in sibling artifacts)
                        if not isinstance(o, Literal) and p != HMAS.contains:
                            add_triples_recursive(o, visited)

                add_triples_recursive(artifact_uri)
                self.artifact_graphs[artifact_uri_str] = artifact_graph

    def reset_home(self, home_id: str):
        """Reset a home to its initial state from the state file"""
        # Construct file paths
        state_file = self.home_description_dir / f"home_{home_id}_state.json"

        if not state_file.exists():
            raise HTTPException(status_code=404, detail=f"State file not found for home: {home_id}")

        # Load the initial state
        with open(state_file, 'r') as f:
            states = json.load(f)

        # Reset all devices for this home
        reset_count = 0
        for artifact_uri_str, initial_state in states.items():
            if artifact_uri_str in self.devices:
                # Reset the device state to the initial state
                self.devices[artifact_uri_str].state = initial_state.copy()
                reset_count += 1

        return reset_count

    def _get_device_type(self, g: Graph, artifact_uri: URIRef) -> Optional[str]:
        """Extract device type from RDF graph"""
        for type_uri in g.objects(artifact_uri, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
            type_str = str(type_uri)
            if type_str.startswith("http://example.org/"):
                device_type = type_str.replace("http://example.org/", "")
                if device_type in DEVICE_MAP:
                    return device_type
        return None

    def _get_available_actions(self, g: Graph, artifact_uri: URIRef) -> set:
        """Extract available action names from RDF graph"""
        available_actions = set()

        for action_aff in g.objects(artifact_uri, TD.hasActionAffordance):
            for name in g.objects(action_aff, TD.name):
                action_name = str(name)
                available_actions.add(action_name)

        return available_actions

    def _register_routes(self, g: Graph, artifact_uri: URIRef, artifact_uri_str: str):
        """Register property and action routes from RDF graph"""
        # Register property affordances
        for prop_aff in g.objects(artifact_uri, TD.hasPropertyAffordance):
            # Get property name
            prop_name = None
            for name in g.objects(prop_aff, TD.name):
                prop_name = str(name)
                break

            if not prop_name:
                continue

            # Determine output schema type (default to ObjectSchema for backward compatibility)
            output_schema_type = "object"  # Default wraps in {"value": ...}
            for output_schema in g.objects(prop_aff, TD.hasOutputSchema):
                # Check the schema type
                for schema_type in g.objects(output_schema, RDF.type):
                    schema_type_str = str(schema_type)
                    if "StringSchema" in schema_type_str:
                        output_schema_type = "string"
                    elif "IntegerSchema" in schema_type_str:
                        output_schema_type = "integer"
                    elif "NumberSchema" in schema_type_str:
                        output_schema_type = "number"
                    elif "BooleanSchema" in schema_type_str:
                        output_schema_type = "boolean"
                    elif "ArraySchema" in schema_type_str:
                        output_schema_type = "array"
                    elif "ObjectSchema" in schema_type_str:
                        output_schema_type = "object"
                    break

            # Get target URL from form
            for form in g.objects(prop_aff, TD.hasForm):
                for target in g.objects(form, HCTL.hasTarget):
                    target_path = self._extract_path(str(target))
                    self.property_routes[target_path] = (artifact_uri_str, prop_name, output_schema_type)

        # Register action affordances
        for action_aff in g.objects(artifact_uri, TD.hasActionAffordance):
            # Get action name
            action_name = None
            for name in g.objects(action_aff, TD.name):
                action_name = str(name)
                break

            if not action_name:
                continue

            # Get parameters and validation rules from input schema
            params = []
            input_schema_data = {}  # param_name -> {type, enum, min, max}

            JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")

            for input_schema in g.objects(action_aff, TD.hasInputSchema):
                for prop in g.objects(input_schema, JSONSCHEMA.properties):
                    param_name = None
                    for pn in g.objects(prop, JSONSCHEMA.propertyName):
                        param_name = str(pn)
                        params.append(param_name)
                        break

                    if param_name:
                        # Extract schema validation info
                        schema_info = {}

                        # Get schema type
                        for schema_type in g.objects(prop, RDF.type):
                            schema_type_str = str(schema_type)
                            if "StringSchema" in schema_type_str:
                                schema_info['type'] = 'string'
                            elif "IntegerSchema" in schema_type_str:
                                schema_info['type'] = 'integer'
                            elif "NumberSchema" in schema_type_str:
                                schema_info['type'] = 'number'
                            elif "BooleanSchema" in schema_type_str:
                                schema_info['type'] = 'boolean'
                            elif "ArraySchema" in schema_type_str:
                                schema_info['type'] = 'array'
                                # Get item type if specified
                                for items in g.objects(prop, JSONSCHEMA.items):
                                    for item_type in g.objects(items, RDF.type):
                                        item_type_str = str(item_type)
                                        if "IntegerSchema" in item_type_str:
                                            schema_info['item_type'] = 'integer'
                                        elif "StringSchema" in item_type_str:
                                            schema_info['item_type'] = 'string'
                                        elif "BooleanSchema" in item_type_str:
                                            schema_info['item_type'] = 'boolean'
                                        break

                        # Get enum values
                        enum_values = []
                        for enum_val in g.objects(prop, JSONSCHEMA.enum):
                            enum_values.append(str(enum_val))
                        if enum_values:
                            schema_info['enum'] = enum_values

                        # Get min/max for numeric types
                        for min_val in g.objects(prop, JSONSCHEMA.minimum):
                            try:
                                schema_info['minimum'] = int(str(min_val))
                            except ValueError:
                                schema_info['minimum'] = float(str(min_val))

                        for max_val in g.objects(prop, JSONSCHEMA.maximum):
                            try:
                                schema_info['maximum'] = int(str(max_val))
                            except ValueError:
                                schema_info['maximum'] = float(str(max_val))

                        input_schema_data[param_name] = schema_info

            # Get target URL from form
            for form in g.objects(action_aff, TD.hasForm):
                for target in g.objects(form, HCTL.hasTarget):
                    target_path = self._extract_path(str(target))
                    self.action_routes[target_path] = (artifact_uri_str, action_name, params, input_schema_data)

    def _extract_path(self, url: str) -> str:
        """Extract path from full URL"""
        # Remove http://localhost:8080 prefix
        if url.startswith("http://localhost:8080"):
            return url.replace("http://localhost:8080", "")
        return url

    def get_property(self, path: str) -> Any:
        """Get a property value"""
        if path not in self.property_routes:
            raise HTTPException(status_code=404, detail=f"Property endpoint not found: {path}")

        artifact_uri, prop_name, output_schema_type = self.property_routes[path]

        if artifact_uri not in self.devices:
            raise HTTPException(status_code=500, detail=f"Device not found for artifact: {artifact_uri}")

        device = self.devices[artifact_uri]

        try:
            value = device.get_property(prop_name)

            # Return format based on output schema type
            if output_schema_type == "object":
                # ObjectSchema wraps in {"value": ...}
                return {"value": value}
            else:
                # Primitive types (string, integer, number, boolean) return raw value
                return value
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    def invoke_action(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an action"""
        if path not in self.action_routes:
            raise HTTPException(status_code=404, detail=f"Action endpoint not found: {path}")

        artifact_uri, action_name, params, input_schema_data = self.action_routes[path]

        if artifact_uri not in self.devices:
            raise HTTPException(status_code=500, detail=f"Device not found for artifact: {artifact_uri}")

        device = self.devices[artifact_uri]

        # Check if action is available on this device instance
        if not device.is_action_available(action_name):
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action_name}' is not available on this device instance"
            )

        # Convert camelCase action name to snake_case method name
        method_name = self._camel_to_snake(action_name)

        if not hasattr(device, method_name):
            raise HTTPException(status_code=500, detail=f"Method '{method_name}' not implemented for device")

        method = getattr(device, method_name)

        try:
            # Validate parameters
            if params:
                for param in params:
                    if param not in payload:
                        raise HTTPException(status_code=400, detail=f"Missing required parameter: {param}")

                    # Validate parameter value against schema
                    if param in input_schema_data:
                        self._validate_parameter(param, payload[param], input_schema_data[param])

                # Call method with parameters
                method(**payload)
            else:
                # Call method without parameters
                method()

            return {"status": "success", "message": f"Action '{action_name}' executed successfully"}

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _validate_parameter(self, param_name: str, value: Any, schema_info: Dict[str, Any]):
        """Validate a parameter value against its schema constraints"""
        # Check array type
        if 'type' in schema_info and schema_info['type'] == 'array':
            if not isinstance(value, list):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for parameter '{param_name}': '{value}'. Expected array"
                )

            # Validate item types if specified
            if 'item_type' in schema_info:
                item_type = schema_info['item_type']
                for i, item in enumerate(value):
                    if item_type == 'integer' and not isinstance(item, int):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid item type at index {i} in parameter '{param_name}': '{item}'. Expected integer"
                        )
                    elif item_type == 'string' and not isinstance(item, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid item type at index {i} in parameter '{param_name}': '{item}'. Expected string"
                        )
                    elif item_type == 'boolean' and not isinstance(item, bool):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid item type at index {i} in parameter '{param_name}': '{item}'. Expected boolean"
                        )
            return

        # Check enum constraint
        if 'enum' in schema_info:
            if str(value) not in schema_info['enum']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for parameter '{param_name}': '{value}'. Must be one of: {', '.join(schema_info['enum'])}"
                )

        # Check numeric range constraints
        if 'type' in schema_info and schema_info['type'] in ['integer', 'number']:
            try:
                num_value = float(value) if schema_info['type'] == 'number' else int(value)

                if 'minimum' in schema_info and num_value < schema_info['minimum']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for parameter '{param_name}': {value}. Must be >= {schema_info['minimum']}"
                    )

                if 'maximum' in schema_info and num_value > schema_info['maximum']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value for parameter '{param_name}': {value}. Must be <= {schema_info['maximum']}"
                    )
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for parameter '{param_name}': '{value}'. Expected {schema_info['type']}"
                )

    def _camel_to_snake(self, name: str) -> str:
        """Convert camelCase to snake_case"""
        # Insert underscore before uppercase letters and convert to lowercase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def get_platform_rdf(self) -> str:
        """Generate RDF for the HypermediaMASPlatform root"""
        g = Graph()

        # Bind namespaces
        g.bind("hmas", HMAS)
        g.bind("td", TD)
        g.bind("rdf", RDF)

        platform_uri = URIRef("http://localhost:8080/#platform")
        profile_uri = URIRef("http://localhost:8080/")

        # Platform profile
        g.add((profile_uri, RDF.type, HMAS.ResourceProfile))
        g.add((profile_uri, HMAS.isProfileOf, platform_uri))

        # Platform
        g.add((platform_uri, RDF.type, HMAS.HypermediaMASPlatform))
        g.add((platform_uri, RDF.type, TD.Thing))

        # Add all home workspaces
        for home_id in self.home_workspaces.keys():
            # Add the home workspace URI
            home_workspace_uri = URIRef(f"http://localhost:8080/workspaces/home{home_id}#workspace")
            g.add((platform_uri, HMAS.hosts, home_workspace_uri))

        return g.serialize(format='turtle')

    def get_workspace_rdf(self, workspace_path: str) -> str:
        """Generate RDF for a workspace showing contained artifacts or sub-workspaces"""
        # Parse workspace path to get workspace URI
        # Format: home0/balcony or just home0
        parts = workspace_path.strip('/').split('/')

        if len(parts) == 1:
            # Home workspace
            workspace_uri = URIRef(f"http://localhost:8080/workspaces/{parts[0]}#workspace")
        else:
            # Room workspace
            workspace_uri = URIRef(f"http://localhost:8080/workspaces/{parts[0]}/{parts[1]}#workspace")

        workspace_uri_str = str(workspace_uri)

        if workspace_uri_str not in self.workspace_contains:
            raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_path}")

        g = Graph()
        g.bind("hmas", HMAS)
        g.bind("td", TD)
        g.bind("rdf", RDF)

        # Workspace description
        g.add((workspace_uri, RDF.type, HMAS.Workspace))
        g.add((workspace_uri, RDF.type, TD.Thing))

        # Add contained items (could be artifacts or sub-workspaces)
        for contained_uri_str in self.workspace_contains[workspace_uri_str]:
            contained_uri = URIRef(contained_uri_str)
            g.add((workspace_uri, HMAS.contains, contained_uri))

        return g.serialize(format='turtle')

    def get_artifact_rdf(self, artifact_path: str) -> str:
        """Generate RDF for an artifact showing its TD description"""
        # Parse artifact path to construct artifact URI
        # Format: home0/balcony/artifacts/balconyAromatherapy
        artifact_uri_str = f"http://localhost:8080/workspaces/{artifact_path}#artifact"

        if artifact_uri_str not in self.artifact_graphs:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")

        # Get the stored artifact graph (which is already filtered to just this artifact)
        artifact_graph = self.artifact_graphs[artifact_uri_str]

        # Bind namespaces for better output
        artifact_graph.bind("hmas", HMAS)
        artifact_graph.bind("td", TD)
        artifact_graph.bind("rdf", RDF)
        artifact_graph.bind("hctl", HCTL)
        artifact_graph.bind("http", HTTP)
        artifact_graph.bind("jsonschema", Namespace("https://www.w3.org/2019/wot/json-schema#"))
        artifact_graph.bind("ex", EX)

        return artifact_graph.serialize(format='turtle')


# Global simulator instance and config
simulator: Optional[SmartHomeSimulator] = None
config: Dict[str, Any] = {
    "home_description_dir": Path("datasets/HomeBench/hmas_format/home_description")
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global simulator

    # Startup
    home_description_dir = config["home_description_dir"]

    if not home_description_dir.exists():
        print(f"Warning: Home description directory not found: {home_description_dir}")
        print("Creating simulator without loading homes...")
        simulator = SmartHomeSimulator(home_description_dir)
    else:
        simulator = SmartHomeSimulator(home_description_dir)
        simulator.load_homes()
        print(f"Loaded {len(simulator.devices)} devices")
        print(f"Registered {len(simulator.property_routes)} property endpoints")
        print(f"Registered {len(simulator.action_routes)} action endpoints")

    yield

    # Shutdown (cleanup if needed)
    print("Shutting down simulator...")


# Create FastAPI app with lifespan
app = FastAPI(title="Smart Home Simulator", version="1.0.0", lifespan=lifespan)


@app.get("/workspaces/{home_id}")
async def get_home_workspace(home_id: str):
    """GET endpoint for home workspace RDF description"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    rdf_content = simulator.get_workspace_rdf(home_id)
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/{home_id}/{room_name}")
async def get_room_workspace(home_id: str, room_name: str):
    """GET endpoint for room workspace RDF description"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    workspace_path = f"{home_id}/{room_name}"
    rdf_content = simulator.get_workspace_rdf(workspace_path)
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/{home_id}/{room_name}/artifacts/{artifact_name}")
async def get_artifact(home_id: str, room_name: str, artifact_name: str):
    """GET endpoint for artifact RDF description (TD)"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    artifact_path = f"{home_id}/{room_name}/artifacts/{artifact_name}"
    rdf_content = simulator.get_artifact_rdf(artifact_path)
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/{path:path}/properties/{property_name}")
async def get_property(path: str, property_name: str):
    """GET endpoint for property affordances"""
    full_path = f"/workspaces/{path}/properties/{property_name}"
    return simulator.get_property(full_path)


@app.post("/workspaces/{path:path}/{action_name}")
async def invoke_action(path: str, action_name: str, request: Request):
    """POST endpoint for action affordances"""
    full_path = f"/workspaces/{path}/{action_name}"

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    return simulator.invoke_action(full_path, payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """Handle HTTP exceptions with JSON responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "status_code": 500}
    )


@app.get("/")
async def root():
    """Root endpoint returning HypermediaMASPlatform RDF"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    rdf_content = simulator.get_platform_rdf()
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/reset")
async def reset_home(request: Request):
    """Reset endpoint to restore a home to its initial state"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if "home" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'home' parameter in payload")

    home_id = str(payload["home"])

    # Validate that the home exists
    if home_id not in simulator.graphs:
        raise HTTPException(status_code=404, detail=f"Home not found: {home_id}")

    # Reset the home
    reset_count = simulator.reset_home(home_id)

    return {
        "status": "success",
        "message": f"Home {home_id} reset to initial state",
        "devices_reset": reset_count
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Home Simulator - FastAPI-based implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smart_home_simulator.py
  python smart_home_simulator.py --data-dir ../datasets/HomeBench/hmas_format/home_description
  python smart_home_simulator.py --data-dir /path/to/data --port 8081
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("../datasets/HomeBench/hmas_format/home_description"),
        metavar='DIR',
        help='Path to home description data directory (default: ../datasets/HomeBench/hmas_format/home_description)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default="0.0.0.0",
        help='Host to bind to (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )

    args = parser.parse_args()

    # Update global config
    config["home_description_dir"] = args.data_dir

    print(f"Starting Smart Home Simulator...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
