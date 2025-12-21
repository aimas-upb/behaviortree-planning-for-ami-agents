"""
HMAS Client for interacting with simulated hypermedia environments.

This module provides standalone functions to navigate and interact with workspaces,
artifacts, properties, and actions in a hypermedia multi-agent system.
"""

import re
import requests
from typing import List, Dict, Any, Optional
from rdflib import Graph, Namespace, URIRef, RDF


# Namespaces for RDF parsing
TD = Namespace("https://www.w3.org/2019/wot/td#")
HMAS = Namespace("https://purl.org/hmas/")
HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
HTTP = Namespace("http://www.w3.org/2011/http#")
JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 30


class GetPropertyError(Exception):
    """Exception raised when getting a property fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        if status_code:
            super().__init__(f"HTTP {status_code}: {message}")
        else:
            super().__init__(message)


class InvokeActionError(Exception):
    """Exception raised when invoking an action fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        if status_code:
            super().__init__(f"HTTP {status_code}: {message}")
        else:
            super().__init__(message)


def _fetch_rdf(uri: str, timeout: int = DEFAULT_TIMEOUT) -> Graph:
    """
    Fetch and parse RDF data from a URI.

    Args:
        uri: The URI to fetch RDF data from
        timeout: Request timeout in seconds

    Returns:
        RDF graph containing the parsed data

    Raises:
        requests.RequestException: If fetching fails
    """
    # Remove fragment identifier for HTTP request
    base_uri = uri.split('#')[0]

    response = requests.get(base_uri, timeout=timeout)
    response.raise_for_status()

    graph = Graph()
    graph.parse(data=response.text, format='turtle')
    return graph


def _camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.

    Args:
        name: CamelCase string

    Returns:
        snake_case string
    """
    # Insert underscore before uppercase letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _parse_schema(graph: Graph, schema_node: Optional[URIRef]) -> Dict[str, Any]:
    """
    Parse a JSON schema from RDF.

    Args:
        graph: RDF graph
        schema_node: Schema node to parse

    Returns:
        Dictionary containing schema information
    """
    if not schema_node:
        return {}

    schema = {}

    # Get schema type
    for type_triple in graph.objects(schema_node, RDF.type):
        type_str = str(type_triple)
        if 'IntegerSchema' in type_str:
            schema['type'] = 'integer'
        elif 'StringSchema' in type_str:
            schema['type'] = 'string'
        elif 'NumberSchema' in type_str:
            schema['type'] = 'number'
        elif 'BooleanSchema' in type_str:
            schema['type'] = 'boolean'
        elif 'ObjectSchema' in type_str:
            schema['type'] = 'object'
        elif 'ArraySchema' in type_str:
            schema['type'] = 'array'

    # Get minimum/maximum for numeric types
    minimum = graph.value(schema_node, JSONSCHEMA.minimum)
    if minimum:
        schema['minimum'] = int(minimum) if schema.get('type') == 'integer' else float(minimum)

    maximum = graph.value(schema_node, JSONSCHEMA.maximum)
    if maximum:
        schema['maximum'] = int(maximum) if schema.get('type') == 'integer' else float(maximum)

    # Get enum values
    enum_values = list(graph.objects(schema_node, JSONSCHEMA.enum))
    if enum_values:
        schema['enum'] = [str(v) for v in enum_values]

    # Get items for array schemas
    items_node = graph.value(schema_node, JSONSCHEMA.items)
    if items_node:
        items_schema = _parse_schema(graph, items_node)
        schema['items'] = items_schema

    # Get properties for object schemas
    properties = {}
    for prop_node in graph.objects(schema_node, JSONSCHEMA.properties):
        prop_name = graph.value(prop_node, JSONSCHEMA.propertyName)
        if prop_name:
            prop_schema = _parse_schema(graph, prop_node)
            properties[str(prop_name)] = prop_schema

    if properties:
        schema['properties'] = properties

    # Get required properties
    required = graph.value(schema_node, JSONSCHEMA.required)
    if required:
        schema['required'] = [str(required)]

    return schema


def list_workspaces(workspace_uri: str) -> List[str]:
    """
    List sub-workspaces contained in a workspace.

    Args:
        workspace_uri: URI of the workspace to query

    Returns:
        List of workspace URIs referencing sub-workspaces contained in the workspace
    """
    graph = _fetch_rdf(workspace_uri)
    workspace_ref = URIRef(workspace_uri)

    workspaces = []
    for obj in graph.objects(workspace_ref, HMAS.contains):
        # Dereference the URI to check if it's a Workspace
        try:
            obj_graph = _fetch_rdf(str(obj))
            obj_ref = URIRef(str(obj))
            if (obj_ref, RDF.type, HMAS.Workspace) in obj_graph:
                workspaces.append(str(obj))
        except Exception:
            # If we can't fetch or parse, skip this object
            continue

    return workspaces


def list_artifacts(workspace_uri: str) -> List[str]:
    """
    List artifacts contained in a workspace.

    Args:
        workspace_uri: URI of the workspace to query

    Returns:
        List of artifact URIs contained in the workspace
    """
    graph = _fetch_rdf(workspace_uri)
    workspace_ref = URIRef(workspace_uri)

    artifacts = []
    # Find all objects contained in this workspace
    for obj in graph.objects(workspace_ref, HMAS.contains):
        # Dereference the URI to check if it's an Artifact
        try:
            obj_graph = _fetch_rdf(str(obj))
            obj_ref = URIRef(str(obj))
            if (obj_ref, RDF.type, HMAS.Artifact) in obj_graph:
                artifacts.append(str(obj))
        except Exception:
            # If we can't fetch or parse, skip this object
            continue

    return artifacts


def get_artifact_name(artifact_uri: str) -> str:
    """
    Get the name of an artifact.

    Args:
        artifact_uri: URI of the artifact

    Returns:
        Name of the artifact from the td:title property
    """
    graph = _fetch_rdf(artifact_uri)
    artifact_ref = URIRef(artifact_uri)

    title = graph.value(artifact_ref, TD.title)
    return str(title) if title else ""


def list_properties(artifact_uri: str) -> List[Dict[str, Any]]:
    """
    List observable properties of an artifact.

    Args:
        artifact_uri: URI of the artifact

    Returns:
        List of property dictionaries with keys:
        - name: property name from td:title
        - uri: URI to invoke (using GET) from hctl:hasTarget
        - output_schema: JSON schema translation of td:hasOutputSchema
    """
    graph = _fetch_rdf(artifact_uri)
    artifact_ref = URIRef(artifact_uri)

    properties = []
    for prop_affordance in graph.objects(artifact_ref, TD.hasPropertyAffordance):
        # Get property name
        prop_name = graph.value(prop_affordance, TD.title)

        # Get the form containing the target URI
        form = graph.value(prop_affordance, TD.hasForm)
        if form:
            prop_uri = graph.value(form, HCTL.hasTarget)

            # Get output schema
            output_schema_node = graph.value(prop_affordance, TD.hasOutputSchema)
            output_schema = _parse_schema(graph, output_schema_node)

            properties.append({
                'name': str(prop_name) if prop_name else "",
                'uri': str(prop_uri) if prop_uri else "",
                'output_schema': output_schema
            })

    return properties


def list_actions(artifact_uri: str) -> List[Dict[str, Any]]:
    """
    List actions available on an artifact.

    Args:
        artifact_uri: URI of the artifact

    Returns:
        List of action dictionaries with keys:
        - name: action name from td:title
        - uri: URI to invoke (using POST) from hctl:hasTarget
        - input_schema: JSON schema translation of td:hasInputSchema
    """
    graph = _fetch_rdf(artifact_uri)
    artifact_ref = URIRef(artifact_uri)

    actions = []
    for action_affordance in graph.objects(artifact_ref, TD.hasActionAffordance):
        # Get action name
        action_name = graph.value(action_affordance, TD.title)

        # Get the form containing the target URI
        form = graph.value(action_affordance, TD.hasForm)
        if form:
            action_uri = graph.value(form, HCTL.hasTarget)

            # Get input schema
            input_schema_node = graph.value(action_affordance, TD.hasInputSchema)
            input_schema = _parse_schema(graph, input_schema_node)

            actions.append({
                'name': str(action_name) if action_name else "",
                'uri': str(action_uri) if action_uri else "",
                'input_schema': input_schema
            })

    return actions


def get_property_by_uri(property_uri: str) -> Any:
    """
    Get a property value using its direct URI.

    Args:
        property_uri: The URI of the property to retrieve

    Returns:
        Property value (parsed as JSON if possible, otherwise text)

    Raises:
        GetPropertyError: If getting the property fails
    """
    try:
        response = requests.get(property_uri, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()

        # Try to parse as JSON, otherwise return text
        try:
            return response.json()
        except ValueError:
            return response.text

    except requests.HTTPError as e:
        raise GetPropertyError(
            f"Failed to get property from {property_uri}: {str(e)}",
            status_code=e.response.status_code if e.response else None
        )
    except requests.RequestException as e:
        raise GetPropertyError(f"Failed to get property from {property_uri}: {str(e)}")


def get_property(artifact_uri: str, property_name: str) -> Any:
    """
    Get a property value from an artifact by name.

    Args:
        artifact_uri: URI of the artifact
        property_name: Name of the property to retrieve

    Returns:
        Property value (parsed as JSON if possible, otherwise text)

    Raises:
        GetPropertyError: If getting the property fails
    """
    # Construct property URI from artifact URI and property name
    base_uri = artifact_uri.split('#')[0]
    snake_name = _camel_to_snake(property_name)
    property_uri = f"{base_uri}/properties/{snake_name}"

    return get_property_by_uri(property_uri)


def invoke_action_by_uri(action_uri: str, params: Dict[str, Any]) -> bool:
    """
    Invoke an action using its direct URI.

    Args:
        action_uri: The URI of the action to invoke
        params: Parameters to pass to the action as JSON payload

    Returns:
        True if successful, False otherwise

    Raises:
        InvokeActionError: If invoking the action fails
    """
    try:
        response = requests.post(
            action_uri,
            json=params,
            timeout=DEFAULT_TIMEOUT,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return True

    except requests.HTTPError as e:
        raise InvokeActionError(
            f"Failed to invoke action at {action_uri}: {str(e)}",
            status_code=e.response.status_code if e.response else None
        )
    except requests.RequestException as e:
        raise InvokeActionError(f"Failed to invoke action at {action_uri}: {str(e)}")


def invoke_action(artifact_uri: str, action_name: str, params: Dict[str, Any]) -> bool:
    """
    Invoke an action on an artifact by name.

    Args:
        artifact_uri: URI of the artifact
        action_name: Name of the action to invoke
        params: Parameters to pass to the action as JSON payload

    Returns:
        True if successful, False otherwise

    Raises:
        InvokeActionError: If invoking the action fails
    """
    # Construct action URI from artifact URI and action name
    base_uri = artifact_uri.split('#')[0]
    snake_name = _camel_to_snake(action_name)
    action_uri = f"{base_uri}/{snake_name}"

    return invoke_action_by_uri(action_uri, params)
