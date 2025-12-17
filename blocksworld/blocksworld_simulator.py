#!/usr/bin/env python3
"""
Blocksworld Simulator - FastAPI-based implementation

Simulates blocksworld artifacts based on TD descriptions
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Set
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


class BlocksWorldDevice:
    """BlocksWorld device managing block states and actions"""

    def __init__(self, artifact_uri: str, initial_state: Dict[str, Any], goal_state: Optional[Dict[str, Any]] = None):
        """
        Initialize a blocksworld device

        Args:
            artifact_uri: URI of the artifact
            initial_state: Initial state with blocks and hand
            goal_state: Goal state to check against (optional)
        """
        self.artifact_uri = artifact_uri
        self.state = self._deep_copy_state(initial_state)
        self.goal_state = self._deep_copy_state(goal_state) if goal_state else None
        self.blocks = {block['name'] for block in self.state.get('blocks', [])}

    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a state dictionary"""
        return json.loads(json.dumps(state))

    def get_device_type(self) -> str:
        """Return the device type name"""
        return "blocksworld"

    def get_block_by_name(self, name: str) -> Optional[Dict]:
        """Get block data by name"""
        for block in self.state['blocks']:
            if block['name'] == name:
                return block
        return None

    def is_clear(self, block_name: str) -> bool:
        """Check if a block is clear"""
        block = self.get_block_by_name(block_name)
        return block and block['properties'].get('clear', False)

    def is_ontable(self, block_name: str) -> bool:
        """Check if a block is on the table"""
        block = self.get_block_by_name(block_name)
        return block and block['properties'].get('ontable', False)

    def is_on(self, top_block: str, bottom_block: str) -> bool:
        """Check if top_block is on bottom_block"""
        block = self.get_block_by_name(top_block)
        return block and block['properties'].get('on') == bottom_block

    def get_property(self, property_name: str) -> Any:
        """Get a property value"""
        if property_name == "state":
            return self.state
        raise KeyError(f"Property '{property_name}' not found")

    def validate_pickup(self, target_block: str) -> tuple[bool, Optional[str]]:
        """Validate pickup action"""
        if self.state['hand'] != 'empty':
            return False, json.dumps({
                "error": f"Cannot pick up block '{target_block}': hand is not empty (holding '{self.state['hand']}')"
            })

        if target_block not in self.blocks:
            return False, json.dumps({"error": f"Block '{target_block}' does not exist"})

        if not self.is_clear(target_block):
            return False, json.dumps({"error": f"Cannot pick up block '{target_block}': block is not clear"})

        if not self.is_ontable(target_block):
            return False, json.dumps({"error": f"Cannot pick up block '{target_block}': block is not on the table"})

        return True, None

    def validate_putdown(self, target_block: str) -> tuple[bool, Optional[str]]:
        """Validate putdown action"""
        if self.state['hand'] == 'empty':
            return False, json.dumps({"error": "Cannot put down block: hand is empty"})

        if self.state['hand'] != target_block:
            return False, json.dumps({
                "error": f"Cannot put down block '{target_block}': hand is holding '{self.state['hand']}'"
            })

        return True, None

    def validate_stack(self, target_block: str, to_block: str) -> tuple[bool, Optional[str]]:
        """Validate stack action"""
        if self.state['hand'] == 'empty':
            return False, json.dumps({"error": "Cannot stack block: hand is empty"})

        if self.state['hand'] != target_block:
            return False, json.dumps({
                "error": f"Cannot stack block '{target_block}': hand is holding '{self.state['hand']}'"
            })

        if to_block not in self.blocks:
            return False, json.dumps({"error": f"Block '{to_block}' does not exist"})

        if not self.is_clear(to_block):
            return False, json.dumps({"error": f"Cannot stack on block '{to_block}': block is not clear"})

        return True, None

    def validate_unstack(self, target_block: str, from_block: str) -> tuple[bool, Optional[str]]:
        """Validate unstack action"""
        if self.state['hand'] != 'empty':
            return False, json.dumps({
                "error": f"Cannot unstack block '{target_block}': hand is not empty (holding '{self.state['hand']}')"
            })

        if target_block not in self.blocks:
            return False, json.dumps({"error": f"Block '{target_block}' does not exist"})

        if from_block not in self.blocks:
            return False, json.dumps({"error": f"Block '{from_block}' does not exist"})

        if not self.is_clear(target_block):
            return False, json.dumps({"error": f"Cannot unstack block '{target_block}': block is not clear"})

        if not self.is_on(target_block, from_block):
            return False, json.dumps({
                "error": f"Cannot unstack block '{target_block}' from '{from_block}': '{target_block}' is not on '{from_block}'"
            })

        return True, None

    def pickup(self, target_block: str):
        """Apply pickup action"""
        valid, error = self.validate_pickup(target_block)
        if not valid:
            raise ValueError(error)

        block = self.get_block_by_name(target_block)
        block['properties']['ontable'] = False
        block['properties']['clear'] = True
        self.state['hand'] = target_block

    def putdown(self, target_block: str):
        """Apply putdown action"""
        valid, error = self.validate_putdown(target_block)
        if not valid:
            raise ValueError(error)

        block = self.get_block_by_name(target_block)
        block['properties']['ontable'] = True
        block['properties']['clear'] = True
        self.state['hand'] = 'empty'

    def stack(self, target_block: str, to_block: str):
        """Apply stack action"""
        valid, error = self.validate_stack(target_block, to_block)
        if not valid:
            raise ValueError(error)

        target = self.get_block_by_name(target_block)
        bottom = self.get_block_by_name(to_block)

        target['properties']['on'] = to_block
        target['properties'].pop('ontable', None)
        bottom['properties']['clear'] = False
        self.state['hand'] = 'empty'

    def unstack(self, target_block: str, from_block: str):
        """Apply unstack action"""
        valid, error = self.validate_unstack(target_block, from_block)
        if not valid:
            raise ValueError(error)

        target = self.get_block_by_name(target_block)
        bottom = self.get_block_by_name(from_block)

        target['properties'].pop('on', None)
        bottom['properties']['clear'] = True
        self.state['hand'] = target_block

    def check_goal_reached(self) -> bool:
        """
        Check if current state satisfies all goal constraints
        Goal state contains only explicit constraints that must be satisfied
        """
        if not self.goal_state:
            return False

        # Check hand state if specified in goal
        if 'hand' in self.goal_state:
            if self.state.get('hand') != self.goal_state.get('hand'):
                return False

        # Build property map for current state
        current_props = {block['name']: block['properties'] for block in self.state['blocks']}

        # Check all blocks mentioned in goal
        for goal_block in self.goal_state.get('blocks', []):
            block_name = goal_block['name']
            goal_block_props = goal_block['properties']

            if block_name not in current_props:
                return False

            current_block_props = current_props[block_name]

            # Check each property constraint in goal
            for prop_key, prop_value in goal_block_props.items():
                # The current state must have this property with the exact value
                if prop_key not in current_block_props:
                    return False
                elif current_block_props[prop_key] != prop_value:
                    return False

        return True

    def get_goal_state(self) -> Optional[Dict[str, Any]]:
        """Get the goal state"""
        return self._deep_copy_state(self.goal_state) if self.goal_state else None


class BlocksworldSimulator:
    """Blocksworld simulator that manages devices and handles HTTP requests"""

    def __init__(self, description_dir: Path):
        self.description_dir = Path(description_dir)
        self.devices: Dict[str, BlocksWorldDevice] = {}
        self.property_routes: Dict[str, str] = {}  # path -> artifact_uri
        self.action_routes: Dict[str, tuple] = {}  # path -> (artifact_uri, action_name, params)
        self.graph: Optional[Graph] = None
        self.workspace_uri: Optional[str] = None
        self.artifact_graphs: Dict[str, Graph] = {}  # artifact_uri -> subgraph with TD description

    def load_blocksworld(self):
        """Load blocksworld descriptions from the directory"""
        ttl_file = self.description_dir / "blocksworld.ttl"
        state_file = self.description_dir / "blocksworld_state.json"
        goals_file = self.description_dir / "blocksworld_goals.json"

        if not ttl_file.exists():
            raise FileNotFoundError(f"TTL file not found: {ttl_file}")

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        print("Loading blocksworld...")
        self._load_world(ttl_file, state_file, goals_file if goals_file.exists() else None)

    def _load_world(self, ttl_file: Path, state_file: Path, goals_file: Optional[Path]):
        """Load blocksworld from TTL and state files"""
        # Load state
        with open(state_file, 'r') as f:
            states = json.load(f)

        # Load goals if available
        goals = {}
        if goals_file and goals_file.exists():
            with open(goals_file, 'r') as f:
                goals = json.load(f)

        # Parse TTL file
        g = Graph()
        g.parse(ttl_file, format='turtle')
        self.graph = g

        # Find workspace
        for workspace_uri in g.subjects(predicate=RDF.type, object=HMAS.Workspace):
            self.workspace_uri = str(workspace_uri)
            break

        # Find all artifacts
        for artifact_uri in g.subjects(predicate=RDF.type, object=EX.BlocksWorldSim):
            artifact_uri_str = str(artifact_uri)

            # Get initial state
            initial_state = states.get(artifact_uri_str, {})

            # Get goal state
            goal_state = goals.get(artifact_uri_str)

            # Create device instance
            device = BlocksWorldDevice(artifact_uri_str, initial_state, goal_state)
            self.devices[artifact_uri_str] = device

            # Register routes
            self._register_routes(g, artifact_uri, artifact_uri_str)

            # Store artifact subgraph
            artifact_graph = Graph()

            def add_triples_recursive(node, visited=None):
                if visited is None:
                    visited = set()

                node_id = str(node)
                if node_id in visited:
                    return
                visited.add(node_id)

                for s, p, o in g.triples((node, None, None)):
                    artifact_graph.add((s, p, o))
                    if not isinstance(o, Literal) and p != HMAS.contains:
                        add_triples_recursive(o, visited)

            add_triples_recursive(artifact_uri)
            self.artifact_graphs[artifact_uri_str] = artifact_graph

        print(f"Loaded {len(self.devices)} blocksworld artifacts")
        print(f"Registered {len(self.property_routes)} property endpoints")
        print(f"Registered {len(self.action_routes)} action endpoints")

    def _register_routes(self, g: Graph, artifact_uri: URIRef, artifact_uri_str: str):
        """Register property and action routes from RDF graph"""
        # Register property affordances
        for prop_aff in g.objects(artifact_uri, TD.hasPropertyAffordance):
            prop_name = None
            for name in g.objects(prop_aff, TD.name):
                prop_name = str(name)
                break

            if not prop_name:
                continue

            # Get target URL from form
            for form in g.objects(prop_aff, TD.hasForm):
                for target in g.objects(form, HCTL.hasTarget):
                    target_path = self._extract_path(str(target))
                    self.property_routes[target_path] = artifact_uri_str

        # Register action affordances
        for action_aff in g.objects(artifact_uri, TD.hasActionAffordance):
            action_name = None
            for name in g.objects(action_aff, TD.name):
                action_name = str(name)
                break

            if not action_name:
                continue

            # Get parameters from input schema
            params = []
            JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")

            for input_schema in g.objects(action_aff, TD.hasInputSchema):
                for prop in g.objects(input_schema, JSONSCHEMA.properties):
                    param_name = None
                    for pn in g.objects(prop, JSONSCHEMA.propertyName):
                        param_name = str(pn)
                        params.append(param_name)
                        break

            # Get target URL from form
            for form in g.objects(action_aff, TD.hasForm):
                for target in g.objects(form, HCTL.hasTarget):
                    target_path = self._extract_path(str(target))
                    self.action_routes[target_path] = (artifact_uri_str, action_name, params)

    def _extract_path(self, url: str) -> str:
        """Extract path from full URL"""
        if url.startswith("http://localhost:8080"):
            return url.replace("http://localhost:8080", "")
        return url

    def get_property(self, path: str) -> Any:
        """Get a property value"""
        if path not in self.property_routes:
            raise HTTPException(status_code=404, detail=f"Property endpoint not found: {path}")

        artifact_uri = self.property_routes[path]

        if artifact_uri not in self.devices:
            raise HTTPException(status_code=500, detail=f"Device not found for artifact: {artifact_uri}")

        device = self.devices[artifact_uri]

        try:
            # For blocksworld, state property returns the full state
            return device.get_property("state")
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    def invoke_action(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an action"""
        if path not in self.action_routes:
            raise HTTPException(status_code=404, detail=f"Action endpoint not found: {path}")

        artifact_uri, action_name, params = self.action_routes[path]

        if artifact_uri not in self.devices:
            raise HTTPException(status_code=500, detail=f"Device not found for artifact: {artifact_uri}")

        device = self.devices[artifact_uri]

        # Check if method exists
        if not hasattr(device, action_name):
            raise HTTPException(status_code=500, detail=f"Method '{action_name}' not implemented for device")

        method = getattr(device, action_name)

        try:
            # Validate parameters
            if params:
                for param in params:
                    if param not in payload:
                        raise HTTPException(status_code=400, detail=f"Missing required parameter: {param}")

                # Call method with parameters
                method(**payload)
            else:
                # Call method without parameters
                method()

            return {"status": "success", "message": f"Action '{action_name}' executed successfully"}

        except ValueError as e:
            # Validation errors from blocksworld rules
            try:
                error_data = json.loads(str(e))
                raise HTTPException(status_code=400, detail=error_data.get("error", str(e)))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def check_goal(self, artifact_uri: str) -> Dict[str, Any]:
        """Check if an artifact has reached its goal state"""
        if artifact_uri not in self.devices:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_uri}")

        device = self.devices[artifact_uri]

        if device.goal_state is None:
            raise HTTPException(status_code=404, detail=f"No goal state defined for artifact: {artifact_uri}")

        goal_reached = device.check_goal_reached()

        return {
            "artifact_uri": artifact_uri,
            "goal_reached": goal_reached,
            "current_state": device.state,
            "goal_state": device.goal_state
        }

    def get_platform_rdf(self) -> str:
        """Generate RDF for the HypermediaMASPlatform root"""
        g = Graph()

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

        # Add blocksworld workspace
        if self.workspace_uri:
            workspace_uri = URIRef(self.workspace_uri)
            g.add((platform_uri, HMAS.hosts, workspace_uri))

        return g.serialize(format='turtle')

    def get_workspace_rdf(self) -> str:
        """Generate RDF for the blocksworld workspace"""
        if not self.workspace_uri or not self.graph:
            raise HTTPException(status_code=404, detail="Workspace not found")

        g = Graph()
        g.bind("hmas", HMAS)
        g.bind("td", TD)
        g.bind("rdf", RDF)

        workspace_uri = URIRef(self.workspace_uri)

        # Workspace description
        g.add((workspace_uri, RDF.type, HMAS.Workspace))
        g.add((workspace_uri, RDF.type, TD.Thing))

        # Add contained artifacts
        for artifact_uri_str in self.devices.keys():
            artifact_uri = URIRef(artifact_uri_str)
            g.add((workspace_uri, HMAS.contains, artifact_uri))

        return g.serialize(format='turtle')

    def get_artifact_rdf(self, artifact_name: str) -> str:
        """Generate RDF for an artifact showing its TD description"""
        artifact_uri_str = f"http://localhost:8080/workspaces/blocksworld/artifacts/{artifact_name}#artifact"

        if artifact_uri_str not in self.artifact_graphs:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")

        artifact_graph = self.artifact_graphs[artifact_uri_str]

        # Bind namespaces
        artifact_graph.bind("hmas", HMAS)
        artifact_graph.bind("td", TD)
        artifact_graph.bind("rdf", RDF)
        artifact_graph.bind("hctl", HCTL)
        artifact_graph.bind("http", HTTP)
        artifact_graph.bind("jsonschema", Namespace("https://www.w3.org/2019/wot/json-schema#"))
        artifact_graph.bind("ex", EX)

        return artifact_graph.serialize(format='turtle')


# Global simulator instance and config
simulator: Optional[BlocksworldSimulator] = None
config: Dict[str, Any] = {
    "description_dir": Path("../datasets/Blocksworld/hmas_format/generated_basic")
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global simulator

    # Startup
    description_dir = config["description_dir"]

    if not description_dir.exists():
        print(f"Warning: Description directory not found: {description_dir}")
        print("Creating simulator without loading worlds...")
        simulator = BlocksworldSimulator(description_dir)
    else:
        simulator = BlocksworldSimulator(description_dir)
        simulator.load_blocksworld()

    yield

    # Shutdown
    print("Shutting down simulator...")


# Create FastAPI app with lifespan
app = FastAPI(title="Blocksworld Simulator", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    """Root endpoint returning HypermediaMASPlatform RDF"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    rdf_content = simulator.get_platform_rdf()
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/blocksworld")
async def get_workspace():
    """GET endpoint for blocksworld workspace RDF description"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    rdf_content = simulator.get_workspace_rdf()
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/blocksworld/artifacts/{artifact_name}")
async def get_artifact(artifact_name: str):
    """GET endpoint for artifact RDF description (TD)"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    rdf_content = simulator.get_artifact_rdf(artifact_name)
    return Response(content=rdf_content, media_type="text/turtle")


@app.get("/workspaces/blocksworld/artifacts/{artifact_name}/properties/state")
async def get_state(artifact_name: str):
    """GET endpoint for state property"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    path = f"/workspaces/blocksworld/artifacts/{artifact_name}/properties/state"
    return simulator.get_property(path)


@app.post("/workspaces/blocksworld/artifacts/{artifact_name}/{action_name}")
async def invoke_action(artifact_name: str, action_name: str, request: Request):
    """POST endpoint for action affordances"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    path = f"/workspaces/blocksworld/artifacts/{artifact_name}/{action_name}"

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    return simulator.invoke_action(path, payload)


@app.get("/workspaces/blocksworld/artifacts/{artifact_name}/goal")
async def check_goal(artifact_name: str):
    """GET endpoint to check if artifact has reached its goal state"""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")

    artifact_uri = f"http://localhost:8080/workspaces/blocksworld/artifacts/{artifact_name}#artifact"
    return simulator.check_goal(artifact_uri)


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


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blocksworld Simulator - FastAPI-based implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blocksworld_simulator.py
  python blocksworld_simulator.py --data-dir ../datasets/Blocksworld/hmas_format/generated_basic
  python blocksworld_simulator.py --data-dir /path/to/data --port 8081
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("../datasets/Blocksworld/hmas_format/generated_basic"),
        metavar='DIR',
        help='Path to blocksworld data directory (default: ../datasets/Blocksworld/hmas_format/generated_basic)'
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
    config["description_dir"] = args.data_dir

    print(f"Starting Blocksworld Simulator...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)
