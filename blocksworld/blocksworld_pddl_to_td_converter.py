#!/usr/bin/env python3
"""
Blocksworld PDDL to ThingDescription Artifact Converter

Converts Blocksworld PDDL problem files into:
1. TD artifact-based RDF/Turtle representation (Hypermedia Environment)
2. BlocksWorldSim artifacts with state management and action validation
3. A single blocksworld workspace containing all problem instances

Author: Dataset Preparation Script
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from rdflib import Graph, Namespace, RDF, RDFS, XSD, Literal, URIRef, BNode


class BlocksWorldState:
    """Manages blocksworld state and validates actions"""

    def __init__(self, blocks: Set[str], init_state: Dict[str, Any]):
        """
        Initialize blocksworld state

        Args:
            blocks: Set of block names
            init_state: Initial state dictionary with 'blocks' and 'hand'
        """
        self.blocks = blocks
        self.state = init_state

    @classmethod
    def from_pddl(cls, pddl_init: List[tuple], blocks: Set[str]) -> 'BlocksWorldState':
        """
        Create state from PDDL init predicates

        Args:
            pddl_init: List of PDDL predicates like ('ontable', 'a'), ('on', 'b', 'c')
            blocks: Set of all block names
        """
        state = {
            'blocks': [],
            'hand': 'empty'
        }

        # Initialize block properties
        block_props = {block: {'clear': False, 'ontable': False} for block in blocks}

        for predicate in pddl_init:
            if predicate[0] == 'handempty':
                state['hand'] = 'empty'
            elif predicate[0] == 'holding':
                state['hand'] = predicate[1]
            elif predicate[0] == 'ontable':
                block = predicate[1]
                block_props[block]['ontable'] = True
            elif predicate[0] == 'clear':
                block = predicate[1]
                block_props[block]['clear'] = True
            elif predicate[0] == 'on':
                top_block = predicate[1]
                bottom_block = predicate[2]
                block_props[top_block]['on'] = bottom_block

        # Convert to list format
        for block in sorted(blocks):
            block_data = {'name': block, 'properties': block_props[block]}
            state['blocks'].append(block_data)

        return cls(blocks, state)

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

    def validate_pickup(self, target_block: str) -> tuple[bool, Optional[str]]:
        """Validate pickup action"""
        if self.state['hand'] != 'empty':
            return False, f"Cannot pick up block '{target_block}': hand is not empty (holding '{self.state['hand']}')"

        if target_block not in self.blocks:
            return False, f"Block '{target_block}' does not exist"

        if not self.is_clear(target_block):
            return False, f"Cannot pick up block '{target_block}': block is not clear"

        if not self.is_ontable(target_block):
            return False, f"Cannot pick up block '{target_block}': block is not on the table"

        return True, None

    def validate_putdown(self, target_block: str) -> tuple[bool, Optional[str]]:
        """Validate putdown action"""
        if self.state['hand'] == 'empty':
            return False, "Cannot put down block: hand is empty"

        if self.state['hand'] != target_block:
            return False, f"Cannot put down block '{target_block}': hand is holding '{self.state['hand']}'"

        return True, None

    def validate_stack(self, target_block: str, to_block: str) -> tuple[bool, Optional[str]]:
        """Validate stack action"""
        if self.state['hand'] == 'empty':
            return False, "Cannot stack block: hand is empty"

        if self.state['hand'] != target_block:
            return False, f"Cannot stack block '{target_block}': hand is holding '{self.state['hand']}'"

        if to_block not in self.blocks:
            return False, f"Block '{to_block}' does not exist"

        if not self.is_clear(to_block):
            return False, f"Cannot stack on block '{to_block}': block is not clear"

        return True, None

    def validate_unstack(self, target_block: str, from_block: str) -> tuple[bool, Optional[str]]:
        """Validate unstack action"""
        if self.state['hand'] != 'empty':
            return False, f"Cannot unstack block '{target_block}': hand is not empty (holding '{self.state['hand']}')"

        if target_block not in self.blocks:
            return False, f"Block '{target_block}' does not exist"

        if from_block not in self.blocks:
            return False, f"Block '{from_block}' does not exist"

        if not self.is_clear(target_block):
            return False, f"Cannot unstack block '{target_block}': block is not clear"

        if not self.is_on(target_block, from_block):
            return False, f"Cannot unstack block '{target_block}' from '{from_block}': '{target_block}' is not on '{from_block}'"

        return True, None

    def apply_pickup(self, target_block: str):
        """Apply pickup action to state"""
        block = self.get_block_by_name(target_block)
        block['properties']['ontable'] = False
        block['properties']['clear'] = True
        self.state['hand'] = target_block

    def apply_putdown(self, target_block: str):
        """Apply putdown action to state"""
        block = self.get_block_by_name(target_block)
        block['properties']['ontable'] = True
        block['properties']['clear'] = True
        self.state['hand'] = 'empty'

    def apply_stack(self, target_block: str, to_block: str):
        """Apply stack action to state"""
        target = self.get_block_by_name(target_block)
        bottom = self.get_block_by_name(to_block)

        target['properties']['on'] = to_block
        target['properties'].pop('ontable', None)
        bottom['properties']['clear'] = False
        self.state['hand'] = 'empty'

    def apply_unstack(self, target_block: str, from_block: str):
        """Apply unstack action to state"""
        target = self.get_block_by_name(target_block)
        bottom = self.get_block_by_name(from_block)

        target['properties'].pop('on', None)
        bottom['properties']['clear'] = True
        self.state['hand'] = target_block

    def to_json(self) -> Dict:
        """Convert state to JSON format"""
        return self.state.copy()

    @classmethod
    def goal_to_json(cls, pddl_goal: List[tuple], blocks: Set[str]) -> Dict[str, Any]:
        """
        Convert PDDL goal predicates to JSON state format
        Only includes explicitly stated goal constraints, not inferred properties

        Args:
            pddl_goal: List of goal predicates like ('on', 'a', 'b'), ('ontable', 'c')
            blocks: Set of all block names

        Returns:
            JSON state representation of the goal (only explicit constraints)
        """
        state = {
            'blocks': []
        }

        # Track blocks mentioned in goal and their properties
        block_props = {}

        for predicate in pddl_goal:
            if predicate[0] == 'on':
                top_block = predicate[1]
                bottom_block = predicate[2]
                if top_block not in block_props:
                    block_props[top_block] = {}
                block_props[top_block]['on'] = bottom_block
            elif predicate[0] == 'ontable':
                block = predicate[1]
                if block not in block_props:
                    block_props[block] = {}
                block_props[block]['ontable'] = True
            elif predicate[0] == 'clear':
                block = predicate[1]
                if block not in block_props:
                    block_props[block] = {}
                block_props[block]['clear'] = True
            elif predicate[0] == 'holding':
                state['hand'] = predicate[1]

        # Only include blocks that are explicitly mentioned in the goal
        # Convert to list format (sorted for consistency)
        for block in sorted(block_props.keys()):
            block_data = {'name': block, 'properties': block_props[block]}
            state['blocks'].append(block_data)

        return state


class PDDLParser:
    """Parse PDDL blocksworld problem files"""

    @staticmethod
    def tokenize(content: str) -> List[str]:
        """Tokenize PDDL content"""
        # Remove comments
        content = re.sub(r';.*$', '', content, flags=re.MULTILINE)
        # Replace parentheses with spaces around them
        content = content.replace('(', ' ( ').replace(')', ' ) ')
        # Split and filter empty strings
        return [token for token in content.split() if token]

    @staticmethod
    def parse_list(tokens: List[str], start: int = 0) -> tuple[Any, int]:
        """Parse a list from tokens, returning (parsed_list, next_index)"""
        if tokens[start] != '(':
            # Return single token
            return tokens[start], start + 1

        result = []
        i = start + 1
        while i < len(tokens) and tokens[i] != ')':
            item, i = PDDLParser.parse_list(tokens, i)
            result.append(item)

        return result, i + 1

    @classmethod
    def parse_pddl_problem(cls, content: str) -> Dict[str, Any]:
        """Parse a PDDL problem file"""
        tokens = cls.tokenize(content)
        parsed, _ = cls.parse_list(tokens)

        if not parsed or parsed[0] != 'define' or parsed[1][0] != 'problem':
            raise ValueError("Invalid PDDL problem file format")

        problem_name = parsed[1][1]
        result = {'problem_name': problem_name}

        # Parse sections
        for section in parsed[2:]:
            if not isinstance(section, list):
                continue

            if section[0] == ':domain':
                result['domain'] = section[1]
            elif section[0] == ':objects':
                result['objects'] = section[1:]
            elif section[0] == ':init':
                init_predicates = []
                for predicate in section[1:]:
                    if isinstance(predicate, list):
                        init_predicates.append(tuple(predicate))
                    else:
                        init_predicates.append((predicate,))
                result['init'] = init_predicates
            elif section[0] == ':goal':
                # Parse goal (usually wrapped in 'and')
                if section[1][0] == 'and':
                    goal_predicates = []
                    for predicate in section[1][1:]:
                        if isinstance(predicate, list):
                            goal_predicates.append(tuple(predicate))
                    result['goal'] = goal_predicates

        return result


class BlocksworldPDDLToTDConverter:
    """Converts Blocksworld PDDL problems to TD artifact format using RDFLib"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

        # Define namespaces
        self.EX = Namespace("http://example.org/")
        self.HCTL = Namespace("https://www.w3.org/2019/wot/hypermedia#")
        self.HMAS = Namespace("https://purl.org/hmas/")
        self.HTTP = Namespace("http://www.w3.org/2011/http#")
        self.JSONSCHEMA = Namespace("https://www.w3.org/2019/wot/json-schema#")
        self.TD = Namespace("https://www.w3.org/2019/wot/td#")

    def add_state_property_affordance(self, g: Graph, artifact_uri: URIRef, artifact_name: str):
        """Add state property affordance to the artifact"""
        prop_node = BNode()
        g.add((artifact_uri, self.TD.hasPropertyAffordance, prop_node))
        g.add((prop_node, RDF.type, self.TD.PropertyAffordance))
        g.add((prop_node, RDFS.comment, Literal(f"Current state of {artifact_name}")))
        g.add((prop_node, self.TD.name, Literal("state")))
        g.add((prop_node, self.TD.title, Literal("state")))
        g.add((prop_node, self.TD.isObservable, Literal(True)))

        # Property read form
        property_url = f"{self.base_url}/workspaces/blocksworld/artifacts/{artifact_name}/properties/state"
        form_node = BNode()
        g.add((prop_node, self.TD.hasForm, form_node))
        g.add((form_node, self.HTTP.methodName, Literal("GET")))
        g.add((form_node, self.HCTL.forContentType, Literal("application/json")))
        g.add((form_node, self.HCTL.hasOperationType, self.TD.readProperty))
        g.add((form_node, self.HCTL.hasTarget, URIRef(property_url)))

        # Output schema - complex object
        output_schema = BNode()
        g.add((prop_node, self.TD.hasOutputSchema, output_schema))
        g.add((output_schema, RDF.type, self.JSONSCHEMA.ObjectSchema))

    def add_action_affordance(self, g: Graph, artifact_uri: URIRef, action_name: str,
                              action_schema: Dict, artifact_name: str):
        """Add action affordance to the artifact"""
        action_node = BNode()
        g.add((artifact_uri, self.TD.hasActionAffordance, action_node))
        g.add((action_node, RDF.type, self.EX[f"{action_name.capitalize()}Command"]))
        g.add((action_node, RDF.type, self.TD.ActionAffordance))
        g.add((action_node, self.TD.name, Literal(action_name)))
        g.add((action_node, self.TD.title, Literal(action_name)))

        # Action form
        action_url = f"{self.base_url}/workspaces/blocksworld/artifacts/{artifact_name}/{action_name}"
        form_node = BNode()
        g.add((action_node, self.TD.hasForm, form_node))
        g.add((form_node, self.HTTP.methodName, Literal("POST")))
        g.add((form_node, self.HCTL.forContentType, Literal("application/json")))
        g.add((form_node, self.HCTL.hasOperationType, self.TD.invokeAction))
        g.add((form_node, self.HCTL.hasTarget, URIRef(action_url)))

        # Input schema
        input_schema = BNode()
        g.add((action_node, self.TD.hasInputSchema, input_schema))
        g.add((input_schema, RDF.type, self.JSONSCHEMA.ObjectSchema))

        for param_name in action_schema:
            prop_node = BNode()
            g.add((input_schema, self.JSONSCHEMA.properties, prop_node))
            g.add((prop_node, RDF.type, self.JSONSCHEMA.StringSchema))
            g.add((prop_node, self.JSONSCHEMA.propertyName, Literal(param_name)))
            g.add((input_schema, self.JSONSCHEMA.required, Literal(param_name)))

    def add_artifact(self, g: Graph, artifact_name: str, initial_state: Dict) -> URIRef:
        """Add a BlocksWorldSim artifact to the graph"""
        artifact_uri = URIRef(f"{self.base_url}/workspaces/blocksworld/artifacts/{artifact_name}#artifact")
        workspace_uri = URIRef(f"{self.base_url}/workspaces/blocksworld#workspace")

        # Add artifact triples
        g.add((artifact_uri, RDF.type, self.EX.BlocksWorldSim))
        g.add((artifact_uri, RDF.type, self.HMAS.Artifact))
        g.add((artifact_uri, RDF.type, self.TD.Thing))
        g.add((artifact_uri, self.HMAS.isContainedIn, workspace_uri))
        g.add((artifact_uri, self.TD.title, Literal(f"BlocksWorld {artifact_name}")))

        # Add state property
        self.add_state_property_affordance(g, artifact_uri, artifact_name)

        # Add action affordances
        actions = {
            'pickup': {'target_block': 'string'},
            'putdown': {'target_block': 'string'},
            'stack': {'target_block': 'string', 'to_block': 'string'},
            'unstack': {'target_block': 'string', 'from_block': 'string'}
        }

        for action_name, schema in actions.items():
            self.add_action_affordance(g, artifact_uri, action_name, schema, artifact_name)

        return artifact_uri

    def add_workspace(self, g: Graph, artifact_uris: List[URIRef]):
        """Add the blocksworld workspace to the graph"""
        workspace_uri = URIRef(f"{self.base_url}/workspaces/blocksworld#workspace")
        platform_uri = URIRef(f"{self.base_url}#platform")

        g.add((workspace_uri, RDF.type, self.HMAS.Workspace))
        g.add((workspace_uri, RDF.type, self.TD.Thing))
        g.add((workspace_uri, self.TD.title, Literal("Blocksworld Workspace")))
        g.add((workspace_uri, self.HMAS.isHostedOn, platform_uri))

        for artifact_uri in artifact_uris:
            g.add((workspace_uri, self.HMAS.contains, artifact_uri))

    def add_platform(self, g: Graph):
        """Add the HMAS platform to the graph"""
        platform_uri = URIRef(f"{self.base_url}#platform")

        g.add((platform_uri, RDF.type, self.HMAS.HypermediaMASPlatform))
        g.add((platform_uri, self.TD.title, Literal("Blocksworld Platform")))

    def convert_pddl_folder(self, input_folder: Path) -> tuple[Graph, Dict[str, Dict], Dict[str, Dict]]:
        """
        Convert all PDDL files in a folder to TD format

        Returns:
            tuple: (RDF Graph, dict of artifact_uri -> initial_state, dict of artifact_uri -> goal_state)
        """
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

        artifact_uris = []
        artifact_states = {}
        artifact_goals = {}

        # Process each PDDL file
        pddl_files = sorted(input_folder.glob("instance-*.pddl"))

        if not pddl_files:
            print(f"Warning: No instance-*.pddl files found in {input_folder}", file=sys.stderr)

        for pddl_file in pddl_files:
            try:
                # Extract ID from filename
                match = re.match(r'instance-(\d+)\.pddl', pddl_file.name)
                if not match:
                    print(f"Warning: Skipping file {pddl_file.name} (invalid name format)", file=sys.stderr)
                    continue

                world_id = match.group(1)
                artifact_name = f"world-{world_id}"

                # Parse PDDL file
                with open(pddl_file, 'r') as f:
                    content = f.read()

                problem_data = PDDLParser.parse_pddl_problem(content)

                # Extract blocks from objects
                blocks = set(problem_data.get('objects', []))

                # Create initial state
                initial_state = BlocksWorldState.from_pddl(problem_data['init'], blocks)

                # Create goal state
                goal_state = BlocksWorldState.goal_to_json(problem_data.get('goal', []), blocks)

                # Add artifact to graph
                artifact_uri = self.add_artifact(g, artifact_name, initial_state.to_json())
                artifact_uris.append(artifact_uri)

                # Store initial state and goal state
                artifact_states[str(artifact_uri)] = initial_state.to_json()
                artifact_goals[str(artifact_uri)] = goal_state

                print(f"Processed: {pddl_file.name} -> {artifact_name}")

            except Exception as e:
                print(f"Error processing {pddl_file.name}: {e}", file=sys.stderr)
                continue

        # Add workspace and platform
        self.add_workspace(g, artifact_uris)
        self.add_platform(g)

        return g, artifact_states, artifact_goals


def main():
    """Main entry point for the converter script"""
    parser = argparse.ArgumentParser(
        description="Convert Blocksworld PDDL problem files into TD artifact-based RDF/Turtle and JSON state representations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blocksworld_pddl_to_td_converter.py -i pddl_problems/ -o output_dir/
  python blocksworld_pddl_to_td_converter.py -i pddl_problems/ -o output_dir/ --base-url http://192.168.1.100:8080
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        metavar='DIR',
        help='Input directory containing PDDL problem files (instance-*.pddl)'
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

    input_folder = Path(args.input)
    output_dir = Path(args.output)
    base_url = args.base_url

    # Validate input directory
    if not input_folder.exists():
        print(f"Error: Input directory '{input_folder}' not found")
        sys.exit(1)

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert
    converter = BlocksworldPDDLToTDConverter(base_url=base_url)

    try:
        graph, artifact_states, artifact_goals = converter.convert_pddl_folder(input_folder)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    rdf_file = output_dir / "blocksworld.ttl"
    graph.serialize(destination=rdf_file, format='turtle')

    state_file = output_dir / "blocksworld_state.json"
    with open(state_file, 'w') as f:
        json.dump(artifact_states, f, indent=4)

    goals_file = output_dir / "blocksworld_goals.json"
    with open(goals_file, 'w') as f:
        json.dump(artifact_goals, f, indent=4)

    print(f"\nConversion complete!")
    print(f"  RDF output written to: {rdf_file}")
    print(f"  JSON state written to: {state_file}")
    print(f"  JSON goals written to: {goals_file}")
    print(f"  Generated {len(artifact_states)} BlocksWorldSim artifacts")


if __name__ == "__main__":
    main()
