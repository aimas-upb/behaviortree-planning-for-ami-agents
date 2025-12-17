#!/usr/bin/env python3
"""
Analyze all device instances across all homes to determine:
1. All possible properties for each device type
2. All possible actions for each device type
"""

import json
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, Namespace, URIRef

TD = Namespace("https://www.w3.org/2019/wot/td#")
HMAS = Namespace("https://purl.org/hmas/")
EX = Namespace("http://example.org/")

def analyze_all_homes():
    """Analyze all home descriptions"""
    home_dir = Path("datasets/HomeBench/hmas_format/home_description")

    # Track all properties and actions per device type
    device_properties = defaultdict(set)
    device_actions = defaultdict(set)

    ttl_files = sorted(home_dir.glob("home_*.ttl"))

    for ttl_file in ttl_files:
        home_id = ttl_file.stem.replace("home_", "")
        state_file = home_dir / f"home_{home_id}_state.json"

        if not state_file.exists():
            continue

        # Load state to get properties
        with open(state_file, 'r') as f:
            states = json.load(f)

        # Parse TTL
        g = Graph()
        g.parse(ttl_file, format='turtle')

        # Find all artifacts
        for artifact_uri in g.subjects(predicate=HMAS.isContainedIn, object=None):
            artifact_uri_str = str(artifact_uri)

            # Get device type
            device_type = None
            for type_uri in g.objects(artifact_uri, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")):
                type_str = str(type_uri)
                if type_str.startswith("http://example.org/"):
                    device_type = type_str.replace("http://example.org/", "")
                    break

            if not device_type:
                continue

            # Get properties from state
            if artifact_uri_str in states:
                for prop_name in states[artifact_uri_str].keys():
                    device_properties[device_type].add(prop_name)

            # Get actions from TD
            for action_aff in g.objects(artifact_uri, TD.hasActionAffordance):
                for name in g.objects(action_aff, TD.name):
                    action_name = str(name)
                    device_actions[device_type].add(action_name)

    return device_properties, device_actions

if __name__ == "__main__":
    print("Analyzing all homes...")
    properties, actions = analyze_all_homes()

    print("\n" + "="*80)
    print("DEVICE ANALYSIS RESULTS")
    print("="*80)

    for device_type in sorted(properties.keys()):
        print(f"\n{device_type}:")
        print(f"  Properties: {sorted(properties[device_type])}")
        print(f"  Actions: {sorted(actions[device_type])}")
