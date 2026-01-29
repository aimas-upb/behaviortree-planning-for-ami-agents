"""
Script to extract workspace types, device types, and their action affordances
from one or more HomeBench home description TTL files and generate an ontology
file (homeont.ttl) that describes the types and their affordance classes.

Usage:
    python scripts/generate_homeont.py \
        -o ontologies/homeont.ttl \
        datasets/HomeBench/hmas_format/home_description/home_0.ttl \
        datasets/HomeBench/hmas_format/home_description/home_1.ttl \
        ...
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


# Human-readable descriptions for each device type.
# These are kept generic and do not enumerate specific actions, since not all
# instances of a device type expose every possible action.
DEVICE_DESCRIPTIONS = {
    "AirConditioner": "A smart air conditioning unit used to regulate indoor climate.",
    "AirPurifiers": "A smart air purifier that filters and cleans indoor air.",
    "Aromatherapy": "A smart aromatherapy diffuser that releases scented vapors into the room.",
    "Blinds": "Smart window blinds used to control light and privacy.",
    "Curtain": "A smart window curtain used to control light and privacy.",
    "Dehumidifiers": "A smart dehumidifier that reduces indoor humidity levels.",
    "Fan": "A smart electric fan used to circulate air.",
    "GarageDoor": "A smart motorized garage door.",
    "Heating": "A smart heating unit used to warm the room.",
    "Humidifier": "A smart humidifier that increases indoor humidity levels.",
    "Light": "A smart light fixture with controllable illumination properties.",
    "MediaPlayer": "A smart media player for audio playback.",
    "Trash": "A smart trash compactor.",
    "VacuumRobot": "A smart robotic vacuum cleaner that can autonomously clean designated areas.",
    "WaterHeater": "A smart water heater used to heat water to a target temperature.",
}

# Human-readable descriptions for each command type
COMMAND_DESCRIPTIONS = {
    "CloseCommand": "Command to close the device (e.g. close blinds, curtains, or a garage door).",
    "OpenCommand": "Command to open the device (e.g. open blinds, curtains, or a garage door).",
    "PackCommand": "Command to pack or compress the contents (e.g. compact trash).",
    "PauseCommand": "Command to pause the current playback on a media player.",
    "SetCleaningAreaCommand": "Command to set the cleaning area for a vacuum robot. Requires a string 'area' parameter.",
    "PlayCommand": "Command to start or resume playback on a media player.",
    "SetArtistCommand": "Command to set the artist filter for media playback. Requires a string 'artist' parameter.",
    "SetBrightnessCommand": "Command to set the brightness level. Requires an integer 'brightness' parameter (0-100).",
    "SetColorCommand": "Command to set the color. Requires a 'color' parameter as an array of integers (e.g. RGB values).",
    "SetDegreeCommand": "Command to set the opening degree. Requires an integer 'degree' parameter.",
    "SetFanSpeedCommand": "Command to set the fan speed. Requires an integer 'fan_speed' parameter.",
    "SetIntensityCommand": "Command to set the intensity level. Requires an integer 'intensity' parameter (0-100).",
    "SetIntervalCommand": "Command to set the operating interval. Requires an integer 'interval' parameter (10-60 minutes).",
    "SetModeCommand": "Command to set the operating mode. Requires a string 'mode' parameter.",
    "SetSongCommand": "Command to set the song to play on a media player. Requires a string 'song' parameter.",
    "SetSpeedCommand": "Command to set the speed. Requires an integer 'speed' parameter.",
    "SetStyleCommand": "Command to set the music style/genre on a media player. Requires a string 'style' parameter.",
    "SetSwingCommand": "Command to set the swing direction. Requires a string 'swing' parameter.",
    "SetTemperatureCommand": "Command to set the target temperature. Requires an integer 'temperature' parameter.",
    "SetVolumeCommand": "Command to set the playback volume. Requires an integer 'volume' parameter (0-100).",
    "StopCommand": "Command to stop the current playback on a media player.",
    "TurnOffCommand": "Command to turn the device off.",
    "TurnOnCommand": "Command to turn the device on.",
}

# Workspace (room) type descriptions, keyed by snake_case name from the TTL.
# VacuumRobot is excluded as it is not a room type.
WORKSPACE_DESCRIPTIONS = {
    "balcony": ("Balcony", "An outdoor or semi-outdoor elevated platform attached to the home."),
    "bathroom": ("Bathroom", "A room equipped with bathing and sanitary facilities."),
    "corridor": ("Corridor", "A hallway or passage connecting different rooms in the home."),
    "dining_room": ("DiningRoom", "A room designated for eating meals."),
    "foyer": ("Foyer", "An entrance hall or lobby area of the home."),
    "garage": ("Garage", "An enclosed space for parking vehicles and storage."),
    "guest_bedroom": ("GuestBedroom", "A bedroom designated for guests."),
    "kitchen": ("Kitchen", "A room used for cooking and food preparation."),
    "living_room": ("LivingRoom", "A main living area used for relaxation and socializing."),
    "master_bedroom": ("MasterBedroom", "The primary bedroom in the home."),
    "store_room": ("StoreRoom", "A room used for storing household items."),
    "study_room": ("StudyRoom", "A room designated for reading, working, or studying."),
}

# Workspace names to exclude (not actual rooms)
WORKSPACE_EXCLUDE = {"VacuumRobot"}


def parse_ttl(ttl_path: str) -> dict:
    """Parse a TTL file and extract device types, commands, and room types."""
    with open(ttl_path, "r") as f:
        content = f.read()

    # Extract device types: patterns like `a ex:DeviceType,\n        hmas:Artifact`
    device_types = set(re.findall(r"a ex:(\w+),\s+hmas:Artifact", content))

    # Extract command types: patterns like `a ex:SomeCommand`
    command_types = set(re.findall(r"a ex:(\w+Command)", content))

    # Extract room/workspace names from workspace URIs
    room_names = set(re.findall(r"home\d+/([a-zA-Z_]+)#workspace", content))
    room_names -= WORKSPACE_EXCLUDE

    # Build device-to-commands mapping by finding each artifact block
    device_commands = defaultdict(set)

    artifact_pattern = re.compile(
        r"<[^>]+#artifact>\s+a\s+ex:(\w+),\s+hmas:Artifact.*?(?=<[^>]+#artifact>|\Z)",
        re.DOTALL,
    )

    for match in artifact_pattern.finditer(content):
        device_type = match.group(1)
        block = match.group(0)
        commands = re.findall(r"a ex:(\w+Command)", block)
        for cmd in commands:
            device_commands[device_type].add(cmd)

    return {
        "device_types": device_types,
        "command_types": command_types,
        "device_commands": device_commands,
        "room_names": room_names,
    }


def merge_parsed(all_parsed: list[dict]) -> dict:
    """Merge parsed results from multiple TTL files."""
    device_types = set()
    command_types = set()
    device_commands = defaultdict(set)
    room_names = set()

    for parsed in all_parsed:
        device_types |= parsed["device_types"]
        command_types |= parsed["command_types"]
        room_names |= parsed["room_names"]
        for device, cmds in parsed["device_commands"].items():
            device_commands[device] |= cmds

    return {
        "device_types": sorted(device_types),
        "command_types": sorted(command_types),
        "device_commands": {k: sorted(v) for k, v in sorted(device_commands.items())},
        "room_names": sorted(room_names),
    }


def generate_ontology(parsed: dict, output_path: str):
    """Generate the homeont.ttl ontology file."""
    lines = []

    # Prefixes
    lines.append("@prefix ex: <http://example.org/> .")
    lines.append("@prefix hmas: <https://purl.org/hmas/> .")
    lines.append("@prefix td: <https://www.w3.org/2019/wot/td#> .")
    lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    lines.append("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    lines.append("")
    lines.append(
        "# Ontology of workspace types, device types, and action affordances"
    )
    lines.append("# for smart home environments.")
    lines.append("# Auto-generated from HomeBench home description data.")
    lines.append("")

    # Property definition
    lines.append("# ============================================================")
    lines.append("# Properties")
    lines.append("# ============================================================")
    lines.append("")
    lines.append("ex:hasPossibleCommand a rdf:Property ;")
    lines.append("    rdfs:domain hmas:Artifact ;")
    lines.append("    rdfs:range td:ActionAffordance ;")
    lines.append(
        '    rdfs:comment "Indicates a command that may be available on instances of a device type. '
        'Not all instances necessarily support every possible command listed." .'
    )
    lines.append("")

    # Workspace types
    lines.append("# ============================================================")
    lines.append("# Workspace Types (Home and Rooms)")
    lines.append("# ============================================================")
    lines.append("")
    lines.append("ex:Home a rdfs:Class ;")
    lines.append("    rdfs:subClassOf hmas:Workspace ;")
    lines.append(
        '    rdfs:comment "A smart home environment composed of multiple rooms and spaces." .'
    )
    lines.append("")

    for room_name in parsed["room_names"]:
        if room_name in WORKSPACE_DESCRIPTIONS:
            class_name, desc = WORKSPACE_DESCRIPTIONS[room_name]
        else:
            # Fallback: convert snake_case to CamelCase
            class_name = "".join(w.capitalize() for w in room_name.split("_"))
            desc = f"A {room_name.replace('_', ' ')} in the home."
        lines.append(f"ex:{class_name} a rdfs:Class ;")
        lines.append("    rdfs:subClassOf hmas:Workspace ;")
        lines.append(f'    rdfs:comment "{desc}" .')
        lines.append("")

    # Device type classes
    lines.append("# ============================================================")
    lines.append("# Device Types")
    lines.append("# ============================================================")
    lines.append("")

    for device in parsed["device_types"]:
        desc = DEVICE_DESCRIPTIONS.get(device, f"A {device} smart home device.")
        lines.append(f"ex:{device} a rdfs:Class ;")
        lines.append("    rdfs:subClassOf hmas:Artifact ;")
        lines.append(f'    rdfs:comment "{desc}" .')
        lines.append("")

    # Command (action affordance) classes
    lines.append("# ============================================================")
    lines.append("# Action Affordance Types (Commands)")
    lines.append("# ============================================================")
    lines.append("")

    for cmd in parsed["command_types"]:
        desc = COMMAND_DESCRIPTIONS.get(cmd, f"A {cmd} action affordance.")
        lines.append(f"ex:{cmd} a rdfs:Class ;")
        lines.append("    rdfs:subClassOf td:ActionAffordance ;")
        lines.append(f'    rdfs:comment "{desc}" .')
        lines.append("")

    # Device-to-command associations
    lines.append("# ============================================================")
    lines.append("# Device Type to Possible Action Affordance Associations")
    lines.append("# ============================================================")
    lines.append("")

    for device, commands in parsed["device_commands"].items():
        if commands:
            cmd_list = ", ".join(f"ex:{c}" for c in commands)
            lines.append(f"ex:{device} ex:hasPossibleCommand {cmd_list} .")
            lines.append("")

    ontology_content = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(ontology_content)

    print(f"Generated ontology at {output_path}")
    print(f"  - {len(parsed['room_names'])} room types")
    print(f"  - {len(parsed['device_types'])} device types")
    print(f"  - {len(parsed['command_types'])} command types")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a home device ontology from HomeBench TTL files."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more home description TTL files to parse.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ontologies/homeont.ttl",
        help="Output path for the generated ontology (default: ontologies/homeont.ttl).",
    )
    args = parser.parse_args()

    all_parsed = []
    for ttl_path in args.input_files:
        print(f"Parsing {ttl_path} ...")
        all_parsed.append(parse_ttl(ttl_path))

    merged = merge_parsed(all_parsed)
    generate_ontology(merged, args.output)


if __name__ == "__main__":
    main()
