# Quick Start Guide - Smart Home to TD Converter

## Get Started in 3 Steps

### Step 1: Prepare Your Input

Create a JSON file with your smart home configuration:

```json
{
    "home_id": 1,
    "method": [
        {
            "room_name": "living_room",
            "device_name": "light",
            "operation": "turn_on",
            "parameters": []
        }
    ],
    "home_status": {
        "living_room": {
            "room_name": "living_room",
            "light": {
                "state": "on",
                "attributes": {
                    "brightness": {
                        "value": 75,
                        "lowest": 0,
                        "highest": 100
                    }
                }
            }
        }
    }
}
```

### Step 2: Run the Converter

```bash
python smart_home_to_td_converter.py my_home.json
```

This generates:
- `output.ttl` - ThingDescription artifacts in RDF/Turtle format
- `state.json` - Current device states in JSON format

### Step 3: View Your Results

**RDF Output (`output.ttl`)** contains:
- Workspace definitions (rooms as HMAS workspaces)
- Artifact definitions (devices as TD Things)
- ActionAffordances (device operations)
- PropertyAffordances (device attributes)

**JSON State (`state.json`)** contains:
- Current state of each device
- Property values indexed by artifact URI

## Common Use Cases

### Adding a New Device Type

Just add it to your input JSON - no code changes needed!

```json
{
    "room_name": "garage",
    "device_name": "door_opener",
    "operation": "open_door",
    "parameters": []
}
```

The converter automatically creates:
- Artifact name: `garageDoorOpener`
- Action: `openDoor`
- Command class: `OpenDoorCommand`

### Multiple Rooms and Devices

```json
{
    "home_status": {
        "bedroom": {
            "light": {...},
            "thermostat": {...}
        },
        "kitchen": {
            "light": {...},
            "refrigerator": {...}
        }
    }
}
```

Creates:
- 2 workspaces (bedroom, kitchen)
- 4 artifacts (bedroomLight, bedroomThermostat, kitchenLight, kitchenRefrigerator)

### Device with Multiple Parameters

```json
{
    "operation": "set_climate",
    "parameters": [
        {"name": "temperature", "type": "int"},
        {"name": "humidity", "type": "int"},
        {"name": "mode", "type": "str"}
    ]
}
```

Generates proper input schema with all parameters.

## Naming Conventions Summary

| Input                    | Output Type    | Example Output           |
|--------------------------|----------------|--------------------------|
| room: `master_bedroom`   | Workspace      | `master_bedroom`         |
| device: `light`          | Artifact       | `masterBedroomLight`     |
| operation: `turn_on`     | Action         | `turnOn`                 |
| operation: `set_brightness` | Action      | `setBrightness`          |
| attribute: `brightness`  | Property       | `brightness`             |

## Sample Output Structure

```
Room (Workspace)
└── Device (Artifact)
    ├── Actions (ActionAffordances)
    │   ├── turnOn
    │   ├── turnOff
    │   └── setBrightness
    └── Properties (PropertyAffordances)
        ├── state
        └── brightness
```

## Testing Your Conversion

Run the test suite:

```bash
python test_converter.py
```

Validates:
- Naming conventions
- RDF structure
- JSON state format
- Parameter handling
- Multiple devices
- Range constraints
- Enumerated options

## Need Help?

See the full `README.md` for:
- Complete input/output format specifications
- Detailed conversion rules
- Advanced usage examples
- Troubleshooting guide
- API reference

## Example Files Included

- `sample_input.json` - Example input with multiple devices
- `output.ttl` - Example RDF output
- `state.json` - Example JSON state

Try the converter with these examples:

```bash
python smart_home_to_td_converter.py sample_input.json
```
