# Smart Home to ThingDescription Converter

A Python script that converts smart home device capabilities and states from JSON format into W3C ThingDescription (TD) artifact-based Hypermedia environment representations.

## Overview

This converter transforms smart home configuration data into semantic web representations using:
- **RDF/Turtle format** for device capabilities (ThingDescriptions with ActionAffordances and PropertyAffordances)
- **JSON format** for current device states
- **HMAS (Hypermedia Multi-Agent Systems)** ontology for workspace organization

## Features

- ✅ Converts rooms to HMAS workspaces
- ✅ Transforms devices into TD artifacts with proper containment relationships
- ✅ Maps device operations to ActionAffordances with appropriate input schemas
- ✅ Converts device attributes to PropertyAffordances with output schemas
- ✅ Generates camelCase naming conventions for artifacts
- ✅ Handles multiple device types (lights, air conditioners, thermostats, etc.)
- ✅ Preserves attribute constraints (ranges, enumerations)
- ✅ Creates separate JSON state representation

## Installation

No external dependencies required - uses only Python 3 standard library.

```bash
# Make the script executable (optional)
chmod +x smart_home_to_td_converter.py
```

## Usage

### Basic Usage

```bash
python smart_home_to_td_converter.py <input_json_file> [output_rdf_file] [output_state_file]
```

### Parameters

- `input_json_file` (required): Path to the input JSON file containing smart home data
- `output_rdf_file` (optional): Path for the output RDF/Turtle file (default: `output.ttl`)
- `output_state_file` (optional): Path for the output JSON state file (default: `state.json`)

### Example

```bash
python smart_home_to_td_converter.py sample_input.json artifacts.ttl device_states.json
```

## Input Format

The input JSON must follow this structure:

```json
{
    "home_id": 99,
    "method": [
        {
            "room_name": "master_bedroom",
            "device_name": "light",
            "operation": "turn_on",
            "parameters": []
        },
        {
            "room_name": "master_bedroom",
            "device_name": "light",
            "operation": "set_brightness",
            "parameters": [
                {
                    "name": "brightness",
                    "type": "int"
                }
            ]
        }
    ],
    "home_status": {
        "master_bedroom": {
            "room_name": "master_bedroom",
            "light": {
                "state": "on",
                "attributes": {
                    "brightness": {
                        "value": 83,
                        "lowest": 0,
                        "highest": 100
                    }
                }
            }
        }
    }
}
```

### Input Structure Details

#### `method` Section
Defines device capabilities:
- `room_name`: Name of the room containing the device
- `device_name`: Type of device (e.g., "light", "air_conditioner")
- `operation`: Operation name (e.g., "turn_on", "set_brightness")
- `parameters`: List of parameters with `name` and `type` fields

#### `home_status` Section
Defines current device states:
- Room-level organization
- Device-level state and attributes
- `state`: Current on/off state
- `attributes`: Dictionary of device properties with:
  - `value`: Current value
  - `lowest`/`highest`: Numeric range constraints (optional)
  - `options`: List of valid enum values (optional)

## Output Format

### RDF/Turtle Output

The converter generates W3C ThingDescription artifacts in RDF/Turtle format:

```turtle
<http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact>
    rdf:type ex:Light ,
             hmas:Artifact ,
             td:Thing ;
    hmas:isContainedIn <http://localhost:8080/workspaces/master_bedroom#workspace> ;
    td:hasActionAffordance
        [
            rdf:type ex:TurnOnCommand ,
                     td:ActionAffordance ;
            td:hasForm [
                http:methodName "POST" ;
                hctl:forContentType "application/json" ;
                hctl:hasOperationType td:invokeAction ;
                hctl:hasTarget <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/turn_on>
            ] ;
            td:name "turnOn" ;
            td:title "turnOn"
        ] ;
    td:hasPropertyAffordance
        [
            rdf:type td:PropertyAffordance ;
            rdfs:comment "brightness of masterBedroomLight" ;
            td:hasForm [
                http:methodName "GET" ;
                hctl:forContentType "application/json" ;
                hctl:hasOperationType td:readProperty ;
                hctl:hasTarget <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/properties/brightness>
            ] ;
            td:hasOutputSchema [
                rdf:type jsonschema:IntegerSchema ;
                jsonschema:minimum 0 ;
                jsonschema:maximum 100
            ] ;
            td:isObservable "true"^^xsd:boolean ;
            td:name "brightness" ;
            td:title "brightness"
        ] ;
    td:title "Masterbedroomlight" .

<http://localhost:8080/workspaces/master_bedroom#workspace>
    rdf:type hmas:Workspace ,
             td:Thing ;
    hmas:contains
        <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact> .
```

### JSON State Output

Current device states following PropertyAffordance naming:

```json
{
    "http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact": {
        "state": "on",
        "brightness": 83
    }
}
```

## Conversion Rules

### Naming Conventions

1. **Workspace Names**: Direct mapping from room names
   - `master_bedroom` → `master_bedroom` workspace

2. **Artifact Names**: camelCase combining room and device
   - Room: `master_bedroom`, Device: `light` → `masterBedroomLight`
   - Room: `living_room`, Device: `air_conditioner` → `livingRoomAirConditioner`

3. **Action Names**: camelCase from operations
   - `turn_on` → `turnOn`
   - `set_brightness` → `setBrightness`
   - `set_target_temperature` → `setTargetTemperature`

4. **Property Names**: Direct mapping from attribute names
   - `brightness`, `temperature`, `mode`, etc.

### Type Mappings

#### ActionAffordances
- Each device operation becomes a `td:ActionAffordance`
- Operation command classes: `ex:TurnOnCommand`, `ex:SetBrightnessCommand`, etc.
- Parameters generate `jsonschema:ObjectSchema` input schemas
- Parameter types: `int` → `IntegerSchema`, `str` → `StringSchema`, `bool` → `BooleanSchema`

#### PropertyAffordances
- Each device attribute becomes a `td:PropertyAffordance`
- State property is automatically added with on/off enum
- Attributes with `lowest`/`highest` generate range-constrained integer schemas
- Attributes with `options` generate string enum schemas
- All properties are marked as observable

### URL Structure

- Base URL: `http://localhost:8080` (configurable)
- Workspace: `{base_url}/workspaces/{room_name}#workspace`
- Artifact: `{base_url}/workspaces/{room_name}/artifacts/{artifactName}#artifact`
- Action: `{base_url}/workspaces/{room_name}/artifacts/{artifactName}/{operation}`
- Property: `{base_url}/workspaces/{room_name}/artifacts/{artifactName}/properties/{property_name}`

## Supported Device Types

The converter is generic and supports any device type. Tested examples include:

- **Lights**: on/off, brightness control
- **Air Conditioners**: temperature, mode, fan speed, swing
- **Thermostats**: current/target temperature
- **And more**: The converter adapts to any device structure

## Namespaces Used

- `hmas`: https://purl.org/hmas/
- `td`: https://www.w3.org/2019/wot/td#
- `hctl`: https://www.w3.org/2019/wot/hypermedia#
- `jsonschema`: https://www.w3.org/2019/wot/json-schema#
- `ex`: http://example.org/
- `rdf`: http://www.w3.org/1999/02/22-rdf-syntax-ns#
- `rdfs`: http://www.w3.org/2000/01/rdf-schema#
- `http`: http://www.w3.org/2011/http#
- `xsd`: http://www.w3.org/2001/XMLSchema#

## Example Conversion

### Input: Light in Master Bedroom

```json
{
    "method": [
        {
            "room_name": "master_bedroom",
            "device_name": "light",
            "operation": "set_brightness",
            "parameters": [{"name": "brightness", "type": "int"}]
        }
    ],
    "home_status": {
        "master_bedroom": {
            "light": {
                "state": "on",
                "attributes": {
                    "brightness": {
                        "value": 83,
                        "lowest": 0,
                        "highest": 100
                    }
                }
            }
        }
    }
}
```

### Output: TD Artifact

- **Artifact Name**: `masterBedroomLight`
- **Workspace**: `master_bedroom`
- **ActionAffordance**: `setBrightness` with integer input parameter
- **PropertyAffordances**: 
  - `brightness` (integer, 0-100 range)
  - `state` (enum: on/off)
- **JSON State**: `{"state": "on", "brightness": 83}`

## Advanced Usage

### Custom Base URL

Modify the base URL by editing the converter initialization:

```python
converter = SmartHomeToTDConverter(base_url="http://your-domain.com:8080")
```

### Batch Processing

Process multiple files:

```bash
for file in inputs/*.json; do
    python smart_home_to_td_converter.py "$file" "outputs/$(basename $file .json).ttl" "outputs/$(basename $file .json)_state.json"
done
```

## Validation

The generated RDF/Turtle can be validated using standard RDF tools:

```bash
# Using rapper (from raptor-utils)
rapper -i turtle -o ntriples output.ttl > /dev/null

# Using riot (from Apache Jena)
riot --validate output.ttl
```

## Troubleshooting

### Common Issues

1. **Invalid JSON**: Ensure input JSON is well-formed
2. **Missing fields**: All required fields (`room_name`, `device_name`, `operation`) must be present
3. **Type mismatches**: Parameter types should be "int", "str", or "bool"

### Error Messages

- `Input file not found`: Check the input file path
- `Invalid JSON in input file`: Validate JSON syntax
- `KeyError`: Missing required field in input structure

## Output Examples

See the `sample_input.json`, `output.ttl`, and `state.json` files for complete working examples.

## License

This script is provided as-is for dataset preparation purposes.

## Contributing

Suggestions and improvements welcome! Key areas for enhancement:
- Additional JSON schema constraints
- Support for event subscriptions
- Custom namespace configuration
- Validation of generated output

## References

- [W3C Web of Things (WoT) Thing Description](https://www.w3.org/TR/wot-thing-description/)
- [Hypermedia Multi-Agent Systems (HMAS)](https://purl.org/hmas/)
- [JSON Schema](https://json-schema.org/)
- [RDF 1.1 Turtle](https://www.w3.org/TR/turtle/)
