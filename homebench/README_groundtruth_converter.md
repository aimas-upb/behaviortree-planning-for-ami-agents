# HomeBench Ground Truth Converter

This script converts HomeBench ground truth data from the original format to ThingDescription representation.

## Requirements

```bash
pip install rdflib
```

## Usage

```bash
python3 ground_truth_converter.py -i INPUT_FILE -o OUTPUT_FILE [-t TTL_DIR]
```

### Arguments

- `-i, --input`: Input JSONL file (required)
  - Example: `datasets/HomeBench/original/train_data_part1.jsonl`

- `-o, --output`: Output JSON file path (required)
  - Example: `datasets/HomeBench/converted/train_data_part1.json`

- `-t, --ttl-dir`: Directory containing TTL files (optional)
  - Default: `datasets/HomeBench/hmas_format`

### Examples

Convert training data part 1:
```bash
python3 ground_truth_converter.py \
  -i datasets/HomeBench/original/train_data_part1.jsonl \
  -o datasets/HomeBench/converted/train_data_part1.json
```

Convert validation data:
```bash
python3 ground_truth_converter.py \
  -i datasets/HomeBench/original/valid_data.jsonl \
  -o datasets/HomeBench/converted/valid_data.json
```

Convert all datasets:
```bash
for file in datasets/HomeBench/original/*.jsonl; do
    filename=$(basename "$file" .jsonl)
    python3 ground_truth_converter.py \
        -i "$file" \
        -o "datasets/HomeBench/converted/${filename}.json"
done
```

## Output Format

The script converts entries from:

**Original format:**
```json
{
  "id": "home76_multi_201",
  "input": "Turn off the lights in the living room",
  "output": "'''living_room.light.turn_off()'''",
  "home_id": 76,
  "type": "normal"
}
```

**Converted format:**
```json
{
  "id": "home76_multi_201",
  "input": "Turn off the lights in the living room",
  "output": [
    {
      "execution": "success",
      "affordance": "http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight/turn_off",
      "params": {},
      "test": {
        "property": "http://localhost:8080/workspaces/home76/living_room/artifacts/livingRoomLight/properties/state",
        "expected_value": "off"
      }
    }
  ]
}
```

### Output Schema

Each action in the output list has the following structure:

**For errors:**
```json
{
  "execution": "error_input"
}
```

**For successful actions:**
```json
{
  "execution": "success",
  "affordance": "http://localhost:8080/workspaces/home76/room_name/artifacts/artifactName/action_name",
  "params": {
    "param_name": param_value
  },
  "test": {
    "property": "http://localhost:8080/workspaces/home76/room_name/artifacts/artifactName/properties/property_name",
    "expected_value": expected_value
  }
}
```

The `test` field specifies:
- `property`: The URL of the artifact property to check after executing the action
- `expected_value`: The value that the property should have if the action succeeded

#### Test Mapping Rules:

The script automatically determines test information based on action patterns:

- **State-changing actions** (`turn_on`, `turn_off`, `open`, `close`):
  - Property: `state`
  - Expected value: `"on"` or `"off"`

- **Setter actions** (`set_temperature`, `set_brightness`, `set_degree`, etc.):
  - Property: extracted from action name (e.g., `set_temperature` → `temperature`)
  - Expected value: the parameter value passed to the action

## How It Works

1. **Extracts home_id** from the entry ID (e.g., `home76_multi_201` → 76)
2. **Loads TTL file** for the corresponding home (`home_76.ttl`)
3. **Builds affordance and property maps** from the ThingDescription in the TTL
4. **Parses the output** string by splitting on commas
5. **For each action:**
   - Identifies if it's an error (`error_input`) or an action call
   - Parses the action call format: `room.device.action(params)`
   - Matches to the corresponding affordance URL in the TTL file
   - Extracts parameter schemas from the ThingDescription
   - Determines the test property and expected value based on the action
   - Constructs the converted output format with test information

## Notes

- The script automatically handles parameter extraction from ThingDescription schemas
- Parameters can be positional (e.g., `set_temperature(20)`) or named (e.g., `set_temperature(temperature=20)`)
- If an affordance cannot be found in the TTL file, it's marked as `error_input`
- Progress is printed every 100 entries
