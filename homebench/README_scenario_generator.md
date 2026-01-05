# HomeBench Scenario Generator

This script generates new scenarios to extend the HomeBench dataset with two new types of queries:

1. **Future Scheduling**: Augments existing examples with time constraints
2. **Dependency Scheduling**: Creates queries that link two devices where one action triggers another

## Usage

```bash
python3 scenario_generator.py --mode MODE [OPTIONS]
```

### Modes

#### Future Scheduling (`--mode future`)

Augments existing HomeBench entries with time constraints. Takes existing simple commands and adds scheduling context.

**Example transformations:**
- `"Turn off the lights"` → `"Turn off the lights in 10 minutes"`
- `"Set temperature to 20"` → `"I will go to sleep in 30 minutes. Can you set temperature to 20?"`

```bash
python3 homebench/scenario_generator.py --mode future \
    --input datasets/HomeBench/original/train_data_part1.jsonl \
    --output datasets/HomeBench/extended/future_scheduling.jsonl \
    --sample-ratio 0.3 \
    --seed 42
```

**Output format:**
```json
{
  "id": "future_home76_one_133",
  "input": "In 10 minutes, set the volume of the media player to 50 in the master bedroom.",
  "output": "'''schedule(10, master_bedroom.media_player.set_volume(50)),'''",
  "home_id": 76,
  "type": "future_normal",
  "original_id": "home76_one_133",
  "delay_minutes": 10
}
```

#### Dependency Scheduling (`--mode dependency`)

Creates queries that link two devices where one action triggers another. Generates scenarios for both same-room and cross-room dependencies.

**Example queries:**
- `"When the light in the living room is turned on, set the air conditioner to cool mode."`
- `"After the dishwasher finishes, turn off the kitchen lights."`

```bash
python3 homebench/scenario_generator.py --mode dependency \
    --home-desc datasets/HomeBench/original/home-description.json \
    --output datasets/HomeBench/extended/dependency_scheduling.jsonl \
    --scenarios-per-room 2 \
    --cross-room-scenarios 5 \
    --seed 42
```

**Output format:**
```json
{
  "id": "home76_dep_1",
  "input": "When the light in the living room is turned on, set the air conditioner to cool mode in the living room.",
  "output": "'''when_completed(living_room.light.turn_on(), living_room.air_conditioner.set_mode(cool)),'''",
  "home_id": 76,
  "type": "dependency_same_room"
}
```

#### Both Modes (`--mode both`)

Generates both types of scenarios and saves them to separate files in a directory.

```bash
python3 homebench/scenario_generator.py --mode both \
    --input datasets/HomeBench/original/train_data_part1.jsonl \
    --home-desc datasets/HomeBench/original/home-description.json \
    --output-dir datasets/HomeBench/extended/ \
    --seed 42
```

### Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--mode` | Type of scenarios: `future`, `dependency`, or `both` | Yes |
| `-i, --input` | Input JSONL file for future scheduling | For `future`/`both` |
| `--home-desc` | Home descriptions JSON file | For `dependency`/`both` |
| `-o, --output` | Output file path | For single mode |
| `--output-dir` | Output directory | For `both` mode |
| `--sample-ratio` | Ratio of entries to sample for future scheduling (default: 0.3) | No |
| `--max-samples` | Maximum number of future scheduling samples | No |
| `--scenarios-per-room` | Dependency scenarios per room (default: 2) | No |
| `--cross-room-scenarios` | Cross-room dependency scenarios per home (default: 5) | No |
| `--seed` | Random seed for reproducibility | No |
| `--format` | Output format: `jsonl` or `json` (default: jsonl) | No |

## Output Schema

### Future Scheduling Output

The output wraps original actions in a `schedule()` call:

```
schedule(delay_minutes, original_action)
```

Where:
- `delay_minutes`: The number of minutes to delay execution
- `original_action`: The original HomeBench action format

### Dependency Scheduling Output

The output uses a `when_completed()` call to link trigger and response:

```
when_completed(trigger_action, response_action)
```

Where:
- `trigger_action`: The action that triggers the dependency (e.g., `living_room.light.turn_on()`)
- `response_action`: The action to execute when trigger completes (e.g., `living_room.air_conditioner.set_mode(cool)`)

## Scenario Types

### Future Scheduling Types

The generator uses various natural language patterns:

**Prefix patterns:**
- "In 10 minutes, ..."
- "After 30 minutes, ..."
- "Please schedule for 1 hour from now: ..."

**Suffix patterns:**
- "... in 15 minutes"
- "... after 20 minutes"

**Context patterns (with reasons):**
- "I will go to sleep in 10 minutes. Can you ...?"
- "I'm leaving the house in 30 minutes. Please ..."
- "We have guests arriving in 1 hour. ..."

### Dependency Scheduling Types

**Same-room dependencies** (`dependency_same_room`):
Links two devices within the same room.

**Cross-room dependencies** (`dependency_cross_room`):
Links devices across different rooms.

**Trigger conditions include:**
- Device turned on/off
- Device completion (for operational devices)
- State changes (e.g., reaching target temperature)

**Response actions include:**
- Turn on/off
- Set mode/temperature/brightness
- Open/close

## Examples

### Generate a small test set:

```bash
# Generate 50 future scheduling scenarios
python3 homebench/scenario_generator.py --mode future \
    --input datasets/HomeBench/original/train_data_part1.jsonl \
    --output /tmp/future_test.jsonl \
    --max-samples 50 \
    --seed 42

# Generate dependency scenarios for homes 0-9
python3 homebench/scenario_generator.py --mode dependency \
    --home-desc datasets/HomeBench/original/home-description.json \
    --output /tmp/dependency_test.jsonl \
    --scenarios-per-room 1 \
    --cross-room-scenarios 2 \
    --seed 42
```

### Generate full dataset extension:

```bash
# Generate all types with default settings
python3 homebench/scenario_generator.py --mode both \
    --input datasets/HomeBench/original/train_data_part1.jsonl \
    --home-desc datasets/HomeBench/original/home-description.json \
    --output-dir datasets/HomeBench/extended/ \
    --format json \
    --seed 42
```

## Integration with Ground Truth Converter

The generated scenarios can be converted to ThingDescription format using the `ground_truth_converter.py` script. The new output formats (`schedule()` and `when_completed()`) would need to be handled by extending the converter.

### Extending the converter

To support the new scenario types, the `ground_truth_converter.py` would need:

1. **For `schedule()`**: Parse the delay and inner action, then create output with scheduling metadata
2. **For `when_completed()`**: Parse both trigger and response actions, then create output with dependency information

Example converted output for future scheduling:
```json
{
  "execution": "success",
  "affordance": "http://localhost:8080/.../turn_off",
  "params": {},
  "schedule": {
    "delay_minutes": 10,
    "absolute_time": "2024-01-15T10:30:00Z"
  }
}
```

Example converted output for dependency scheduling:
```json
{
  "execution": "success",
  "trigger": {
    "affordance": "http://localhost:8080/.../turn_on",
    "condition": "completed"
  },
  "response": {
    "affordance": "http://localhost:8080/.../set_mode",
    "params": {"mode": "cool"}
  }
}
```
