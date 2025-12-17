# Smart Home Simulator

A FastAPI-based simulator that implements PropertyAffordance (GET) and ActionAffordance (POST) endpoints for smart home devices based on Thing Description (TD) artifact descriptions.

## Features

- **Dynamic Route Registration**: Automatically registers endpoints from TD artifact descriptions (.ttl files)
- **Device State Management**: Loads and maintains device states from JSON state files
- **16 Device Types Supported**: Lights, HVAC, blinds, media players, and more
- **Hierarchical Organization**: Supports multiple homes, rooms, and artifacts
- **Full Error Handling**: Proper HTTP status codes for errors (404, 400, 500)
- **RESTful API**: GET for properties, POST for actions

## Supported Device Types

- Light
- Heating
- Fan
- Air Conditioner
- Garage Door
- Blinds
- Curtain
- Media Player
- Vacuum Robot
- Trash
- Humidifier
- Dehumidifier
- Aromatherapy
- Water Heater
- Air Purifier
- Pet Feeder

## Installation

The simulator requires:
- FastAPI
- Uvicorn
- RDFLib

If using conda:
```bash
conda activate ami-agents
# Dependencies should already be installed
```

## Usage

### Starting the Simulator

```bash
python smart_home_simulator.py
```

The simulator will:
1. Load all home descriptions from `datasets/HomeBench/hmas_format/home_description/`
2. Parse `.ttl` files for artifact descriptions
3. Load initial states from `*_state.json` files
4. Register all property and action endpoints
5. Start the FastAPI server on `http://localhost:8080`

### API Endpoints

#### Root Endpoint
```bash
GET /
```
Returns API information and statistics.

**Example:**
```bash
curl http://localhost:8080/
```

**Response:**
```json
{
    "message": "Smart Home Simulator API",
    "version": "1.0.0",
    "devices": 4453,
    "property_endpoints": 8999,
    "action_endpoints": 13697
}
```

#### Health Check
```bash
GET /health
```
Returns health status.

#### Property Affordances (GET)

Get a property value from a device:

```bash
GET /workspaces/{home_id}/{room_name}/artifacts/{artifact_name}/properties/{property_name}
```

**Example:**
```bash
curl http://localhost:8080/workspaces/home0/balcony/artifacts/balconyAromatherapy/properties/state
```

**Response:**
```json
{
    "value": "off"
}
```

#### Action Affordances (POST)

Invoke an action on a device:

```bash
POST /workspaces/{home_id}/{room_name}/artifacts/{artifact_name}/{action_name}
Content-Type: application/json
```

**Example 1: Action without parameters**
```bash
curl -X POST http://localhost:8080/workspaces/home0/balcony/artifacts/balconyAromatherapy/turn_on \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
    "status": "success",
    "message": "Action 'turnOn' executed successfully"
}
```

**Example 2: Action with parameters**
```bash
curl -X POST http://localhost:8080/workspaces/home0/balcony/artifacts/balconyAromatherapy/set_interval \
  -H "Content-Type: application/json" \
  -d '{"interval": 45}'
```

**Response:**
```json
{
    "status": "success",
    "message": "Action 'setInterval' executed successfully"
}
```

## Error Handling

The simulator provides proper error responses:

### 404 - Endpoint Not Found
```bash
curl http://localhost:8080/workspaces/home0/nonexistent/properties/state
```

**Response:**
```json
{
    "error": "Property endpoint not found: /workspaces/home0/nonexistent/properties/state",
    "status_code": 404
}
```

### 400 - Bad Request (Missing Parameters)
```bash
curl -X POST http://localhost:8080/workspaces/home0/balcony/artifacts/balconyAromatherapy/set_interval \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
    "error": "Missing required parameter: interval",
    "status_code": 400
}
```

### 500 - Internal Server Error
Returned for unexpected server errors with detailed error information.

## Testing Examples

### Test a Light Device
```bash
# Get current state
curl http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/properties/state

# Turn on
curl -X POST http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/turn_on \
  -H "Content-Type: application/json"

# Verify state changed
curl http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomLight/properties/state
```

### Test Air Conditioner
```bash
# Get temperature
curl http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/properties/temperature

# Set temperature
curl -X POST http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/set_temperature \
  -H "Content-Type: application/json" \
  -d '{"temperature": 22}'

# Verify change
curl http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner/properties/temperature
```

### Test Curtain
```bash
# Get degree
curl http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomCurtain/properties/degree

# Set degree
curl -X POST http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomCurtain/set_degree \
  -H "Content-Type: application/json" \
  -d '{"degree": 50}'
```

## Architecture

### Device Models
Each device type inherits from the base `Device` class and implements:
- Property getters/setters
- Action methods (mapped from camelCase to snake_case)
- State management

### Route Registration
The simulator:
1. Parses RDF/Turtle files using rdflib
2. Extracts PropertyAffordances and ActionAffordances
3. Extracts target URLs from hypermedia forms
4. Dynamically registers FastAPI routes
5. Maps endpoints to device instances

### State Management
- Initial states loaded from JSON files
- State changes persist during runtime
- Each device maintains its own state dictionary

## Development

### Adding New Device Types

1. Create a new device class inheriting from `Device`:
```python
class NewDevice(Device):
    def get_device_type(self) -> str:
        return "new_device"

    def your_action(self, param: int):
        self.state['property'] = param
```

2. Add to `DEVICE_MAP`:
```python
DEVICE_MAP = {
    ...
    "NewDevice": NewDevice,
}
```

### Running in Development Mode

```bash
uvicorn smart_home_simulator:app --reload --host 0.0.0.0 --port 8080
```

## Notes

- The simulator maintains state only during runtime (no persistence to disk)
- All property names are sanitized to valid URI characters
- Action names are converted from camelCase to snake_case for method mapping
- The simulator supports all homes in the dataset simultaneously
