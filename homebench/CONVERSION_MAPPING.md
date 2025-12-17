# Conversion Mapping Reference

This document provides a comprehensive reference for how smart home JSON elements map to ThingDescription (TD) artifact components.

## High-Level Architecture

```
Smart Home JSON              →    ThingDescription Environment
─────────────────────────────────────────────────────────────────
home_status (rooms)          →    HMAS Workspaces
  ├── room devices           →    TD Artifacts (Things)
  │   ├── operations         →    ActionAffordances
  │   └── attributes         →    PropertyAffordances
  └── room_name              →    Workspace identifier

method (capabilities)        →    ActionAffordances
  ├── operation              →    Action name (camelCase)
  └── parameters             →    Input JSON Schema
```

## Detailed Mappings

### 1. Room → Workspace

**Input (JSON):**
```json
"home_status": {
    "master_bedroom": { ... }
}
```

**Output (RDF):**
```turtle
<http://localhost:8080/workspaces/master_bedroom#workspace>
    rdf:type hmas:Workspace ,
             td:Thing ;
    hmas:contains <...artifact URIs...> .
```

**Mapping Rules:**
- Room name used directly as workspace identifier
- All devices in room become contained artifacts
- Workspace URI: `{base_url}/workspaces/{room_name}#workspace`

---

### 2. Device → Artifact

**Input (JSON):**
```json
"master_bedroom": {
    "light": {
        "state": "on",
        "attributes": { ... }
    }
}
```

**Output (RDF):**
```turtle
<http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact>
    rdf:type ex:Light ,
             hmas:Artifact ,
             td:Thing ;
    hmas:isContainedIn <http://localhost:8080/workspaces/master_bedroom#workspace> ;
    td:hasActionAffordance [...] ;
    td:hasPropertyAffordance [...] ;
    td:title "Masterbedroomlight" .
```

**Mapping Rules:**
- Artifact name: camelCase(room_name + device_name)
  - Example: `master_bedroom` + `light` → `masterBedroomLight`
- Device type becomes device class: `ex:Light`, `ex:AirConditioner`
- Always includes types: `hmas:Artifact`, `td:Thing`
- Artifact URI: `{base_url}/workspaces/{room}/artifacts/{artifactName}#artifact`

---

### 3. Operation → ActionAffordance

**Input (JSON):**
```json
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
```

**Output (RDF):**
```turtle
[
    rdf:type ex:SetBrightnessCommand ,
             td:ActionAffordance ;
    td:hasForm [
        http:methodName "POST" ;
        hctl:forContentType "application/json" ;
        hctl:hasOperationType td:invokeAction ;
        hctl:hasTarget <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/set_brightness>
    ] ;
    td:hasInputSchema [
        rdf:type jsonschema:ObjectSchema ;
        jsonschema:properties [
            rdf:type jsonschema:IntegerSchema ;
            jsonschema:propertyName "brightness"
        ] ;
        jsonschema:required "brightness"
    ] ;
    td:name "setBrightness" ;
    td:title "setBrightness"
]
```

**Mapping Rules:**
- Operation name → camelCase action name
  - `turn_on` → `turnOn`
  - `set_brightness` → `setBrightness`
  - `set_target_temperature` → `setTargetTemperature`
- Operation → Command class: `{Operation}Command`
  - `turn_on` → `TurnOnCommand`
  - `set_brightness` → `SetBrightnessCommand`
- HTTP method: Always `POST`
- Content type: Always `application/json`
- Action URI: `{artifact_base}/`{operation}`

---

### 4. Parameters → Input Schema

**Parameter Type Mappings:**

| JSON Type | JSON Schema Type      | Example |
|-----------|-----------------------|---------|
| `"int"`   | `jsonschema:IntegerSchema` | Temperature, brightness |
| `"str"`   | `jsonschema:StringSchema` | Mode, name |
| `"bool"`  | `jsonschema:BooleanSchema` | Enabled, active |

**Input (JSON):**
```json
"parameters": [
    {"name": "temperature", "type": "int"},
    {"name": "mode", "type": "str"}
]
```

**Output (RDF):**
```turtle
td:hasInputSchema [
    rdf:type jsonschema:ObjectSchema ;
    jsonschema:properties [
        rdf:type jsonschema:IntegerSchema ;
        jsonschema:propertyName "temperature"
    ] ;
    jsonschema:properties [
        rdf:type jsonschema:StringSchema ;
        jsonschema:propertyName "mode"
    ] ;
    jsonschema:required "temperature" ,
                        "mode"
] ;
```

**Mapping Rules:**
- Each parameter becomes a property in ObjectSchema
- All parameters marked as required
- Type mapping: int→Integer, str→String, bool→Boolean

---

### 5. Attribute → PropertyAffordance

#### 5a. Attribute with Range

**Input (JSON):**
```json
"brightness": {
    "value": 83,
    "lowest": 0,
    "highest": 100
}
```

**Output (RDF):**
```turtle
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
]
```

**Mapping Rules:**
- Attribute name preserved as property name
- `lowest` → `jsonschema:minimum`
- `highest` → `jsonschema:maximum`
- HTTP method: Always `GET`
- Property URI: `{artifact_base}/properties/{property_name}`

#### 5b. Attribute with Enum Options

**Input (JSON):**
```json
"mode": {
    "value": "heat",
    "options": ["cool", "heat", "fan_only", "dry"]
}
```

**Output (RDF):**
```turtle
[
    rdf:type td:PropertyAffordance ;
    rdfs:comment "mode of masterBedroomAirConditioner" ;
    td:hasForm [
        http:methodName "GET" ;
        hctl:forContentType "application/json" ;
        hctl:hasOperationType td:readProperty ;
        hctl:hasTarget <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomAirConditioner/properties/mode>
    ] ;
    td:hasOutputSchema [
        rdf:type jsonschema:StringSchema ;
        jsonschema:enum "cool" ,
                        "heat" ,
                        "fan_only" ,
                        "dry"
    ] ;
    td:isObservable "true"^^xsd:boolean ;
    td:name "mode" ;
    td:title "mode"
]
```

**Mapping Rules:**
- `options` array → `jsonschema:enum` values
- Schema type: StringSchema for enums
- All enum values listed in RDF

#### 5c. State Property (Auto-generated)

**Input (JSON):**
```json
"light": {
    "state": "on",
    "attributes": { ... }
}
```

**Output (RDF):**
```turtle
[
    rdf:type td:PropertyAffordance ;
    rdfs:comment "state of masterBedroomLight" ;
    td:hasForm [
        http:methodName "GET" ;
        hctl:forContentType "application/json" ;
        hctl:hasOperationType td:readProperty ;
        hctl:hasTarget <http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/properties/state>
    ] ;
    td:hasOutputSchema [
        rdf:type jsonschema:StringSchema ;
        jsonschema:enum "on" ,
                        "off"
    ] ;
    td:isObservable "true"^^xsd:boolean ;
    td:name "state" ;
    td:title "state"
]
```

**Mapping Rules:**
- State automatically becomes a PropertyAffordance
- Always enum with values "on" and "off"
- Added in addition to explicit attributes

---

### 6. Device State → JSON Output

**Input (JSON - from home_status):**
```json
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
```

**Output (JSON State):**
```json
{
    "http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact": {
        "state": "on",
        "brightness": 83
    }
}
```

**Mapping Rules:**
- Key: Full artifact URI
- Values: PropertyAffordance names with current values
- State included as property
- Only `value` field extracted from attributes (constraints not included)

---

## Naming Convention Table

| Component | Input Format | Output Format | Example |
|-----------|--------------|---------------|---------|
| Workspace | `snake_case` | `snake_case` | `master_bedroom` |
| Artifact | `room_name` + `device_name` | `camelCase` | `masterBedroomLight` |
| Device Class | `device_name` | `PascalCase` | `Light`, `AirConditioner` |
| Action | `operation` | `camelCase` | `turnOn`, `setBrightness` |
| Command Class | `operation` | `PascalCase + "Command"` | `TurnOnCommand` |
| Property | `attribute_name` | `snake_case` (preserved) | `brightness`, `fan_speed` |

---

## URI Structure Reference

```
Base URL: http://localhost:8080

Workspace:
  {base_url}/workspaces/{room_name}#workspace
  Example: http://localhost:8080/workspaces/master_bedroom#workspace

Artifact:
  {base_url}/workspaces/{room_name}/artifacts/{artifactName}#artifact
  Example: http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight#artifact

Action Endpoint:
  {base_url}/workspaces/{room_name}/artifacts/{artifactName}/{operation}
  Example: http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/set_brightness

Property Endpoint:
  {base_url}/workspaces/{room_name}/artifacts/{artifactName}/properties/{property_name}
  Example: http://localhost:8080/workspaces/master_bedroom/artifacts/masterBedroomLight/properties/brightness
```

---

## Complete Example Mapping

### Input
```json
{
    "home_id": 99,
    "method": [
        {
            "room_name": "bedroom",
            "device_name": "light",
            "operation": "set_brightness",
            "parameters": [{"name": "brightness", "type": "int"}]
        }
    ],
    "home_status": {
        "bedroom": {
            "light": {
                "state": "on",
                "attributes": {
                    "brightness": {"value": 75, "lowest": 0, "highest": 100}
                }
            }
        }
    }
}
```

### Output Components

**1. Workspace:**
```turtle
<http://localhost:8080/workspaces/bedroom#workspace>
    rdf:type hmas:Workspace , td:Thing ;
    hmas:contains <http://localhost:8080/workspaces/bedroom/artifacts/bedroomLight#artifact> .
```

**2. Artifact with Action:**
```turtle
<http://localhost:8080/workspaces/bedroom/artifacts/bedroomLight#artifact>
    rdf:type ex:Light , hmas:Artifact , td:Thing ;
    hmas:isContainedIn <http://localhost:8080/workspaces/bedroom#workspace> ;
    td:hasActionAffordance [
        rdf:type ex:SetBrightnessCommand , td:ActionAffordance ;
        td:name "setBrightness" ;
        td:hasInputSchema [...]
    ] ;
    td:hasPropertyAffordance [...] .
```

**3. JSON State:**
```json
{
    "http://localhost:8080/workspaces/bedroom/artifacts/bedroomLight#artifact": {
        "state": "on",
        "brightness": 75
    }
}
```

---

## Type Inference Rules

When attribute has:
- `lowest` & `highest` → IntegerSchema with min/max constraints
- `options` array → StringSchema with enum values
- Just `value` (integer) → IntegerSchema (no constraints)
- Just `value` (string) → StringSchema (no constraints)
- Just `value` (boolean) → BooleanSchema (no constraints)

---

## Standard RDF Namespaces Used

```turtle
@prefix : <http://localhost:8080/workspaces/> .
@prefix ex: <http://example.org/> .
@prefix hctl: <https://www.w3.org/2019/wot/hypermedia#> .
@prefix hmas: <https://purl.org/hmas/> .
@prefix http: <http://www.w3.org/2011/http#> .
@prefix jsonschema: <https://www.w3.org/2019/wot/json-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix td: <https://www.w3.org/2019/wot/td#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
```
