# TODO: Ablation Findings & Next Steps

## Experiment Summary

Ran ablation experiments comparing 7 configurations across simple and complex goals using the HMAS smart home environment.

### Configurations Tested
| Config | Reasoning | Output | State |
|--------|-----------|--------|-------|
| baseline_json_ir | None | JSON IR | None |
| baseline_with_state | None | JSON IR | All |
| cot_reasoning | Chain of Thought | JSON IR | All |
| reflection_reasoning | Reflection | JSON IR | All |
| multi_turn_reasoning | Multi-turn | JSON IR | All |
| python_code_direct | None | Python | None |
| python_code_cot | Chain of Thought | Python | All |

### Results

**Simple Goal** ("Turn on the bathroom light"):
- All 7 configurations: SUCCESS

**Complex Goal** ("Turn on all lights in the living room and set the temperature to 22 degrees"):
- `baseline_json_ir` (no state): SUCCESS - but used wrong device (masterBedroomAirConditioner)
- `baseline_with_state`: SUCCESS - same wrong device selection
- `cot_reasoning`: FAILURE - correctly selected livingRoomHeating but hit min temp constraint (30°C)
- `reflection_reasoning`: FAILURE - same issue
- `multi_turn_reasoning`: FAILURE - same issue
- `python_code_direct`: SUCCESS - wrong device selection
- `python_code_cot`: FAILURE - same constraint issue

## Key Finding

**State information improves device selection but exposes constraint violations.**

Models with state context correctly identified `livingRoomHeating` as the appropriate device for the living room. However, the affordance schema didn't expose the temperature constraint (min: 30°C), so the LLM generated an invalid action (setTemperature: 22).

Models without state selected `masterBedroomAirConditioner` - wrong room but it happened to accept 22°C.

## Action Items

### High Priority

- [ ] **Enhance affordance parameter modeling**
  - Current: Only captures action name and basic input schema
  - Needed: Include min/max constraints, allowed values, data types
  - Location: `src/discovery/base.py` - `Affordance` dataclass
  - Example: `{"type": "number", "minimum": 30, "maximum": 40}` should be parsed and surfaced

- [ ] **Parse Thing Description constraints**
  - HMAS Thing Descriptions include JSON Schema constraints
  - Extract `minimum`, `maximum`, `enum`, `pattern` from action input schemas
  - Surface these in the `CapabilityModel` passed to the LLM

- [ ] **Update prompts to highlight constraints**
  - Modify `src/prompts/ir/detailed.py` to clearly show parameter constraints
  - Format: "setTemperature(value: number, min=30, max=40)"

### Medium Priority

- [ ] **Add constraint validation in planning**
  - Before generating the final plan, validate parameters against known constraints
  - Return planning error if constraints would be violated

- [ ] **Implement "relevant" affordance strategy**
  - Currently stubbed out
  - Should filter affordances based on goal keywords and device types

- [ ] **Add execution result feedback loop**
  - When execution fails due to constraint violation, feed error back to planner
  - Allow re-planning with constraint awareness

### Low Priority

- [ ] **Expand test scenarios**
  - Add scenarios that specifically test constraint boundaries
  - Test enum-constrained actions (e.g., mode: "cool" | "heat" | "auto")

- [ ] **Add metrics collection**
  - Track: planning time, token usage, retry count, constraint violations
  - Compare across configurations

- [ ] **Improve agentic discovery**
  - Current implementation is basic
  - Could use LLM to intelligently navigate workspace hierarchy

## Architecture Notes

The modular structure supports these improvements:

```
src/discovery/
├── base.py          # Add constraint fields to Affordance dataclass
└── affordances/
    └── exhaustive.py  # Parse constraints from Thing Descriptions

src/planning/
├── planner.py       # Add pre-validation step
└── output/
    └── json_ir.py   # Include constraints in prompt context
```

## References

- Thing Description spec: https://www.w3.org/TR/wot-thing-description/
- JSON Schema validation: https://json-schema.org/understanding-json-schema/reference/numeric.html
