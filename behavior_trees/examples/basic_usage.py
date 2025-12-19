"""
Example Usage of Behavior Tree Affordance Nodes

This script demonstrates how to use the behavior tree templates for
working with HMAS Thing Descriptions from HomeBench and Blocksworld datasets.

Running examples:
    python -m behavior_trees.examples.basic_usage
"""

import py_trees
from py_trees.common import Status
import logging

# Configure logging to see node activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from behavior_trees import (
    ActionAffordanceNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator,
)


# =============================================================================
# Example 1: HomeBench - Smart Home Light Control
# =============================================================================

def create_light_control_tree() -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree for controlling a smart home light.
    
    This tree:
    1. Checks if the light is off
    2. If off, turns it on
    3. Sets the brightness to 75%
    
    Uses a selector (fallback) pattern with sequence for conditional execution.
    """
    
    # Base URL for the light artifact
    light_base = "http://localhost:8080/workspaces/home0/bathroom/artifacts/bathroomLight"
    
    # Condition: Check if light is already on
    is_light_on = PropertyConditionNode(
        name="IsLightOn",
        property_url=f"{light_base}/properties/state",
        expected_value="on"
    )
    
    # Action: Turn on the light
    turn_on_light = ActionAffordanceNode(
        name="TurnOnLight",
        action_url=f"{light_base}/turn_on"
    )
    
    # Action: Set color
    set_color = ActionAffordanceNode(
        name="SetColor",
        action_url=f"{light_base}/set_color",
        parameters={"color": [100, 100, 75]}  # HSL values
    )
    
    # Sequence: Turn on then set color
    turn_on_sequence = py_trees.composites.Sequence(
        name="TurnOnAndConfigure",
        memory=True,
        children=[turn_on_light, set_color]
    )
    
    # Selector: Either light is already on, or turn it on
    ensure_light_on = py_trees.composites.Selector(
        name="EnsureLightOn",
        memory=False,
        children=[is_light_on, turn_on_sequence]
    )
    
    return ensure_light_on


def create_ac_control_tree() -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree for air conditioner temperature control.
    
    This tree:
    1. Reads current temperature
    2. If temperature > 25, sets mode to cool
    3. Sets target temperature to 22
    """
    
    ac_base = "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner"
    
    # Condition: Check if temperature is above threshold
    is_too_hot = ComparisonPropertyConditionNode(
        name="IsTooHot",
        property_url=f"{ac_base}/properties/temperature",
        expected_value=25,
        operator=ComparisonOperator.GREATER_THAN
    )
    
    # Action: Set mode to cool
    set_cool_mode = ActionAffordanceNode(
        name="SetCoolMode",
        action_url=f"{ac_base}/set_mode",
        parameters={"mode": "cool"}
    )
    
    # Action: Set temperature
    set_temperature = ActionAffordanceNode(
        name="SetTemperature",
        action_url=f"{ac_base}/set_temperature",
        parameters={"temperature": 18}
    )
    
    # Sequence: If too hot, cool down
    cooling_sequence = py_trees.composites.Sequence(
        name="CoolingSequence",
        memory=True,
        children=[is_too_hot, set_cool_mode, set_temperature]
    )
    
    return cooling_sequence


# =============================================================================
# Example 2: Blocksworld - Block Manipulation
# =============================================================================

def create_stack_blocks_tree(
    world_id: str = "world-1",
    source_block: str = "a",
    target_block: str = "b"
) -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree for stacking one block on another in Blocksworld.
    
    This tree implements the classic blocks world operation:
    1. Check if hand is empty
    2. Pick up the source block
    3. Stack it on the target block
    
    Args:
        world_id: The world instance identifier
        source_block: Block to pick up
        target_block: Block to stack onto
    """
    
    base_url = f"http://localhost:8081/workspaces/blocksworld/artifacts/{world_id}"
    
    # Condition: Check if hand is empty
    hand_empty = PropertyConditionNode(
        name="IsHandEmpty",
        property_url=f"{base_url}/properties/state",
        expected_value="empty",
        value_path=["hand"]
    )
    
    # Action: Pick up the source block
    pickup_block = ActionAffordanceNode(
        name=f"Pickup_{source_block}",
        action_url=f"{base_url}/pickup",
        parameters={"target_block": source_block}
    )
    
    # Action: Stack on target
    stack_block = ActionAffordanceNode(
        name=f"Stack_{source_block}_on_{target_block}",
        action_url=f"{base_url}/stack",
        parameters={
            "target_block": source_block,
            "to_block": target_block
        }
    )
    
    # Main sequence
    stack_sequence = py_trees.composites.Sequence(
        name=f"Stack_{source_block}_on_{target_block}_Sequence",
        memory=True,
        children=[hand_empty, pickup_block, stack_block]
    )
    
    return stack_sequence


def create_unstack_blocks_tree(
    world_id: str = "world-1",
    top_block: str = "a",
    bottom_block: str = "b"
) -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree for unstacking a block from another.
    
    Args:
        world_id: The world instance identifier
        top_block: Block on top to remove
        bottom_block: Block underneath
    """
    
    base_url = f"http://localhost:8081/workspaces/blocksworld/artifacts/{world_id}"
    
    # Condition: Check if hand is empty
    hand_empty = PropertyConditionNode(
        name="IsHandEmpty",
        property_url=f"{base_url}/properties/state",
        expected_value="empty",
        value_path=["hand"]
    )
    
    # Action: Unstack the top block
    unstack_block = ActionAffordanceNode(
        name=f"Unstack_{top_block}_from_{bottom_block}",
        action_url=f"{base_url}/unstack",
        parameters={
            "target_block": top_block,
            "from_block": bottom_block
        }
    )
    
    # Action: Put down the block on table
    putdown_block = ActionAffordanceNode(
        name=f"Putdown_{top_block}",
        action_url=f"{base_url}/putdown",
        parameters={"target_block": top_block}
    )
    
    # Main sequence
    unstack_sequence = py_trees.composites.Sequence(
        name=f"Unstack_{top_block}_from_{bottom_block}_Sequence",
        memory=True,
        children=[hand_empty, unstack_block, putdown_block]
    )
    
    return unstack_sequence


# =============================================================================
# Example 3: Dynamic Parameters from Blackboard
# =============================================================================

def create_dynamic_color_tree() -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree that reads color from blackboard.
    
    This demonstrates how to use dynamic parameters that can be
    set at runtime via the blackboard.
    """
    
    light_base = "http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight"

    # Property node to read current color
    get_color = PropertyAffordanceNode(
        name="GetColor",
        result_key="user/selected_color",
        property_url=f"{light_base}/properties/color"
    )

    change_color = ActionAffordanceNode(
        name="ChangeColor",
        action_url=f"{light_base}/set_color",
        parameters={"color": [255, 0, 0]}  # Red
    )

    # Action with dynamic parameter from blackboard
    set_color = ActionAffordanceNode(
        name="SetColorFromBlackboard",
        action_url=f"{light_base}/set_color",
        parameter_keys={"color": "user/selected_color"}
    )

    # Sequence to get and set color
    color_sequence = py_trees.composites.Sequence(
        name="ColorSequence",
        memory=True,
        children=[get_color, change_color, set_color]
    )

    return color_sequence


# =============================================================================
# Example 4: Complex Tree with Multiple Devices
# =============================================================================

def create_room_setup_tree() -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree for setting up a room with multiple devices.
    
    This tree runs multiple device configurations in parallel:
    - Turn on lights
    - Set AC to comfortable temperature
    - Start media player
    """
    
    room_base = f"http://localhost:8080/workspaces/home0/living_room/artifacts"
    
    # Light control sequence
    light_on = ActionAffordanceNode(
        name="TurnOnLight",
        action_url=f"{room_base}/livingRoomLight/turn_on"
    )
    
    # AC control sequence
    ac_on = ActionAffordanceNode(
        name="TurnOnAC",
        action_url=f"{room_base}/livingRoomAirConditioner/turn_on"
    )
    
    set_ac_temp = ActionAffordanceNode(
        name="SetACTemp",
        action_url=f"{room_base}/livingRoomAirConditioner/set_temperature",
        parameters={"temperature": 24}
    )
    
    ac_sequence = py_trees.composites.Sequence(
        name="ACSetup",
        memory=True,
        children=[ac_on, set_ac_temp]
    )
    
    # Run light and AC in parallel
    room_setup = py_trees.composites.Parallel(
        name="RoomSetup",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(),
        children=[light_on, ac_sequence]
    )
    
    return room_setup


# =============================================================================
# Example 5: Condition-Based Device Control
# =============================================================================

def create_temperature_based_control() -> py_trees.behaviour.Behaviour:
    """
    Create a behavior tree that controls AC based on temperature ranges.
    
    This uses ComparisonPropertyConditionNode for range checks.
    """
    
    ac_base = "http://localhost:8080/workspaces/home0/master_bedroom/artifacts/masterBedroomAirConditioner"
    
    # Check temperature ranges
    is_very_hot = ComparisonPropertyConditionNode(
        name="IsVeryHot",
        property_url=f"{ac_base}/properties/temperature",
        expected_value=29,
        operator=ComparisonOperator.GREATER_THAN_OR_EQUAL
    )
    
    is_hot = ComparisonPropertyConditionNode(
        name="IsHot",
        property_url=f"{ac_base}/properties/temperature",
        expected_value=25,
        operator=ComparisonOperator.GREATER_THAN
    )
    
    is_cold = ComparisonPropertyConditionNode(
        name="IsCold",
        property_url=f"{ac_base}/properties/temperature",
        expected_value=18,
        operator=ComparisonOperator.LESS_THAN_OR_EQUAL
    )
    
    # Actions for different conditions
    max_cooling = ActionAffordanceNode(
        name="MaxCooling",
        action_url=f"{ac_base}/set_temperature",
        parameters={"temperature": 18}
    )
    
    moderate_cooling = ActionAffordanceNode(
        name="ModerateCooling",
        action_url=f"{ac_base}/set_temperature",
        parameters={"temperature": 22}
    )
    
    heating = ActionAffordanceNode(
        name="Heating",
        action_url=f"{ac_base}/set_mode",
        parameters={"mode": "heat"}
    )
    
    # Very hot sequence
    very_hot_response = py_trees.composites.Sequence(
        name="VeryHotResponse",
        memory=True,
        children=[is_very_hot, max_cooling]
    )
    
    # Hot sequence
    hot_response = py_trees.composites.Sequence(
        name="HotResponse",
        memory=True,
        children=[is_hot, moderate_cooling]
    )
    
    # Cold sequence
    cold_response = py_trees.composites.Sequence(
        name="ColdResponse",
        memory=True,
        children=[is_cold, heating]
    )
    
    # Priority selector - check conditions in order
    temperature_control = py_trees.composites.Selector(
        name="TemperatureControl",
        memory=False,
        children=[very_hot_response, hot_response, cold_response]
    )
    
    return temperature_control


# =============================================================================
# Running the Examples
# =============================================================================

def run_tree(tree: py_trees.behaviour.Behaviour, ticks: int = 3) -> None:
    """
    Run a behavior tree for a specified number of ticks.
    
    Args:
        tree: The behavior tree to run
        ticks: Number of ticks to execute
    """
    print(f"\n{'='*60}")
    print(f"Running tree: {tree.name}")
    print(f"{'='*60}")
    
    # Setup the tree
    tree.setup_with_descendants()
    
    # Print tree structure
    print("\nTree structure:")
    print(py_trees.display.unicode_tree(tree))
    
    # Run ticks
    for i in range(ticks):
        print(f"\n--- Tick {i + 1} ---")
        tree.tick_once()
        print(f"Status: {tree.status}")
        
        if tree.status in (Status.SUCCESS, Status.FAILURE):
            break
    
    # Shutdown
    tree.shutdown()


def main():
    """Run all example behavior trees."""
    run_tree(create_light_control_tree())
    run_tree(create_ac_control_tree())
    run_tree(create_dynamic_color_tree())
    run_tree(create_stack_blocks_tree())
    run_tree(create_unstack_blocks_tree())
    run_tree(create_room_setup_tree())
    run_tree(create_temperature_based_control())


if __name__ == "__main__":
    main()
