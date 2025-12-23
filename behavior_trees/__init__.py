"""
Behavior Tree Templates for HMAS Affordances

This module provides py-trees based behavior tree node templates for working
with Action Affordances and Property Affordances from HMAS (Hypermedia Multi-Agent Systems)
Thing Descriptions in Turtle (TTL) format.

Main components:
- ActionAffordanceNode: Execute action affordances via HTTP POST
- PropertyAffordanceNode: Read property affordances via HTTP GET  
- PropertyConditionNode: Check property values against expected conditions
- ComparisonPropertyConditionNode: Compare property values using operators

Example usage:
    from behavior_trees import ActionAffordanceNode, PropertyConditionNode
    
    # Create an action node to turn on a light
    turn_on = ActionAffordanceNode(
        name="TurnOnLight",
        action_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/turn_on"
    )
    
    # Create a condition node to check if light is on
    is_on = PropertyConditionNode(
        name="CheckLightOn",
        property_url="http://localhost:8080/workspaces/home0/balcony/artifacts/balconyLight/properties/state",
        expected_value="on"
    )
"""

from .affordance_nodes import (
    ActionAffordanceNode,
    PropertyAffordanceNode,
    PropertyConditionNode,
    ComparisonPropertyConditionNode,
    ComparisonOperator,
)

from .blackboard_keys import BlackboardKeys

__version__ = "0.1.0"

__all__ = [
    # Core node types
    "ActionAffordanceNode",
    "PropertyAffordanceNode", 
    "PropertyConditionNode",
    "ComparisonPropertyConditionNode",
    "ComparisonOperator",
    # Blackboard
    "BlackboardKeys",
]
