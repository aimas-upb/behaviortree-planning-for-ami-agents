#!/usr/bin/env python3
"""
Behavior tree to keep bedroom temperature matching living room when person is present.
Requires: pip install py_trees
"""

import py_trees
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Inverter
from py_trees.behaviour import Behaviour
from py_trees.common import Status


# ============================================================================
# Data Source Behaviors (Conditions)
# ============================================================================

class IsPersonInBedroom(Behaviour):
    """Check if person is currently in the bedroom."""
    
    def __init__(self, name="Person in Bedroom?"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="person_location",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        location = self.blackboard.get("person_location")
        if location == "bedroom":
            self.feedback_message = "Person is in bedroom"
            return Status.SUCCESS
        else:
            self.feedback_message = f"Person is in {location}"
            return Status.FAILURE


class GetBedroomTemperature(Behaviour):
    """Read current bedroom temperature."""
    
    def __init__(self, name="Get Bedroom Temp"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="bedroom_temp",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="current_bedroom_temp",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        temp = self.blackboard.get("bedroom_temp")
        self.blackboard.set("current_bedroom_temp", temp)
        self.feedback_message = f"Bedroom: {temp}°C"
        return Status.SUCCESS


class GetLivingRoomTemperature(Behaviour):
    """Read current living room temperature."""
    
    def __init__(self, name="Get Living Room Temp"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="living_room_temp",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="current_living_room_temp",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        temp = self.blackboard.get("living_room_temp")
        self.blackboard.set("current_living_room_temp", temp)
        self.feedback_message = f"Living room: {temp}°C"
        return Status.SUCCESS


class IsBedroomCoolerThanLivingRoom(Behaviour):
    """Check if bedroom is cooler than living room."""
    
    def __init__(self, name="Bedroom Cooler?", threshold=0.5):
        super().__init__(name)
        self.threshold = threshold
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="current_bedroom_temp",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="current_living_room_temp",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        bedroom_temp = self.blackboard.get("current_bedroom_temp")
        living_temp = self.blackboard.get("current_living_room_temp")
        
        diff = living_temp - bedroom_temp
        
        if diff > self.threshold:
            self.feedback_message = f"Bedroom {diff:.1f}°C cooler"
            return Status.SUCCESS
        else:
            self.feedback_message = f"Temperatures matched (diff: {diff:.1f}°C)"
            return Status.FAILURE


class IsCentralHeaterOn(Behaviour):
    """Check if central heater is operational."""
    
    def __init__(self, name="Heater On?"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="central_heater_status",
            access=py_trees.common.Access.READ
        )
    
    def update(self):
        status = self.blackboard.get("central_heater_status")
        if status == "on":
            self.feedback_message = "Central heater operational"
            return Status.SUCCESS
        else:
            self.feedback_message = f"Central heater {status}"
            return Status.FAILURE


# ============================================================================
# Action Behaviors (Controls)
# ============================================================================

class IncreaseBedroomHeat(Behaviour):
    """Increase bedroom radiator knob setting."""
    
    def __init__(self, name="Increase Bedroom Heat", increment=1):
        super().__init__(name)
        self.increment = increment
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="bedroom_radiator_knob",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="bedroom_radiator_control",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        current_setting = self.blackboard.get("bedroom_radiator_knob")
        new_setting = min(current_setting + self.increment, 5)  # Max setting of 5
        
        self.blackboard.set("bedroom_radiator_control", new_setting)
        self.feedback_message = f"Set radiator: {current_setting} → {new_setting}"
        return Status.SUCCESS


class DecreaseBedroomHeat(Behaviour):
    """Decrease bedroom radiator knob setting."""
    
    def __init__(self, name="Decrease Bedroom Heat", decrement=1):
        super().__init__(name)
        self.decrement = decrement
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="bedroom_radiator_knob",
            access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="bedroom_radiator_control",
            access=py_trees.common.Access.WRITE
        )
    
    def update(self):
        current_setting = self.blackboard.get("bedroom_radiator_knob")
        new_setting = max(current_setting - self.decrement, 0)  # Min setting of 0
        
        self.blackboard.set("bedroom_radiator_control", new_setting)
        self.feedback_message = f"Set radiator: {current_setting} → {new_setting}"
        return Status.SUCCESS


class MaintainBedroomHeat(Behaviour):
    """Keep bedroom radiator at current setting."""
    
    def __init__(self, name="Maintain Bedroom Heat"):
        super().__init__(name)
    
    def update(self):
        self.feedback_message = "Temperature stable"
        return Status.SUCCESS


# ============================================================================
# Behavior Tree Construction
# ============================================================================

def create_temperature_matching_tree():
    """
    Create the main behavior tree.
    
    Tree structure:
    - Root (Selector): Try to match temperatures or do nothing
        - Active Sequence: When person is in bedroom
            - Check person in bedroom
            - Get temperatures
            - Heating control (Selector):
                - Heat if too cold
                - Cool if too warm
                - Maintain if matched
        - Fallback: Person not in bedroom, do nothing
    """
    
    root = Selector(name="Temperature Control Root", memory=False)
    
    # Main sequence when person is in bedroom
    active_sequence = Sequence(name="Active Control", memory=True)
    
    # Check conditions
    active_sequence.add_child(IsPersonInBedroom())
    active_sequence.add_child(GetBedroomTemperature())
    active_sequence.add_child(GetLivingRoomTemperature())
    active_sequence.add_child(IsCentralHeaterOn())
    
    # Temperature adjustment selector
    adjustment_selector = Selector(name="Adjust Temperature", memory=False)
    
    # Heat up sequence: if bedroom is cooler, increase heat
    heat_up_sequence = Sequence(name="Heat Up", memory=False)
    heat_up_sequence.add_child(IsBedroomCoolerThanLivingRoom())
    heat_up_sequence.add_child(IncreaseBedroomHeat())
    
    # Cool down sequence: if bedroom is warmer, decrease heat
    cool_down_sequence = Sequence(name="Cool Down", memory=False)
    cool_down_sequence.add_child(
        Inverter(
            name="Bedroom Warmer?",
            child=IsBedroomCoolerThanLivingRoom(threshold=-0.5)
        )
    )
    cool_down_sequence.add_child(DecreaseBedroomHeat())
    
    # Maintain: temperatures are matched
    maintain_behaviour = MaintainBedroomHeat()
    
    adjustment_selector.add_children([
        heat_up_sequence,
        cool_down_sequence,
        maintain_behaviour
    ])
    
    active_sequence.add_child(adjustment_selector)
    
    # Idle behavior when person is not in bedroom
    idle = py_trees.behaviours.Success(name="Idle (Not in Bedroom)")
    
    root.add_children([active_sequence, idle])
    
    return root


# ============================================================================
# Example Usage
# ============================================================================

def setup_simulation_blackboard():
    """Set up blackboard with simulated data sources."""
    blackboard = py_trees.blackboard.Client()
    blackboard.register_key(key="person_location", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="bedroom_temp", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="living_room_temp", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="central_heater_status", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="bedroom_radiator_knob", access=py_trees.common.Access.WRITE)
    
    # Initial values
    blackboard.set("person_location", "bedroom")
    blackboard.set("bedroom_temp", 18.0)
    blackboard.set("living_room_temp", 21.0)
    blackboard.set("central_heater_status", "on")
    blackboard.set("bedroom_radiator_knob", 2)
    
    return blackboard


if __name__ == "__main__":
    # Create and setup the behavior tree
    root = create_temperature_matching_tree()
    tree = py_trees.trees.BehaviourTree(root)
    
    # Setup blackboard with simulated data
    blackboard = setup_simulation_blackboard()
    
    # Setup tree
    tree.setup(timeout=15)
    
    print("=" * 60)
    print("Temperature Matching Behavior Tree")
    print("=" * 60)
    print("\nTree Structure:")
    py_trees.display.print_ascii_tree(root, show_status=True)
    
    # Simulate a few ticks
    print("\n" + "=" * 60)
    print("Simulation")
    print("=" * 60)
    
    for i in range(5):
        print(f"\n--- Tick {i+1} ---")
        tree.tick()
        
        # Simulate temperature changes
        current_bedroom = blackboard.get("bedroom_temp")
        control = blackboard.get("bedroom_radiator_control")
        if control:
            # Simulate heating effect
            blackboard.set("bedroom_temp", current_bedroom + 0.5)
        
        print(f"Status: {root.status}")
        print(f"Bedroom: {blackboard.get('bedroom_temp'):.1f}°C, "
              f"Living Room: {blackboard.get('living_room_temp'):.1f}°C")
        print(f"Radiator: {blackboard.get('bedroom_radiator_knob')}")
