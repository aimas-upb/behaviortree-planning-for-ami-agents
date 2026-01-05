#!/usr/bin/env python3
"""
HomeBench Scenario Generator

Generates new scenarios for extending the HomeBench dataset with:
1. Future Scheduling: Augments existing examples with time constraints
2. Dependency Scheduling: Creates queries that link two devices where one action triggers another

Usage:
    python3 scenario_generator.py --mode future --input INPUT_FILE --output OUTPUT_FILE
    python3 scenario_generator.py --mode dependency --home-desc HOME_DESC_FILE --output OUTPUT_FILE
    python3 scenario_generator.py --mode both --input INPUT_FILE --home-desc HOME_DESC_FILE --output-dir OUTPUT_DIR
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DeviceAction:
    """Represents a device action with its parameters."""
    room: str
    device: str
    operation: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)

    def to_call_string(self, param_values: Optional[Dict[str, Any]] = None) -> str:
        """Convert to HomeBench call format: room.device.operation(params)"""
        if param_values:
            params_str = ','.join(str(v) for v in param_values.values())
            return f"{self.room}.{self.device}.{self.operation}({params_str})"
        return f"{self.room}.{self.device}.{self.operation}()"


class FutureSchedulingGenerator:
    """
    Generates Future Scheduling scenarios by augmenting existing HomeBench entries
    with time constraints.
    
    Example transformations:
    - "Turn off the lights" -> "Turn off the lights in 10 minutes"
    - "Set temperature to 20" -> "In 30 minutes, set temperature to 20"
    """
    
    # Time-related phrases for augmentation
    TIME_PHRASES_PREFIX = [
        "In {time},",
        "After {time},",
        "{time} from now,",
        "Please schedule for {time} from now:",
        "At {time} from now,",
    ]
    
    TIME_PHRASES_SUFFIX = [
        " in {time}",
        " after {time}",
        " {time} from now",
        ". Schedule this for {time} from now",
    ]
    
    TIME_VALUES = [
        ("5 minutes", 5),
        ("10 minutes", 10),
        ("15 minutes", 15),
        ("20 minutes", 20),
        ("30 minutes", 30),
        ("45 minutes", 45),
        ("1 hour", 60),
        ("2 hours", 120),
        ("half an hour", 30),
    ]
    
    # Context phrases that give reasons for scheduling
    CONTEXT_PHRASES = [
        "I will go to sleep in {time}. Can you {action}?",
        "I'm leaving the house in {time}. Please {action}.",
        "We have guests arriving in {time}. {action}.",
        "I'm heading out in {time}, so {action}.",
        "My meeting starts in {time}. {action}.",
        "I'll be back in {time}. Until then, {action}.",
        "The kids' bedtime is in {time}. {action}.",
        "I'm going to take a nap. Wake me up by adjusting things in {time}: {action}.",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def _add_time_constraint(self, original_input: str, time_value: str) -> str:
        """Add a time constraint to the original input."""
        mode = random.choice(['prefix', 'suffix', 'context'])
        
        if mode == 'prefix':
            phrase = random.choice(self.TIME_PHRASES_PREFIX).format(time=time_value)
            # Lowercase the first letter of original if adding prefix
            modified = original_input[0].lower() + original_input[1:] if original_input else original_input
            return f"{phrase} {modified}"
        elif mode == 'suffix':
            phrase = random.choice(self.TIME_PHRASES_SUFFIX).format(time=time_value)
            # Remove trailing period if exists before adding suffix
            modified = original_input.rstrip('.')
            return f"{modified}{phrase}"
        else:  # context
            phrase = random.choice(self.CONTEXT_PHRASES)
            # Extract the action description (lowercase first letter)
            action = original_input[0].lower() + original_input[1:] if original_input else original_input
            action = action.rstrip('.')
            return phrase.format(time=time_value, action=action)
    
    def _augment_output(self, original_output: str, delay_minutes: int) -> str:
        """
        Augment the output format with scheduling information.
        
        The output format adds a schedule wrapper around actions:
        Original: '''room.device.action()'''
        Augmented: '''schedule({delay_minutes}, room.device.action())'''
        """
        # Strip outer quotes if present
        output = original_output.strip().strip("'")
        
        if not output:
            return original_output
        
        # Split multiple actions by comma
        actions = [a.strip() for a in output.split(',') if a.strip()]
        
        augmented_actions = []
        for action in actions:
            if action == 'error_input':
                augmented_actions.append('error_input')
            else:
                # Wrap action in schedule call
                augmented_actions.append(f"schedule({delay_minutes}, {action})")
        
        return f"'''{','.join(augmented_actions)},'''"
    
    def generate_from_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a future scheduling entry from an existing entry."""
        time_phrase, delay_minutes = random.choice(self.TIME_VALUES)
        
        new_input = self._add_time_constraint(entry['input'], time_phrase)
        new_output = self._augment_output(entry['output'], delay_minutes)
        
        # Create new ID with 'future_' prefix
        original_id = entry['id']
        new_id = f"future_{original_id}"
        
        return {
            'id': new_id,
            'input': new_input,
            'output': new_output,
            'home_id': entry['home_id'],
            'type': f"future_{entry.get('type', 'unknown')}",
            'original_id': original_id,
            'delay_minutes': delay_minutes
        }
    
    def generate_batch(
        self, 
        entries: List[Dict[str, Any]], 
        sample_ratio: float = 0.3,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate future scheduling entries from a batch of existing entries.
        
        Args:
            entries: List of original HomeBench entries
            sample_ratio: Fraction of entries to augment (default 0.3)
            max_samples: Maximum number of samples to generate
        
        Returns:
            List of augmented future scheduling entries
        """
        # Filter to only include 'normal' entries (those that aren't errors)
        valid_entries = [e for e in entries if e.get('type') == 'normal']
        
        # Sample entries
        sample_size = int(len(valid_entries) * sample_ratio)
        if max_samples is not None:
            sample_size = min(sample_size, max_samples)
        
        sampled = random.sample(valid_entries, min(sample_size, len(valid_entries)))
        
        return [self.generate_from_entry(e) for e in sampled]


class DependencySchedulingGenerator:
    """
    Generates Dependency Scheduling scenarios by creating queries that link
    two devices where one action triggers another.
    
    Examples:
    - "When the dishwasher finishes, turn off the kitchen lights"
    - "When the light is turned on, set the air conditioner to cool mode"
    - "After the washing machine completes, turn on the dryer"
    """
    
    # Trigger conditions for different devices
    TRIGGER_CONDITIONS = {
        'light': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
            ("after the {device} in the {room} turns on", "turn_on"),
            ("once the {device} in the {room} is off", "turn_off"),
        ],
        'air_conditioner': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
            ("when the {device} in the {room} reaches the target temperature", "set_temperature"),
            ("after the {device} in the {room} starts", "turn_on"),
        ],
        'heating': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
            ("when the {device} in the {room} finishes heating", "set_temperature"),
            ("once the {device} in the {room} warms up", "turn_on"),
        ],
        'humidifier': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
            ("after the {device} in the {room} starts running", "turn_on"),
        ],
        'dehumidifiers': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
            ("when the {device} in the {room} finishes its cycle", "turn_off"),
        ],
        'media_player': [
            ("when the {device} in the {room} starts playing", "play"),
            ("when the {device} in the {room} stops", "stop"),
            ("when the {device} in the {room} is paused", "pause"),
            ("after the {device} in the {room} begins playback", "play"),
        ],
        'curtain': [
            ("when the {device} in the {room} opens", "open"),
            ("when the {device} in the {room} closes", "close"),
            ("after the {device} in the {room} is opened", "open"),
        ],
        'blinds': [
            ("when the {device} in the {room} opens", "open"),
            ("when the {device} in the {room} closes", "close"),
        ],
        'fan': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
        ],
        'vacuum_robot': [
            ("when the {device} finishes cleaning", "set_mode"),
            ("after the {device} completes its cycle", "set_mode"),
        ],
        'water_heater': [
            ("when the {device} in the {room} heats up", "set_temperature"),
            ("when the {device} in the {room} is turned on", "turn_on"),
        ],
        'aromatherapy': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
        ],
        'air_purifiers': [
            ("when the {device} in the {room} is turned on", "turn_on"),
            ("when the {device} in the {room} is turned off", "turn_off"),
        ],
    }
    
    # Response actions for different devices
    RESPONSE_ACTIONS = {
        'light': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} brightness to {brightness} in the {room}", "set_brightness", {"brightness": [10, 30, 50, 70, 90, 100]}),
        ],
        'air_conditioner': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} to {mode} mode in the {room}", "set_mode", {"mode": ["cool", "heat", "auto", "fan_only"]}),
            ("set the {device} temperature to {temperature} in the {room}", "set_temperature", {"temperature": [18, 20, 22, 24, 26, 28]}),
            ("set the {device} fan speed to {fan_speed} in the {room}", "set_fan_speed", {"fan_speed": ["low", "medium", "high", "auto"]}),
        ],
        'heating': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} to {mode} mode in the {room}", "set_mode", {"mode": ["heat", "fan_only"]}),
            ("set the {device} temperature to {temperature} in the {room}", "set_temperature", {"temperature": [18, 20, 22, 24, 26]}),
        ],
        'humidifier': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} intensity to {intensity} in the {room}", "set_intensity", {"intensity": [20, 40, 60, 80]}),
            ("set the {device} to {mode} mode in the {room}", "set_mode", {"mode": ["normal", "sleep"]}),
        ],
        'media_player': [
            ("start the {device} in the {room}", "play", {}),
            ("pause the {device} in the {room}", "pause", {}),
            ("stop the {device} in the {room}", "stop", {}),
            ("set the {device} volume to {volume} in the {room}", "set_volume", {"volume": [20, 40, 50, 60, 80]}),
        ],
        'curtain': [
            ("open the {device} in the {room}", "open", {}),
            ("close the {device} in the {room}", "close", {}),
            ("set the {device} to {degree} percent in the {room}", "set_degree", {"degree": [25, 50, 75, 100]}),
        ],
        'fan': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} speed to {speed} in the {room}", "set_speed", {"speed": ["low", "medium", "high"]}),
        ],
        'aromatherapy': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} intensity to {intensity} in the {room}", "set_intensity", {"intensity": [20, 40, 60, 80]}),
        ],
        'dehumidifiers': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} intensity to {intensity} in the {room}", "set_intensity", {"intensity": [20, 40, 60, 80]}),
        ],
        'air_purifiers': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} fan speed to {fan_speed} in the {room}", "set_fan_speed", {"fan_speed": ["low", "medium", "high"]}),
        ],
        'water_heater': [
            ("turn on the {device} in the {room}", "turn_on", {}),
            ("turn off the {device} in the {room}", "turn_off", {}),
            ("set the {device} temperature to {temperature} in the {room}", "set_temperature", {"temperature": [40, 45, 50, 55, 60]}),
        ],
        'blinds': [
            ("open the {device} in the {room}", "open", {}),
            ("close the {device} in the {room}", "close", {}),
        ],
    }
    
    # Human-readable device names
    DEVICE_DISPLAY_NAMES = {
        'light': 'light',
        'air_conditioner': 'air conditioner',
        'heating': 'heating',
        'humidifier': 'humidifier',
        'dehumidifiers': 'dehumidifier',
        'media_player': 'media player',
        'curtain': 'curtain',
        'blinds': 'blinds',
        'fan': 'fan',
        'vacuum_robot': 'vacuum robot',
        'water_heater': 'water heater',
        'aromatherapy': 'aromatherapy device',
        'air_purifiers': 'air purifier',
        'trash': 'trash bin',
        'garage_door': 'garage door',
    }
    
    # Human-readable room names
    ROOM_DISPLAY_NAMES = {
        'master_bedroom': 'master bedroom',
        'guest_bedroom': 'guest bedroom',
        'living_room': 'living room',
        'ding_room': 'dining room',
        'dining_room': 'dining room',
        'study_room': 'study room',
        'store_room': 'store room',
        'bathroom': 'bathroom',
        'kitchen': 'kitchen',
        'balcony': 'balcony',
        'corridor': 'corridor',
        'foyer': 'foyer',
        'garage': 'garage',
    }
    
    def __init__(self, home_descriptions_path: str, seed: Optional[int] = None):
        self.home_descriptions = self._load_home_descriptions(home_descriptions_path)
        if seed is not None:
            random.seed(seed)
    
    def _load_home_descriptions(self, path: str) -> List[Dict[str, Any]]:
        """Load home descriptions from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _get_devices_by_room(self, home: Dict[str, Any]) -> Dict[str, List[DeviceAction]]:
        """Group devices by room for a given home."""
        devices_by_room: Dict[str, List[DeviceAction]] = {}
        
        seen = set()  # Track (room, device, operation) to avoid duplicates
        
        for method in home['method']:
            room = method['room_name']
            if room == 'None':
                continue
            
            device = method['device_name']
            operation = method['operation']
            key = (room, device, operation)
            
            if key not in seen:
                seen.add(key)
                if room not in devices_by_room:
                    devices_by_room[room] = []
                
                devices_by_room[room].append(DeviceAction(
                    room=room,
                    device=device,
                    operation=operation,
                    parameters=method['parameters']
                ))
        
        return devices_by_room
    
    def _get_unique_devices_in_room(
        self, 
        devices_by_room: Dict[str, List[DeviceAction]],
        room: str
    ) -> List[str]:
        """Get unique device types in a room."""
        if room not in devices_by_room:
            return []
        return list(set(d.device for d in devices_by_room[room]))
    
    def _format_room_name(self, room: str) -> str:
        """Convert room name to human-readable format."""
        return self.ROOM_DISPLAY_NAMES.get(room, room.replace('_', ' '))
    
    def _format_device_name(self, device: str) -> str:
        """Convert device name to human-readable format."""
        return self.DEVICE_DISPLAY_NAMES.get(device, device.replace('_', ' '))
    
    def _generate_trigger_phrase(
        self, 
        device: str, 
        room: str
    ) -> Tuple[str, str]:
        """Generate a trigger phrase for a device."""
        if device not in self.TRIGGER_CONDITIONS:
            return None, None
        
        phrase_template, trigger_op = random.choice(self.TRIGGER_CONDITIONS[device])
        phrase = phrase_template.format(
            device=self._format_device_name(device),
            room=self._format_room_name(room)
        )
        return phrase, trigger_op
    
    def _generate_response_phrase(
        self, 
        device: str, 
        room: str
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a response action phrase for a device."""
        if device not in self.RESPONSE_ACTIONS:
            return None, None, None
        
        phrase_template, operation, param_options = random.choice(self.RESPONSE_ACTIONS[device])
        
        # Generate random parameter values
        params = {}
        format_params = {'device': self._format_device_name(device), 'room': self._format_room_name(room)}
        
        for param_name, possible_values in param_options.items():
            value = random.choice(possible_values)
            params[param_name] = value
            format_params[param_name] = value
        
        phrase = phrase_template.format(**format_params)
        return phrase, operation, params
    
    def _create_output_string(
        self,
        trigger_room: str,
        trigger_device: str,
        trigger_op: str,
        response_room: str,
        response_device: str,
        response_op: str,
        response_params: Dict[str, Any]
    ) -> str:
        """Create the output string in HomeBench format with dependency information."""
        # Build the response action call
        if response_params:
            params_str = ','.join(str(v) for v in response_params.values())
            response_call = f"{response_room}.{response_device}.{response_op}({params_str})"
        else:
            response_call = f"{response_room}.{response_device}.{response_op}()"
        
        # Build the trigger condition
        trigger_call = f"{trigger_room}.{trigger_device}.{trigger_op}()"
        
        # Format: when_completed(trigger_action, response_action)
        return f"'''when_completed({trigger_call}, {response_call}),'''"
    
    def generate_for_home(
        self, 
        home_id: int,
        scenarios_per_room: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate dependency scheduling scenarios for a specific home."""
        if home_id >= len(self.home_descriptions):
            raise ValueError(f"Home ID {home_id} not found in descriptions")
        
        home = self.home_descriptions[home_id]
        devices_by_room = self._get_devices_by_room(home)
        
        scenarios = []
        scenario_count = 0
        
        for room, devices in devices_by_room.items():
            unique_devices = list(set(d.device for d in devices))
            
            # Need at least 2 different devices in the room for dependency
            if len(unique_devices) < 2:
                continue
            
            for _ in range(scenarios_per_room):
                # Pick two different devices
                trigger_device, response_device = random.sample(unique_devices, 2)
                
                # Generate trigger and response phrases
                trigger_phrase, trigger_op = self._generate_trigger_phrase(trigger_device, room)
                if trigger_phrase is None:
                    continue
                
                response_phrase, response_op, response_params = self._generate_response_phrase(response_device, room)
                if response_phrase is None:
                    continue
                
                # Construct the input query
                # Capitalize first letter of trigger phrase
                trigger_phrase_cap = trigger_phrase[0].upper() + trigger_phrase[1:]
                input_text = f"{trigger_phrase_cap}, {response_phrase}."
                
                # Construct the output
                output_text = self._create_output_string(
                    room, trigger_device, trigger_op,
                    room, response_device, response_op, response_params
                )
                
                scenario_count += 1
                scenarios.append({
                    'id': f"home{home_id}_dep_{scenario_count}",
                    'input': input_text,
                    'output': output_text,
                    'home_id': home_id,
                    'type': 'dependency_same_room'
                })
        
        return scenarios
    
    def generate_cross_room(
        self, 
        home_id: int,
        scenarios_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate dependency scenarios across different rooms."""
        if home_id >= len(self.home_descriptions):
            raise ValueError(f"Home ID {home_id} not found in descriptions")
        
        home = self.home_descriptions[home_id]
        devices_by_room = self._get_devices_by_room(home)
        
        rooms = list(devices_by_room.keys())
        if len(rooms) < 2:
            return []
        
        scenarios = []
        
        for i in range(scenarios_count):
            # Pick two different rooms
            trigger_room, response_room = random.sample(rooms, 2)
            
            # Pick devices from each room
            trigger_devices = self._get_unique_devices_in_room(devices_by_room, trigger_room)
            response_devices = self._get_unique_devices_in_room(devices_by_room, response_room)
            
            if not trigger_devices or not response_devices:
                continue
            
            trigger_device = random.choice([d for d in trigger_devices if d in self.TRIGGER_CONDITIONS])
            response_device = random.choice([d for d in response_devices if d in self.RESPONSE_ACTIONS])
            
            if not trigger_device or not response_device:
                continue
            
            # Generate trigger and response phrases
            trigger_phrase, trigger_op = self._generate_trigger_phrase(trigger_device, trigger_room)
            if trigger_phrase is None:
                continue
            
            response_phrase, response_op, response_params = self._generate_response_phrase(response_device, response_room)
            if response_phrase is None:
                continue
            
            # Construct the input query
            trigger_phrase_cap = trigger_phrase[0].upper() + trigger_phrase[1:]
            input_text = f"{trigger_phrase_cap}, {response_phrase}."
            
            # Construct the output
            output_text = self._create_output_string(
                trigger_room, trigger_device, trigger_op,
                response_room, response_device, response_op, response_params
            )
            
            scenarios.append({
                'id': f"home{home_id}_dep_cross_{i+1}",
                'input': input_text,
                'output': output_text,
                'home_id': home_id,
                'type': 'dependency_cross_room'
            })
        
        return scenarios
    
    def generate_all(
        self,
        scenarios_per_room: int = 2,
        cross_room_scenarios: int = 5,
        home_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Generate dependency scenarios for all homes or specified homes."""
        all_scenarios = []
        
        if home_ids is None:
            home_ids = range(len(self.home_descriptions))
        
        for home_id in home_ids:
            # Same-room dependencies
            same_room = self.generate_for_home(home_id, scenarios_per_room)
            all_scenarios.extend(same_room)
            
            # Cross-room dependencies
            cross_room = self.generate_cross_room(home_id, cross_room_scenarios)
            all_scenarios.extend(cross_room)
        
        return all_scenarios


class ScenarioGenerator:
    """Main class that combines both generators."""
    
    def __init__(
        self,
        home_descriptions_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.seed = seed
        self.future_gen = FutureSchedulingGenerator(seed=seed)
        self.dependency_gen = None
        
        if home_descriptions_path:
            self.dependency_gen = DependencySchedulingGenerator(
                home_descriptions_path, 
                seed=seed
            )
    
    def generate_future_scheduling(
        self,
        input_file: str,
        sample_ratio: float = 0.3,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate future scheduling scenarios from existing data."""
        # Load input JSONL file
        entries = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        return self.future_gen.generate_batch(entries, sample_ratio, max_samples)
    
    def generate_dependency_scheduling(
        self,
        scenarios_per_room: int = 2,
        cross_room_scenarios: int = 5,
        home_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Generate dependency scheduling scenarios."""
        if self.dependency_gen is None:
            raise ValueError("Home descriptions path required for dependency scheduling")
        
        return self.dependency_gen.generate_all(
            scenarios_per_room,
            cross_room_scenarios,
            home_ids
        )


def main():
    parser = argparse.ArgumentParser(
        description='Generate new scenarios for HomeBench dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate future scheduling scenarios:
    python3 scenario_generator.py --mode future \\
        --input datasets/HomeBench/original/train_data_part1.jsonl \\
        --output datasets/HomeBench/extended/future_scheduling.jsonl

  Generate dependency scheduling scenarios:
    python3 scenario_generator.py --mode dependency \\
        --home-desc datasets/HomeBench/original/home-description.json \\
        --output datasets/HomeBench/extended/dependency_scheduling.jsonl

  Generate both types:
    python3 scenario_generator.py --mode both \\
        --input datasets/HomeBench/original/train_data_part1.jsonl \\
        --home-desc datasets/HomeBench/original/home-description.json \\
        --output-dir datasets/HomeBench/extended/
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['future', 'dependency', 'both'],
        required=True,
        help='Type of scenarios to generate'
    )
    parser.add_argument(
        '-i', '--input',
        help='Input JSONL file for future scheduling (required for future/both modes)'
    )
    parser.add_argument(
        '--home-desc',
        help='Home descriptions JSON file (required for dependency/both modes)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (for single mode)'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory (for both mode)'
    )
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=0.3,
        help='Ratio of entries to sample for future scheduling (default: 0.3)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of future scheduling samples to generate'
    )
    parser.add_argument(
        '--scenarios-per-room',
        type=int,
        default=2,
        help='Number of dependency scenarios per room (default: 2)'
    )
    parser.add_argument(
        '--cross-room-scenarios',
        type=int,
        default=5,
        help='Number of cross-room dependency scenarios per home (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--format',
        choices=['jsonl', 'json'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['future', 'both'] and not args.input:
        parser.error("--input is required for future/both modes")
    
    if args.mode in ['dependency', 'both'] and not args.home_desc:
        parser.error("--home-desc is required for dependency/both modes")
    
    if args.mode != 'both' and not args.output:
        parser.error("--output is required for single mode generation")
    
    if args.mode == 'both' and not args.output_dir:
        parser.error("--output-dir is required for both mode")
    
    # Create generator
    generator = ScenarioGenerator(
        home_descriptions_path=args.home_desc,
        seed=args.seed
    )
    
    def save_scenarios(scenarios: List[Dict[str, Any]], output_path: str, fmt: str):
        """Save scenarios to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if fmt == 'jsonl':
            with open(output_path, 'w') as f:
                for scenario in scenarios:
                    f.write(json.dumps(scenario) + '\n')
        else:
            with open(output_path, 'w') as f:
                json.dump(scenarios, f, indent=2)
        
        print(f"Saved {len(scenarios)} scenarios to {output_path}")
    
    if args.mode == 'future':
        scenarios = generator.generate_future_scheduling(
            args.input,
            args.sample_ratio,
            args.max_samples
        )
        save_scenarios(scenarios, args.output, args.format)
    
    elif args.mode == 'dependency':
        scenarios = generator.generate_dependency_scheduling(
            args.scenarios_per_room,
            args.cross_room_scenarios
        )
        save_scenarios(scenarios, args.output, args.format)
    
    else:  # both
        output_dir = Path(args.output_dir)
        
        # Generate future scheduling
        future_scenarios = generator.generate_future_scheduling(
            args.input,
            args.sample_ratio,
            args.max_samples
        )
        future_output = output_dir / f"future_scheduling.{args.format}"
        save_scenarios(future_scenarios, str(future_output), args.format)
        
        # Generate dependency scheduling
        dependency_scenarios = generator.generate_dependency_scheduling(
            args.scenarios_per_room,
            args.cross_room_scenarios
        )
        dependency_output = output_dir / f"dependency_scheduling.{args.format}"
        save_scenarios(dependency_scenarios, str(dependency_output), args.format)
        
        print(f"\nTotal: {len(future_scenarios) + len(dependency_scenarios)} scenarios generated")


if __name__ == '__main__':
    main()
