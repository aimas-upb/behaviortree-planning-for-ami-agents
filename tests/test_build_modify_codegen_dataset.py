import unittest
from pathlib import Path

from scripts.analysis.build_modify_codegen_dataset import (
    LocalHomeGraphStore,
    build_example,
    parse_relative_clause,
    split_into_clauses,
)


class BuildModifyCodegenDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        description_dir = (
            repo_root / "data" / "homebench" / "hmas" / "home_description"
        )
        self.store = LocalHomeGraphStore(
            ttl_dir=description_dir,
            state_dir=description_dir,
        )

    def test_split_into_clauses_splits_top_level_and(self) -> None:
        text = (
            "Lower the living room air conditioner temperature by 29 degrees "
            "and turn on the heating in the study room."
        )
        clauses = split_into_clauses(text)
        self.assertEqual(
            clauses,
            [
                "Lower the living room air conditioner temperature by 29 degrees",
                "turn on the heating in the study room",
            ],
        )

    def test_parse_relative_clause_rejects_absolute_lower_to_value(
        self,
    ) -> None:
        relative = parse_relative_clause(
            "Decrease the volume of the media player in the master bedroom by 20 percent."
        )
        absolute = parse_relative_clause(
            "Lower the temperature of the air conditioner in the master bedroom to 20 degrees."
        )

        self.assertTrue(relative.is_relative)
        self.assertEqual(relative.direction, "decrease")
        self.assertEqual(relative.delta_value, 20)
        self.assertFalse(absolute.is_relative)

    def test_local_graph_metadata_extracts_schema(self) -> None:
        action_url = (
            "http://localhost:8080/workspaces/home95/master_bedroom/"
            "artifacts/masterBedroomMediaPlayer/set_volume"
        )
        property_url = (
            "http://localhost:8080/workspaces/home95/master_bedroom/"
            "artifacts/masterBedroomMediaPlayer/properties/volume"
        )

        action_meta = self.store.action_metadata(action_url)
        property_meta = self.store.property_metadata(property_url)

        self.assertEqual(action_meta["semantic_type"], "SetVolumeCommand")
        self.assertEqual(
            action_meta["schema"]["properties"]["volume"]["minimum"], 0
        )
        self.assertEqual(
            action_meta["schema"]["properties"]["volume"]["maximum"], 100
        )
        self.assertEqual(property_meta["schema"]["type"], "integer")
        self.assertEqual(property_meta["schema"]["minimum"], 0)
        self.assertEqual(property_meta["schema"]["maximum"], 100)

    def test_build_example_end_to_end_for_mixed_row(self) -> None:
        raw_entry = {
            "id": "home95_multi_436",
            "input": (
                "Set the brightness of the light on the balcony to 0, turn on "
                "the humidifier in the store room, adjust the brightness of the "
                "light in the living room to 40, turn on the light in the foyer, "
                "set the humidifier in the store room to sleep mode, set the fan "
                "speed of the air conditioner in the guest bedroom to auto, "
                "decrease the volume of the media player in the master bedroom by "
                "20 percent, set the fan speed of the heating in the living room "
                "to auto, turn off the light in the bathroom, and set the "
                "brightness of the light in the corridor to 80."
            ),
            "output": (
                "'''error_input,store_room.humidifier.turn_on(),error_input,"
                "foyer.light.turn_on(),error_input,error_input,"
                "master_bedroom.media_player.set_volume(40),"
                "living_room.heating.set_fan_speed(auto),"
                "bathroom.light.turn_off(),error_input,'''"
            ),
            "home_id": 95,
            "type": "multi10_mix",
            "official_split": "train",
            "source_file": "train_data_part1",
        }
        converted_entry = {
            "id": "home95_multi_436",
            "input": raw_entry["input"],
            "output": [
                {"execution": "error_input"},
                {
                    "execution": "success",
                    "affordance": "http://localhost:8080/workspaces/home95/store_room/artifacts/storeRoomHumidifier/turn_on",
                    "params": {},
                    "test": {
                        "property": "http://localhost:8080/workspaces/home95/store_room/artifacts/storeRoomHumidifier/properties/state",
                        "expected_value": "on",
                    },
                },
                {"execution": "error_input"},
                {
                    "execution": "success",
                    "affordance": "http://localhost:8080/workspaces/home95/foyer/artifacts/foyerLight/turn_on",
                    "params": {},
                    "test": {
                        "property": "http://localhost:8080/workspaces/home95/foyer/artifacts/foyerLight/properties/state",
                        "expected_value": "on",
                    },
                },
                {"execution": "error_input"},
                {"execution": "error_input"},
                {
                    "execution": "success",
                    "affordance": "http://localhost:8080/workspaces/home95/master_bedroom/artifacts/masterBedroomMediaPlayer/set_volume",
                    "params": {"volume": 40},
                    "test": {
                        "property": "http://localhost:8080/workspaces/home95/master_bedroom/artifacts/masterBedroomMediaPlayer/properties/volume",
                        "expected_value": 40,
                    },
                },
                {
                    "execution": "success",
                    "affordance": "http://localhost:8080/workspaces/home95/living_room/artifacts/livingRoomHeating/set_fan_speed",
                    "params": {"fan_speed": "auto"},
                    "test": {
                        "property": "http://localhost:8080/workspaces/home95/living_room/artifacts/livingRoomHeating/properties/fan_speed",
                        "expected_value": "auto",
                    },
                },
                {
                    "execution": "success",
                    "affordance": "http://localhost:8080/workspaces/home95/bathroom/artifacts/bathroomLight/turn_off",
                    "params": {},
                    "test": {
                        "property": "http://localhost:8080/workspaces/home95/bathroom/artifacts/bathroomLight/properties/state",
                        "expected_value": "off",
                    },
                },
                {"execution": "error_input"},
            ],
        }
        home_split_map = {home_id: "train" for home_id in range(100)}

        example, skip_reason = build_example(
            raw_entry=raw_entry,
            converted_entry=converted_entry,
            store=self.store,
            home_split_map=home_split_map,
        )

        self.assertIsNone(skip_reason)
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["modify_intent_count"], 1)
        self.assertEqual(
            example["intents"][0]["action"]["affordance_type"],
            "ex:SetVolumeCommand",
        )
        self.assertEqual(example["intents"][0]["action"]["parameter"], "volume")
        self.assertIn("action.verb: modify", example["runtime_modify_goal"])
        self.assertIn("set_volume", example["target_code"])
        self.assertTrue(example["validation_status"]["tree_build_ok"])


if __name__ == "__main__":
    unittest.main()
