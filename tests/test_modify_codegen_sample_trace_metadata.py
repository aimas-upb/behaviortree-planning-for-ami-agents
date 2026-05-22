import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.run_parallel_experience_homebench import (
    _reshape_ns_trace_for_viewer,
    generate_eval_report,
)
from scripts.analysis.fix_modify_codegen_reporting import fix_results_dir
from viewers.experience_trace_viewer import export_experience_html


class ModifyCodegenSampleTraceMetadataTests(unittest.TestCase):
    def test_ns_trace_preserves_modify_codegen_metadata(self) -> None:
        raw_result = {
            "success": True,
            "duration_seconds": 1.25,
            "intents": [{"text_intent": "increase the light brightness"}],
            "modify_plan_trace": {},
            "execution": {"success": True},
            "modify_intent_count": 2,
            "target_code": "tree = seq_1\n",
        }

        trace = _reshape_ns_trace_for_viewer(
            raw_result,
            test_id="home0_one_1",
            config={},
        )

        self.assertEqual(trace["modify_intent_count"], 2)
        self.assertEqual(trace["target_code"], "tree = seq_1\n")

    def test_experience_html_renders_modify_metadata(self) -> None:
        trace = {
            "config_name": "neurosymbolic_home0_one_1",
            "success": True,
            "duration_seconds": 1.25,
            "goal": "increase the light brightness",
            "config": {"model": {"name": "gpt-4o"}},
            "execution": {"success": True},
            "modify_intent_count": 2,
            "target_code": "tree = seq_1\n",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace.html"
            export_experience_html(trace, str(output_path))
            html = output_path.read_text()

        self.assertIn("Modify Intent Count", html)
        self.assertIn("Reference Target Code", html)
        self.assertIn("tree = seq_1", html)

    def test_eval_report_uses_explicit_modify_codegen_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "eval_report.html").write_text("<html></html>")

            with mock.patch(
                "scripts.experiments.run_parallel_experience_homebench.subprocess.run"
            ) as run_mock:
                generated = generate_eval_report(
                    output_dir,
                    test_data="data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference.json",
                )

        self.assertTrue(generated)
        command = run_mock.call_args.args[0]
        self.assertIn("--test-data", command)
        self.assertIn(
            "data/homebench/benchmarks/modify_codegen/home_disjoint/samples_for_inference.json",
            command,
        )

    def test_modify_codegen_metric_fix_can_skip_report_regeneration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            results_dir = base_dir / "results"
            results_dir.mkdir()
            sample_file = base_dir / "samples.json"
            sample_file.write_text("""[
  {
    "id": "home1_one_1",
    "home_id": 1,
    "input": "turn on the light",
    "output": [
      {
        "execution": "success",
        "affordance": "http://localhost:8080/workspaces/home1/kitchen/artifacts/kitchenLight/turn_on",
        "params": {},
        "test": {
          "property": "http://localhost:8080/workspaces/home1/kitchen/artifacts/kitchenLight/properties/state",
          "expected_value": "on"
        }
      }
    ]
  }
]""")
            result = {
                "test_id": "home1_one_1",
                "success": "True",
                "plan_generated": True,
                "execution_success": True,
                "expected_actions": [
                    "http://localhost:8080/workspaces/home1/kitchen/artifacts/kitchenLight/turn_on"
                ],
                "matched_actions": [
                    "http://localhost:8080/workspaces/home1/kitchen/artifacts/kitchenLight/turn_on"
                ],
                "missing_actions": [],
                "extra_actions": [],
                "properties_checked": 1,
                "properties_matched": 1,
                "property_results": [],
                "expected_impossible": 0,
                "detected_impossible": [],
                "duration": 1.0,
                "trace": {"resolution_results": []},
            }
            (results_dir / "home1_one_1.json").write_text("{}")
            (base_dir / "results_20260101_000000.json").write_text(
                json.dumps([result])
            )
            (base_dir / "metrics_20260101_000000.json").write_text(
                json.dumps({"metrics": {}, "experience_metrics": {}})
            )

            with (
                mock.patch(
                    "scripts.analysis.fix_modify_codegen_reporting._build_simulator"
                ),
                mock.patch(
                    "scripts.analysis.fix_modify_codegen_reporting._regenerate_html_report"
                ) as regenerate_mock,
            ):
                corrected_sample = fix_results_dir(
                    base_dir=base_dir,
                    sample_file=sample_file,
                    regenerate_report=False,
                )

        self.assertIsNotNone(corrected_sample)
        regenerate_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
