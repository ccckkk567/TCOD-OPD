"""Check concrete submitted paths, not just shell syntax or generated YAML alone."""

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class SubmissionConfigPaths(unittest.TestCase):
    def test_all_concrete_generated_config_paths_exist(self):
        for script in (ROOT / "orion/scripts").glob("submit_*.sh"):
            for relative in re.findall(r'\$checkout/(orion/generated/[^"$]+/config.yaml)',
                                       script.read_text()):
                with self.subTest(script=script.name, path=relative):
                    self.assertTrue((ROOT / relative).is_file(), relative)

    def test_monitor_run_guards_and_gate_match_actual_configs(self):
        text = (ROOT / "orion/scripts/submit_monitor_recovery_chain.sh").read_text()
        for phase in ("smoke-first", "smoke-resume", "pilot"):
            config = yaml.safe_load((ROOT / f"orion/generated/alfworld-s17-monitor-fix-{phase}/config.yaml").read_text())
            self.assertIn(config["name"], text)
            if phase == "smoke-resume":
                self.assertIn('$OUTPUT_ROOT/' + config["name"] + '"', text)
