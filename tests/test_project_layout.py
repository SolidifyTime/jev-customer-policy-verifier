"""Portable CLI dispatch and immutable historical-path resolution."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from customer_policy_verifier import cli
from customer_policy_verifier.common.paths import resolve_path


class ProjectLayoutTests(unittest.TestCase):
    def test_historical_paths_resolve_without_modifying_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "jev-customer-policy-verifier"
            config = root / "configs" / "layout-migration.json"
            config.parent.mkdir(parents=True)
            frozen = {"path_migrations": {"data/task2": "data/source/task2",
                                           "src/jev_data.py": "src/customer_policy_verifier/training/jev_data.py"}}
            config.write_text(json.dumps(frozen))
            before = config.read_bytes()
            expected = root / "data/source/task2/raw/example.json"
            for value in ("data/task2/raw/example.json", root / "data/task2/raw/example.json",
                          "/other-machine/customer-policy-verifier/data/task2/raw/example.json",
                          "/other-machine/jev-customer-policy-verifier/data/task2/raw/example.json"):
                self.assertEqual(resolve_path(value, root=root), expected)
            self.assertEqual(resolve_path("data/task2-other/example.json", root=root),
                             root / "data/task2-other/example.json")
            self.assertEqual(resolve_path("data/training/task2-v1/train.jsonl", root=root),
                             root / "data/training/task2-v1/train.jsonl")
            external = Path(directory) / "external/data.json"
            self.assertEqual(resolve_path(external, root=root), external)
            self.assertEqual(config.read_bytes(), before)

    def test_dispatch_returns_success_and_restores_process_arguments(self):
        previous = sys.argv
        calls = []
        module = SimpleNamespace(main=lambda: calls.append(list(sys.argv)) or {"integrity": "passed"})
        with patch.object(cli.importlib, "import_module", return_value=module) as imported:
            self.assertEqual(cli.main(["train", "check", "--data-dir", "fixture"]), 0)
            imported.assert_called_once_with("customer_policy_verifier.training.train_jev")
        self.assertEqual(calls, [["cpv train", "check", "--data-dir", "fixture"]])
        self.assertIs(sys.argv, previous)

    def test_dispatch_restores_arguments_after_command_failure(self):
        previous = sys.argv
        def fail():
            raise ValueError("fixture failure")
        with patch.object(cli.importlib, "import_module", return_value=SimpleNamespace(main=fail)):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                cli.main(["review", "export", "--help"])
        self.assertIs(sys.argv, previous)

    def test_help_does_not_load_training_dependencies(self):
        output = io.StringIO()
        with patch.object(cli.importlib, "import_module") as imported, contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--help"]), 0)
        imported.assert_not_called()
        self.assertIn("train", output.getvalue())


if __name__ == "__main__":
    unittest.main()
