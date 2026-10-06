"""Offline checks for basic ModelScope download and local-loading readiness."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from customer_policy_verifier.training import backbone_download as backbone
from customer_policy_verifier.training import train_jev as training


class BackboneDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "model"
        self.payloads = {
            "config.json": b'{"model_type":"qwen3"}',
            "tokenizer_config.json": b'{}', "tokenizer.json": b'{}',
            "generation_config.json": b'{}', "model.safetensors": b'fixture weights',
        }
        self.cfg = {"model_repo": "offline/tiny", "model_revision": "master", "download_source": "modelscope"}
        self.args = SimpleNamespace(backbone=self.directory, tokenizer_only=True)
        self.snapshot = Mock(side_effect=self.transfer)
        modules = {name: ModuleType(name) for name in
                   ("modelscope", "modelscope.hub", "modelscope.hub.snapshot_download", "huggingface_hub")}
        modules["modelscope.hub.snapshot_download"].snapshot_download = self.snapshot
        modules["huggingface_hub"].snapshot_download = Mock(side_effect=AssertionError("HF must not be contacted"))
        replacement = patch.dict(sys.modules, modules)
        replacement.start()
        self.addCleanup(replacement.stop)

    def transfer(self, *, model_id, revision, local_dir, allow_patterns):
        import fnmatch
        self.assertEqual((model_id, revision), ("offline/tiny", "master"))
        for name, data in self.payloads.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                (Path(local_dir) / name).write_bytes(data)
        (Path(local_dir) / ".msc").write_bytes(b"sdk bookkeeping")
        return local_dir

    def source(self):
        return self.directory / "backbone-source.json"

    def test_default_download_uses_modelscope_current_branch_without_hf(self):
        self.assertEqual(training.parser().parse_args(["download"]).backbone,
                         PROJECT / "models/Qwen3-0.6B-modelscope")
        result = backbone.download(self.args, self.cfg)
        self.assertEqual(result["source"]["provider"], "modelscope")
        self.assertEqual(result["source"]["revision"], "master")
        self.assertFalse(result["source"]["has_weights"])
        self.assertNotIn("model.safetensors", result["source"]["files"])
        self.assertIsInstance(result["source"]["files"], list)
        sys.modules["huggingface_hub"].snapshot_download.assert_not_called()

    def test_tokenizer_then_weights_and_local_validation(self):
        backbone.download(self.args, self.cfg)
        with self.assertRaisesRegex(ValueError, "only tokenizer downloaded"):
            backbone.validate_backbone(self.directory, self.cfg, require_weights=True)
        self.args.tokenizer_only = False
        backbone.download(self.args, self.cfg)
        self.snapshot.reset_mock()
        source = backbone.validate_backbone(self.directory, self.cfg, require_weights=True)
        self.assertTrue(source["has_weights"])
        self.snapshot.assert_not_called()

    def test_updated_model_files_are_accepted_without_hash_pins(self):
        backbone.download(self.args, self.cfg)
        self.payloads["tokenizer_config.json"] = b'{"updated": true}'
        source = backbone.download(self.args, self.cfg)["source"]
        self.assertNotIn("content_manifest_sha256", source)
        self.assertEqual((self.directory / "tokenizer_config.json").read_bytes(), self.payloads["tokenizer_config.json"])
        backbone.validate_backbone(self.directory, self.cfg)

    def test_old_revision_and_file_hash_notes_do_not_block_local_loading(self):
        backbone.download(self.args, self.cfg)
        self.source().write_text(json.dumps({"repo_id": "offline/tiny", "revision": "old-fixed-commit",
                                           "has_weights": True, "files": {"config.json": "obsolete hash"}}))
        result = backbone.validate_backbone(self.directory, self.cfg)
        self.assertFalse(result["has_weights"], "check actual files instead of trusting old notes")

    def test_incomplete_download_preserves_previous_source_note(self):
        backbone.download(self.args, self.cfg)
        original = self.source().read_bytes()
        self.args.tokenizer_only = False
        del self.payloads["model.safetensors"]
        with self.assertRaisesRegex(ValueError, "weights missing"):
            backbone.download(self.args, self.cfg)
        self.assertEqual(self.source().read_bytes(), original)

    def test_missing_tokenizer_file_is_rejected(self):
        del self.payloads["tokenizer.json"]
        with self.assertRaisesRegex(ValueError, "missing or empty backbone file: tokenizer.json"):
            backbone.download(self.args, self.cfg)
        self.assertFalse(self.source().exists())

    def test_empty_weights_are_rejected(self):
        self.args.tokenizer_only = False
        self.payloads["model.safetensors"] = b''
        with self.assertRaisesRegex(ValueError, "weights missing"):
            backbone.download(self.args, self.cfg)

    def test_sharded_weights_require_all_referenced_files(self):
        backbone.download(self.args, self.cfg)
        (self.directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
        (self.directory / "model-00001.safetensors").write_bytes(b"shard1")
        with self.assertRaisesRegex(ValueError, "incomplete safetensors weights"):
            backbone.validate_backbone(self.directory, self.cfg, require_weights=True)
        (self.directory / "model-00002.safetensors").write_bytes(b"shard2")
        self.assertTrue(backbone.validate_backbone(self.directory, self.cfg, require_weights=True)["has_weights"])

    def test_wrong_model_type_is_rejected(self):
        self.payloads["config.json"] = b'{"model_type":"other"}'
        with self.assertRaisesRegex(ValueError, "expected a Qwen3"):
            backbone.download(self.args, self.cfg)

    def test_plain_sdk_directory_does_not_need_our_source_manifest(self):
        backbone.download(self.args, self.cfg)
        self.source().unlink()
        (self.directory / "README.md").write_text("optional model card")
        source = backbone.validate_backbone(self.directory, self.cfg)
        self.assertEqual(source["provider"], "local")
        self.assertIsNone(backbone.source_digest(self.directory))


if __name__ == "__main__":
    unittest.main()
