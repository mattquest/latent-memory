"""CPU artifact-integrity fixtures; scripted outputs are never model results."""
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from eval import controller_screen as screen
from eval.adaptive_experiments import Corpus
from scripts import build_large_controller_report as report


REPOSITORY = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def test_local_tokenizer_preserves_eos_set_without_special_setter_and_returns_token_list(tmp_path, monkeypatch):
    calls = []

    class ModernTokenizer:
        eos_token_id = 511

        @property
        def eos_token_ids(self):
            return {511}

        @eos_token_ids.setter
        def eos_token_ids(self, value):
            raise ValueError("Transformers special-token setters reject a set")

        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            assert messages == [{"role": "user", "content": "Where?"}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
            return {"input_ids": [11, 12]} if return_dict else [11, 12]

    underlying = ModernTokenizer()

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return underlying

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    write_json(tmp_path / "generation_config.json", {"eos_token_id": [511, 512]})
    tokenizer = report.tokenizer_for(tmp_path)
    assert tokenizer.eos_token_ids == {511, 512} and underlying.eos_token_ids == {511}
    assert tokenizer.eos_token_id == 511  # Ordinary tokenizer attributes still delegate.
    assert calls == [(str(tmp_path), {"local_files_only": True, "trust_remote_code": False})]
    messages = [{"role": "user", "content": "Where?"}]
    options = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    assert tokenizer.apply_chat_template(messages, **options) == [11, 12]
    assert tokenizer.apply_chat_template(messages, return_dict=True, **options) == [11, 12]


class CharacterTokenizer:
    """A transparent native template with an out-of-band EOS token."""

    eos_token_ids = {511}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, tokens, *, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens
                       if not skip_special_tokens or token not in self.eos_token_ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking):
        assert tokenize is True and add_generation_prompt is True
        assert enable_thinking is False
        text = "".join(f"<{message['role']}>\n{message['content']}\n" for message in messages)
        return self.encode(text + "<assistant>\n")


class ScriptedBackend:
    """Mirrors the result schema while running no inference or accelerator code."""

    def __init__(self, large=False):
        self.tokenizer = CharacterTokenizer()
        self.large = large

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def chat_tokens(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)

    def clear_cache(self):
        pass

    def synchronize(self):
        pass

    def memory_stats(self):
        return {"active_bytes": 64, "peak_bytes": 128,
                "allocator_cache_bytes": 16, "memory_limit_bytes": 1024 ** 3}

    def generate(self, messages, max_tokens, temperature=0.0, top_p=1.0,
                 top_k=0, seed=None, presence_penalty=0.0):
        body = messages[-1]["content"]
        alpha = "Question: Alpha" in body
        if "Task: Decide" in body:
            raw = "SEARCH: bridgealpha destination" if alpha else "SEARCH: bridgebeta destination"
        elif "destination is FOUND" in body and (alpha or self.large):
            raw = "FOUND"
        else:
            raw = "UNKNOWN"
        output = self.encode(raw)
        assert len(output) < max_tokens
        prompt = self.chat_tokens(messages)
        sampled = output + [511]
        return {"raw_output": raw, "output_token_ids": output,
                "prompt_token_ids": prompt,
                "stats": {"stop_reason": "eos", "prefill_tokens": len(prompt),
                    "generated_tokens": len(output), "sampled_tokens_including_eos": len(sampled),
                    "sampled_token_ids": sampled, "context_reservation_tokens": len(prompt) + max_tokens,
                    "max_output_tokens": max_tokens, "native_cache_types": ["KVCache"],
                    "template_ms": 0.0, "prefill_ms": 0.0, "generation_ms": 0.0,
                    "model_inference_ms": 0.0, "elapsed_ms": 0.0,
                    "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                        "seed": seed, "presence_penalty": presence_penalty,
                        "presence_scope": "unique generated tokens in this call; prompt excluded"},
                    "thinking": False, "memory": self.memory_stats()}}


def fixture(tmp_path, *, qwen35=False):
    root = tmp_path / "repo"
    root.mkdir()
    questions = [{"id": f"q{index}", "question": f"{name} origin?",
                  "answers": ["FOUND"], "category": "2hop", "split": "development",
                  "supporting_context_ids": [f"d{index * 2 + 1}", f"d{index * 2 + 2}"]}
                 for index, name in enumerate(("Alpha", "Beta"))]
    documents = [
        {"id": "d1", "title": "Alpha Alpha", "text": "Alpha connects to bridgealpha."},
        {"id": "d2", "title": "Bridgealpha destination", "text": "Bridgealpha destination is FOUND."},
        {"id": "d3", "title": "Beta Beta", "text": "Beta connects to bridgebeta."},
        {"id": "d4", "title": "Bridgebeta destination", "text": "Bridgebeta destination is FOUND."},
    ]
    dataset, corpus_path = root / "questions.jsonl", root / "corpus.jsonl"
    write_rows(dataset, questions)
    write_rows(corpus_path, documents)
    provenance = root / "provenance.json"
    write_json(provenance, {"outputs": [{"path": path.name, "sha256": digest(path.read_bytes())}
                                      for path in (dataset, corpus_path)]})
    source_names = ("eval/controller_screen.py", "engine/text_backend_mlx.py")
    for name in source_names:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPOSITORY / name).read_bytes())
    source_hashes = {name: digest((root / name).read_bytes()) for name in source_names}
    config = screen.ScreenConfig(initial_top_k=1, documents_per_round=1,
                                 max_rounds=2, max_documents=2)
    corpus = Corpus(corpus_path)
    arms, jobs = ["basic_rag", "iterative_text"], []
    for label in ("small", "large"):
        model_path = root / "models" / label
        model_path.mkdir(parents=True)
        model_config = {"model_type": "qwen3", "vocab_size": 512,
                        "max_position_embeddings": 8192, "quantization": {"bits": 4}}
        if qwen35 and label == "large":
            model_config = {"model_type": "qwen3_5_moe", "quantization": {"bits": 4},
                "text_config": {"vocab_size": 512, "max_position_embeddings": 8192,
                    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True}}}
        for name, content in (("config.json", model_config),
                              ("tokenizer.json", {"fixture": "character tokenizer"}),
                              ("tokenizer_config.json", {"chat_template": "fixture-native"}),
                              ("generation_config.json", {"eos_token_id": 511})):
            write_json(model_path / name, content)
        (model_path / "model.safetensors").write_bytes(b"synthetic fixture, not real model weights")
        revision = digest(label.encode())[:40]
        receipt_path = model_path / "download-manifest.json"
        download_manifest = {"repo_id": f"fixture/{label}", "revision": revision,
            "download_complete": True, "files": [{"path": path.name,
                "size_bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
                for path in sorted(model_path.iterdir())]}
        write_json(receipt_path, download_manifest)
        receipt = {"manifest_path": str(receipt_path.relative_to(root)),
            "manifest_sha256": digest(receipt_path.read_bytes()),
            "repo_id": download_manifest["repo_id"], "revision": revision,
            "verified_files": [{"path": item["path"], "bytes": item["size_bytes"], "sha256": item["sha256"]}
                               for item in download_manifest["files"]],
            "verified_bytes": sum(item["size_bytes"] for item in download_manifest["files"])}
        job = {"id": label, "model_label": label, "model_path": str(model_path.relative_to(root)),
               "model_revision": revision, "run_dir": f"runs/{label}",
               "memory_limit_gib": 1}
        jobs.append(job)
        backend = ScriptedBackend(large=label == "large")
        rows = []
        for question in questions:
            shuffled_arms = list(arms)
            order_seed = int(digest(f"{config.seed}:{question['id']}:arm-order".encode())[:8], 16)
            random.Random(order_seed).shuffle(shuffled_arms)
            for arm in shuffled_arms:
                row = screen.run_case(backend, corpus, question, arm, config, revision)
                # Stable timings test paired statistics rather than CPU scheduler noise.
                row["cold_end_to_end_ms"] = (20 if arm == "basic_rag" else 40) * (2 if label == "large" else 1)
                row["component_ms"] = {"retrieval_ms": 0.0, "controller_ms": 0.0,
                                       "final_ms": 0.0, "other_ms": row["cold_end_to_end_ms"]}
                rows.append(row)
        loaded_config = copy.deepcopy(model_config)
        loaded_config["eos_token_id"] = 511
        if qwen35 and label == "large":
            rope = loaded_config["text_config"]["rope_parameters"]
            rope["type"] = rope.pop("rope_type")
        metadata = {"backend": "mlx-lm-native-template-text-only", "model_path": str(model_path),
            "model_revision": revision, "model_type": model_config["model_type"], "model_config": loaded_config,
            "config_sha256": digest(json.dumps(loaded_config, sort_keys=True).encode()),
            "quantization": {"bits": 4}, "parameter_bytes": 64,
            "parameter_dtypes": ["mlx.core.uint32"],
            "weight_dtype_policy": "unchanged checkpoint precision; no cast or dequantization",
            "native_cache_types": ["KVCache"],
            "chat_template": "checkpoint native; enable_thinking=False",
            "max_context": config.max_context_tokens, "prefill_batch_size": 256,
            "model_load_ms": 1.0, "startup_memory_check": {"memory_limit_bytes": 1024 ** 3},
            "checkpoint_files": [{"name": "model.safetensors", "bytes": (model_path / "model.safetensors").stat().st_size}],
            "wired_limit_modified": False,
            "versions": {"fixture": "1", **({"mlx-lm": "0.31.3"} if qwen35 else {})}}
        identity = {"protocol": screen.PROTOCOL, "config": asdict(config),
            "backend": metadata, "model_revision": revision, "model_path": job["model_path"],
            "source_sha256": source_hashes, "questions_sha256": digest(dataset.read_bytes()),
            "corpus_sha256": digest(corpus_path.read_bytes()), "arms": arms, "limit": 2}
        run_dir = root / job["run_dir"]
        write_json(run_dir / "manifest.json", {"identity": identity,
            "checkpoint_receipt": receipt, "setup_timings_ms": {"model_load": 1.0},
            "execution_order": [row["id"] for row in rows]})
        write_json(run_dir / "checkpoint-receipt.json", receipt)
        write_rows(run_dir / "results.jsonl", rows)
        write_json(run_dir / "summary.json", {"status": "complete",
            "planned_conditions": 4, "successful_conditions": 4, "historical_errors": 0})
    matrix = {"schema_version": getattr(report, "MATRIX_VERSION", "large-controller-screen-matrix-v1"),
        "protocol": screen.PROTOCOL, "cohort": "development", "config": asdict(config),
        "arms": arms, "limit": 2, "dataset": dataset.name, "corpus": corpus_path.name,
        "questions_sha256": digest(dataset.read_bytes()), "corpus_sha256": digest(corpus_path.read_bytes()),
        "data_manifest": provenance.name, "jobs": jobs}
    path = root / "matrix.json"
    write_json(path, matrix)
    for job in jobs:
        manifest_path = root / job["run_dir"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["matrix_sha256"] = digest(path.read_bytes())
        write_json(manifest_path, manifest)
    return root, path, matrix


def build(root, matrix_path, output):
    return report.build(matrix_path, output, root=root, no_plots=True,
                        tokenizer_factory=lambda model_path: CharacterTokenizer())


def production_fixture(tmp_path):
    """Add frozen sources, preflight identity and guarded-process artifacts."""
    root, path, matrix = fixture(tmp_path)
    matrix["source_files"] = ["eval/controller_screen.py", "engine/text_backend_mlx.py"]
    for job in matrix["jobs"]:
        job["preflight_dir"] = f"runs/preflight/{job['id']}"
    write_json(path, matrix)
    matrix_hash = digest(path.read_bytes())
    for job in matrix["jobs"]:
        run_dir, preflight_dir = root / job["run_dir"], root / job["preflight_dir"]
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["matrix_sha256"] = matrix_hash
        preflight = {"status": "passed", "checks": {name: True for name in (
            "greedy_exact_repeat", "grounded_synthetic_final", "valid_synthetic_decision",
            "thinking_disabled", "native_cache_present", "token_accounting", "context_rejected")},
            "calls": [], "backend": manifest["identity"]["backend"],
            "memory": ScriptedBackend().memory_stats()}
        preflight_path = preflight_dir / "preflight.json"
        write_json(preflight_path, preflight)
        manifest["preflight_sha256"] = digest(preflight_path.read_bytes())
        write_json(manifest_path, manifest)
        write_json(preflight_dir / "manifest.json", {"protocol": screen.PROTOCOL,
            "mode": "synthetic_preflight", "source_sha256": manifest["identity"]["source_sha256"],
            "checkpoint_receipt": manifest["checkpoint_receipt"], "matrix_sha256": matrix_hash,
            "elapsed_seconds": 6.0})
        write_json(preflight_dir / "checkpoint-receipt.json", manifest["checkpoint_receipt"])
        for directory, max_seconds in ((run_dir, 2700), (preflight_dir, 600)):
            for name in matrix["source_files"]:
                frozen = directory / "source_snapshot" / name
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes((root / name).read_bytes())
            samples = [{"rss_gib": rss, "available_gib": 64.0, "disk_free_gib": 100.0,
                        "elapsed_seconds": elapsed}
                       for rss, elapsed in ((0.25, 0.0), (0.5, 5.0))]
            write_rows(directory / "resources.jsonl", samples)
            write_json(directory / "guard-status.json", {"status": "complete", "returncode": 0,
                "reason": None, "elapsed_seconds": 6.0, "peak_rss_gib": 0.5,
                "command": ["synthetic-cpu-fixture"], "limits": {"max_seconds": max_seconds,
                    "max_rss_gib": job["memory_limit_gib"], "min_available_gib": 12,
                    "min_disk_gib": 40, "poll_seconds": 5, "min_battery_percent": 20,
                    "allow_battery": True, "nice_level": 0}})
            (directory / "process.log").write_text("Synthetic CPU integrity fixture; no model execution.\n")
    return root, path, matrix


def mutate_rows(root, matrix, mutation):
    path = root / matrix["jobs"][0]["run_dir"] / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    write_rows(path, rows)


def assert_withheld(summary):
    assert summary["report_status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert summary["verification_issues"]
    assert summary["metrics"] == []
    assert summary["paired_comparisons"] == []
    assert summary["trajectories"] == []
    assert summary.get("figures", []) == []


def alter_backend_config_and_rehash(manifest):
    backend = manifest["identity"]["backend"]
    backend["model_config"]["vocab_size"] += 1
    backend["config_sha256"] = digest(json.dumps(backend["model_config"], sort_keys=True).encode())


def rope_config():
    return {"model_type": "qwen3_5_moe", "eos_token_id": 17,
            "text_config": {"rope_parameters": {"rope_type": "default",
                "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10], "mrope_interleaved": True}}}


def test_expected_loaded_config_normalizes_only_known_rope_key_and_preserves_inputs():
    model = rope_config()
    generation = {"eos_token_id": [511, 512]}
    original_model, original_generation = copy.deepcopy(model), copy.deepcopy(generation)
    loaded, normalizations = report.expected_loaded_config(model, generation, {"mlx-lm": "0.31.3"})
    expected = copy.deepcopy(model)
    expected["eos_token_id"] = [511, 512]
    expected["text_config"]["rope_parameters"]["type"] = "default"
    del expected["text_config"]["rope_parameters"]["rope_type"]
    assert loaded == expected
    assert normalizations == [{"path": "text_config.rope_parameters", "old_key": "rope_type",
        "new_key": "type", "value": "default", "library": "mlx-lm", "version": "0.31.3",
        "source": "mlx_lm/models/qwen3_5.py:TextModelArgs.__post_init__"}]
    assert model == original_model and generation == original_generation
    loaded["text_config"]["rope_parameters"]["mrope_section"].append(99)
    assert model == original_model


@pytest.mark.parametrize("architecture,versions", [
    ("qwen3", {"mlx-lm": "0.31.3"}),
    ("qwen3_5", {"mlx-lm": "0.31.3"}),
    ("qwen3_5_moe", {"mlx-lm": "0.31.2"}),
    ("qwen3_5_moe", {"mlx-lm": "0.31.4"}),
    ("qwen3_5_moe", {}),
])
def test_rope_normalization_requires_exact_architecture_and_library_version(architecture, versions):
    model = rope_config()
    model["model_type"] = architecture
    loaded, normalizations = report.expected_loaded_config(model, {}, versions)
    assert loaded == model and normalizations == []
    assert loaded is not model


def test_rope_normalization_does_not_overwrite_existing_type():
    model = rope_config()
    model["text_config"]["rope_parameters"]["type"] = "already-set"
    loaded, normalizations = report.expected_loaded_config(model, {}, {"mlx-lm": "0.31.3"})
    assert loaded == model and normalizations == []
    assert loaded["text_config"]["rope_parameters"]["rope_type"] == "default"


def test_qwen35_loader_normalization_allows_complete_export(tmp_path):
    root, path, matrix = fixture(tmp_path, qwen35=True)
    summary = build(root, path, tmp_path / "report")
    assert summary["report_status"] == "complete", summary["verification_issues"]
    setups = {row["job_id"]: row for row in summary["model_setup"]}
    assert setups["small"]["backend_config_normalizations"] == []
    assert setups["large"]["backend_config_normalizations"][0]["new_key"] == "type"


@pytest.mark.parametrize("field,value", [
    ("rope_theta", 123.0),
    ("partial_rotary_factor", 0.5),
    ("mrope_section", [10, 11, 11]),
])
def test_qwen35_normalization_does_not_hide_unrelated_rope_corruption(tmp_path, field, value):
    root, path, matrix = fixture(tmp_path, qwen35=True)
    manifest_path = root / matrix["jobs"][1]["run_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    backend = manifest["identity"]["backend"]
    backend["model_config"]["text_config"]["rope_parameters"][field] = value
    backend["config_sha256"] = digest(json.dumps(backend["model_config"], sort_keys=True).encode())
    write_json(manifest_path, manifest)
    assert_withheld(build(root, path, tmp_path / "report"))


def test_complete_replayed_screen_has_paired_metrics_and_exact_archives(tmp_path):
    root, path, matrix = fixture(tmp_path)
    output = tmp_path / "report"
    summary = build(root, path, output)
    assert summary["report_status"] == "complete", summary["verification_issues"]
    small = next(row for row in summary["metrics"]
                 if row["job_id"] == "small" and row["arm"] == "iterative_text")
    assert small["n"] == 2 and small["exact_match"] == 0.5
    assert small["new_support_after_initial_total"] == 2
    assert small["mean_counts"]["retrieval_calls"] == 2
    assert small["cold_p50_ms"] == small["cold_p95_ms"] == 40
    large = next(row for row in summary["metrics"]
                 if row["job_id"] == "large" and row["arm"] == "iterative_text")
    assert large["exact_match"] == 1 and large["f1"] == 1
    model_pair = next(row for row in summary["paired_comparisons"]
        if row["comparison_family"] == "model" and row["reference_arm"] == "iterative_text")
    assert model_pair["reference_job"] == "small" and model_pair["comparison_job"] == "large"
    assert model_pair["n_paired"] == 2
    assert model_pair["exact_match_difference"] == 0.5
    assert model_pair["median_paired_latency_ratio"] == 2
    assert model_pair["recovered_example_ids"] == ["q1"]
    assert model_pair["regressed_example_ids"] == []
    retrieval_pair = next(row for row in summary["paired_comparisons"]
        if row["comparison_family"] == "retrieval_arm" and row["comparison_job"] == "large")
    assert retrieval_pair["recovered_example_ids"] == ["q0", "q1"]
    assert retrieval_pair["median_paired_latency_ratio"] == 2
    assert summary["trajectories"]
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    assert manifest["artifacts"]
    assert not any(receipt["source"].endswith(".safetensors") for receipt in manifest["artifacts"])
    for receipt in manifest["artifacts"]:
        packed = (output / receipt["artifact"]).read_bytes()
        assert digest(packed) == receipt["artifact_sha256"]
        raw = gzip.decompress(packed)
        assert digest(raw) == receipt["source_sha256"]
        assert len(raw) == receipt["source_bytes"]
        assert raw == Path(receipt["source"]).read_bytes()


def test_production_artifact_layout_completes_and_archives_all_provenance(tmp_path):
    root, path, matrix = production_fixture(tmp_path)
    output = tmp_path / "report"
    summary = build(root, path, output)
    assert summary["report_status"] == "complete", summary["verification_issues"]
    assert summary["planned_conditions"] == summary["successful_verified_conditions"] == 8
    receipts = json.loads((output / "artifact-manifest.json").read_text())["artifacts"]
    archived_sources = {Path(receipt["source"]) for receipt in receipts}
    for job in matrix["jobs"]:
        for directory_key in ("run_dir", "preflight_dir"):
            directory = root / job[directory_key]
            expected = [directory / name for name in ("manifest.json", "checkpoint-receipt.json",
                "resources.jsonl", "guard-status.json", "process.log")]
            expected += [directory / "source_snapshot" / name for name in matrix["source_files"]]
            assert set(expected) <= archived_sources
        assert root / job["preflight_dir"] / "preflight.json" in archived_sources
    for receipt in receipts:
        packed = (output / receipt["artifact"]).read_bytes()
        raw = gzip.decompress(packed)
        assert digest(packed) == receipt["artifact_sha256"]
        assert digest(raw) == receipt["source_sha256"]
        assert raw == Path(receipt["source"]).read_bytes()


@pytest.mark.parametrize("corruption", ["guard-failure", "changed-snapshot", "preflight-source"])
def test_production_provenance_corruptions_withhold_results(tmp_path, corruption):
    root, path, matrix = production_fixture(tmp_path)
    job = matrix["jobs"][0]
    if corruption == "guard-failure":
        target = root / job["run_dir"] / "guard-status.json"
        data = json.loads(target.read_text())
        data.update(status="stopped", returncode=-15, reason="rss_limit")
        write_json(target, data)
    elif corruption == "changed-snapshot":
        target = root / job["run_dir"] / "source_snapshot" / "eval/controller_screen.py"
        target.write_bytes(target.read_bytes() + b"\n# changed archived source\n")
    else:
        target = root / job["preflight_dir"] / "manifest.json"
        data = json.loads(target.read_text())
        data["source_sha256"]["eval/controller_screen.py"] = "0" * 64
        write_json(target, data)
    assert_withheld(build(root, path, tmp_path / "report"))


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda rows: rows.append(rows[0]), id="duplicate-row"),
    pytest.param(lambda rows: rows.pop(), id="missing-row"),
    pytest.param(lambda rows: rows.reverse(), id="execution-order"),
    pytest.param(lambda rows: rows[0].update(id="fabricated"), id="case-id"),
    pytest.param(lambda rows: rows[0].update(example_id="unselected"), id="example-id"),
    pytest.param(lambda rows: rows[0].update(status="error"), id="failed-condition"),
    pytest.param(lambda rows: rows[0]["scores"].update(exact_match=1), id="score"),
    pytest.param(lambda rows: rows[0]["scores"].update(exact_match=False), id="boolean-score"),
    pytest.param(lambda rows: rows[0].update(answers=["UNKNOWN"]), id="answer-label"),
    pytest.param(lambda rows: rows[0].update(raw_output="FOUND"), id="raw-output"),
    pytest.param(lambda rows: rows[0].update(output_token_ids=[65]), id="output-tokens"),
    pytest.param(lambda rows: rows[0].update(support_annotation_coverage=1), id="support-coverage"),
    pytest.param(lambda rows: rows[1].update(support_annotation_coverage=True), id="boolean-support-coverage"),
    pytest.param(lambda rows: rows[0]["delivered_fragments"][0].update(text="invented evidence"), id="fragment-text"),
    pytest.param(lambda rows: rows[1]["trace"][-1].update(query="invented search"), id="retrieval-query"),
    pytest.param(lambda rows: rows[1]["trace"][-1].update(new_document_ids=["d1"]), id="new-document-ids"),
    pytest.param(lambda rows: rows[0]["trace"][0]["hit_ids_and_scores"][0].__setitem__(1, 999), id="bm25-score"),
    pytest.param(lambda rows: rows[0]["generations"][0]["messages"][0].update(content="changed task"), id="prompt-message"),
    pytest.param(lambda rows: rows[0]["generations"][0]["prompt_token_ids"].append(65), id="prompt-tokens"),
    pytest.param(lambda rows: rows[1]["generations"][0]["stats"]["sampling"].update(seed=1), id="sampling-seed"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"].update(max_output_tokens=49), id="output-budget"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"].update(context_reservation_tokens=1), id="context-reservation"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"]["sampled_token_ids"].pop(), id="sampled-eos"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"].update(thinking=True), id="thinking-mode"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"].update(native_cache_types=["OtherCache"]), id="native-cache-types"),
    pytest.param(lambda rows: rows[0]["generations"][0]["stats"]["sampling"].update(temperature=0.7), id="final-not-greedy"),
    pytest.param(lambda rows: rows[0]["generations"].clear(), id="missing-final-generation"),
    pytest.param(lambda rows: rows[0]["counts"].update(total_prefill_tokens=0), id="prefill-count"),
    pytest.param(lambda rows: rows[0]["counts"].update(max_model_context_tokens=10000), id="context-budget"),
    pytest.param(lambda rows: rows[0]["counts"].update(retrieval_calls=True), id="boolean-count"),
    pytest.param(lambda rows: rows[1].update(executed_retrieval_rounds=1), id="round-count"),
    pytest.param(lambda rows: rows[1].update(invalid_actions=1), id="invalid-action-count"),
    pytest.param(lambda rows: rows[1].update(stop_reason="controller_answer"), id="stop-reason"),
    pytest.param(lambda rows: rows[0]["skipped_evidence_budget_ids"].append("d4"), id="skipped-document"),
    pytest.param(lambda rows: rows[0]["component_ms"].update(final_ms=5), id="component-timing"),
])
def test_corrupted_rows_withhold_all_headline_results(tmp_path, mutation):
    root, path, matrix = fixture(tmp_path)
    mutate_rows(root, matrix, mutation)
    assert_withheld(build(root, path, tmp_path / "report"))


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda value: value["identity"]["config"].update(max_answer_tokens=49), id="configuration"),
    pytest.param(lambda value: value["identity"].update(model_revision="other"), id="revision"),
    pytest.param(lambda value: value["identity"]["backend"].update(model_path="/wrong/model"), id="checkpoint-path"),
    pytest.param(lambda value: value["identity"].update(questions_sha256="0" * 64), id="question-hash"),
    pytest.param(lambda value: value["identity"]["source_sha256"].clear(), id="missing-source-hashes"),
    pytest.param(lambda value: value["identity"]["source_sha256"].pop("engine/text_backend_mlx.py"), id="different-source-receipts"),
    pytest.param(lambda value: value.update(matrix_sha256="0" * 64), id="matrix-hash"),
    pytest.param(lambda value: value["execution_order"].reverse(), id="declared-execution-order"),
    pytest.param(lambda value: value["checkpoint_receipt"].update(manifest_sha256="0" * 64), id="checkpoint-manifest-hash"),
    pytest.param(lambda value: value["checkpoint_receipt"].update(verified_bytes=0), id="checkpoint-verified-byte-count"),
    pytest.param(lambda value: value["checkpoint_receipt"]["verified_files"][0].update(sha256="0" * 64), id="checkpoint-verified-file-hash"),
    pytest.param(alter_backend_config_and_rehash, id="self-consistent-but-wrong-model-config"),
    pytest.param(lambda value: value["identity"]["backend"].update(quantization={"bits": 5}), id="quantization"),
    pytest.param(lambda value: value["identity"]["backend"].update(model_type="qwen3_moe"), id="model-type"),
    pytest.param(lambda value: value["identity"]["backend"]["checkpoint_files"][0].update(bytes=0), id="loaded-checkpoint-files"),
])
def test_manifest_identity_changes_are_unverified(tmp_path, mutation):
    root, path, matrix = fixture(tmp_path)
    manifest = root / matrix["jobs"][0]["run_dir"] / "manifest.json"
    value = json.loads(manifest.read_text())
    mutation(value)
    write_json(manifest, value)
    assert_withheld(build(root, path, tmp_path / "report"))


@pytest.mark.parametrize("relative_path", [
    "eval/controller_screen.py",
    "models/small/tokenizer.json", "models/small/download-manifest.json",
])
def test_changed_source_data_or_checkpoint_bytes_are_unverified(tmp_path, relative_path):
    root, path, matrix = fixture(tmp_path)
    target = root / relative_path
    target.write_bytes(target.read_bytes() + b"\n")
    assert_withheld(build(root, path, tmp_path / "report"))


@pytest.mark.parametrize("relative_path", ["questions.jsonl", "corpus.jsonl"])
def test_changed_matrix_input_bytes_are_rejected_before_export(tmp_path, relative_path):
    root, path, matrix = fixture(tmp_path)
    target = root / relative_path
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Matrix input hash differs"):
        build(root, path, tmp_path / "report")


def test_missing_provenance_and_incomplete_runner_are_unverified(tmp_path):
    root, path, matrix = fixture(tmp_path)
    write_json(root / "provenance.json", {"outputs": []})
    assert_withheld(build(root, path, tmp_path / "no-provenance"))
    write_json(root / "provenance.json", {"outputs": [
        {"path": name, "sha256": digest((root / name).read_bytes())}
        for name in ("questions.jsonl", "corpus.jsonl")]})
    summary_path = root / matrix["jobs"][0]["run_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "failed"
    write_json(summary_path, summary)
    assert_withheld(build(root, path, tmp_path / "incomplete-runner"))


def test_output_cannot_overwrite_inputs_or_follow_artifact_symlinks(tmp_path):
    root, path, matrix = fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        build(root, path, root)
    output = tmp_path / "report"
    output.mkdir()
    (output / "artifacts").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build(root, path, output)


def test_incomplete_export_removes_previous_headline_plots(tmp_path):
    root, path, matrix = fixture(tmp_path)
    output = tmp_path / "report"
    assert build(root, path, output)["report_status"] == "complete"
    names = ("accuracy-latency.png", "retrieval-productivity.png")
    for name in names:
        (output / name).write_bytes(b"stale headline from a previous export")
    mutate_rows(root, matrix, lambda rows: rows.pop())
    assert_withheld(build(root, path, output))
    assert not any((output / name).exists() for name in names)
    assert "Headline metrics and plots are withheld" in (output / "tables.md").read_text()
    assert json.loads((output / "conditions.json").read_text()) == []


@pytest.mark.parametrize("field,value", [
    ("dataset", "../questions.jsonl"),
    ("corpus", "/tmp/unrelated-corpus.jsonl"),
])
def test_matrix_rejects_paths_outside_repository(tmp_path, field, value):
    root, path, matrix = fixture(tmp_path)
    matrix[field] = value
    write_json(path, matrix)
    with pytest.raises(ValueError):
        build(root, path, tmp_path / "report")
