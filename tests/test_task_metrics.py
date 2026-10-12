import json
import tempfile

import pytest

from bigcode_eval import base
from bigcode_eval.tasks import apps, instruct_wizard_humaneval, multiple, quixbugs
from bigcode_eval.tasks.custom_metrics import beyond_eval


def _bare_task(task_cls):
    # skip Task.__init__, which downloads the dataset
    return task_cls.__new__(task_cls)


def test_apps_process_results_passes_level_to_metric(monkeypatch):
    calls = []

    class FakeAppsMetric:
        def compute(self, **kwargs):
            calls.append(kwargs)
            return {"strict_accuracy": 0.0}

    monkeypatch.setattr(apps, "load", lambda name: FakeAppsMetric())
    task = _bare_task(apps.create_task("introductory"))
    task.DATASET_NAME = "introductory"
    task.k_list = [1]

    results = task.process_results([["print(1)"]], [None])

    assert results == {"strict_accuracy": 0.0}
    assert calls[0]["level"] == "introductory"
    assert calls[0]["k_list"] == [1]


def test_multiple_registers_every_language_separately():
    task_names = set(multiple.create_all_tasks())

    for language in ["clj", "cpp", "ml", "pl"]:
        assert f"multiple-{language}" in task_names
    assert "multiple-cljcpp" not in task_names
    assert "multiple-mlpl" not in task_names
    assert len(task_names) == len(multiple.LANGUAGES)


def test_multiple_pass_at_k_ignores_results_left_in_temp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    # a passing result left in the system temp dir by an earlier run
    (tmp_path / "HumanEval_99_old.results.json").write_text(
        json.dumps({"results": [{"status": "OK", "exit_code": 0}]})
    )

    def failing_evaluate_problem(output_dir, problem_json_path, max_workers):
        with open(problem_json_path[: -len(".json")] + ".results.json", "w") as f:
            json.dump({"results": [{"status": "Exception", "exit_code": 1}]}, f)

    monkeypatch.setattr(multiple, "evaluate_problem", failing_evaluate_problem)
    task = _bare_task(multiple.create_task("py"))
    task.dataset = {
        "test": [{"prompt": "def f():", "name": "HumanEval_0_f", "language": "py"}]
    }

    results = task.process_results([["def f():\n    return 1"]], ["assert f() == 2"])

    assert results == {"pass@1": 0.0}


def test_quixbugs_codex_prompt_keeps_the_generated_fix(monkeypatch):
    doc = {"name": "gcd", "buggy_program": "def gcd(a, b):\n    return gcd(a % b, b)", "tests": ""}
    monkeypatch.setattr(base, "load_dataset", lambda **kwargs: {"train": [doc]})
    task = quixbugs.QuixBugs(prompt="prompt_codex")
    prompt = task.get_prompt(doc)
    fix = "\ndef gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n"
    completion = fix + "### fix the bug in the following function\n"
    # generation stops as soon as the completion contains a stop word
    end = min(completion.find(w) + len(w) for w in task.stop_words if w in completion)

    assert task.postprocess_generation(prompt + completion[:end], 0) == fix


def test_mercury_metric_with_a_difficulty_left_out_by_limit(monkeypatch):
    # canonical solutions run in 0.1s and 0.3s, the generation in 0.2s
    runtimes = iter([0.1, 0.3] + [0.2] * 5)

    def fake_run_sample(sample):
        return dict(result="passed", runtime=next(runtimes), index=sample["solution_index"], error="None")

    monkeypatch.setattr(beyond_eval.Sandbox, "run_sample", staticmethod(fake_run_sample))
    instance = {
        "solutions": [{"solution": "a"}, {"solution": "b"}],
        "convert_offline": "",
        "evaluate_offline": "",
        "entry_point": "f",
        "test_cases": "[]",
        "difficulty": "Easy",
    }

    results = beyond_eval.compute_beyond_eval([["generation"]], [instance])

    assert results["Easy_pass@1"] == 1.0
    assert results["Average_pass@1"] == 1.0
    assert results["Easy_beyond@1"] == pytest.approx(0.5)
    assert not any(key.startswith(("Medium_", "Hard_")) for key in results)


def test_wizard_humaneval_keeps_code_block_without_closing_fence():
    task = _bare_task(instruct_wizard_humaneval.HumanEvalWizardCoder)
    # generation stopped (max length or end of sequence) before the closing fence
    generation = "### Response:\nHere is the function:\n```python\ndef add(a, b):\n    return a + b\n"

    assert task.postprocess_generation(generation, 0) == "def add(a, b):\n    return a + b"
    # a closed block is still cut at its fence
    assert task.postprocess_generation(generation + "```\nDone.", 0) == "def add(a, b):\n    return a + b"
