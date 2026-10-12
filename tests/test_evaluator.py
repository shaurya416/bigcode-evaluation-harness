import json
from types import SimpleNamespace

import pytest

from bigcode_eval import evaluator


class FakeTask:
    requires_execution = False

    def __init__(self, n_problems):
        self.dataset = [{"test": f"assert {i}"} for i in range(n_problems)]
        self.scored = None

    def get_dataset(self):
        return self.dataset

    def get_reference(self, doc):
        return doc["test"]

    def process_results(self, generations, references):
        self.scored = (generations, references)
        return {"pass@1": 1.0}


def _evaluator_for(task, monkeypatch, tmp_path, generations):
    monkeypatch.setattr(evaluator.tasks, "get_task", lambda name, args=None: task)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    generations_path = tmp_path / "generations.json"
    generations_path.write_text(json.dumps(generations))
    args = SimpleNamespace(
        limit=None,
        limit_start=0,
        n_samples=1,
        check_references=False,
        load_generations_path=str(generations_path),
        save_generations_path=str(tmp_path / "out.json"),
        save_every_k_tasks=-1,
        metric_output_path=str(tmp_path / "metrics.json"),
        allow_code_execution=False,
    )
    accelerator = SimpleNamespace(is_main_process=True)
    return evaluator.Evaluator(accelerator, None, None, args)


def test_evaluate_refuses_generations_file_covering_fewer_problems(monkeypatch, tmp_path):
    task = FakeTask(n_problems=3)
    # generations saved by an earlier run with --limit 1
    ev = _evaluator_for(task, monkeypatch, tmp_path, [["x = 0"]])

    with pytest.raises(ValueError, match="1 problems"):
        ev.evaluate("fake")
    assert task.scored is None


def test_evaluate_refuses_problems_without_generations(monkeypatch, tmp_path):
    task = FakeTask(n_problems=3)
    # an intermediate file: the last two problems were never generated
    ev = _evaluator_for(task, monkeypatch, tmp_path, [["x = 0"], [], []])

    with pytest.raises(ValueError, match="no generations"):
        ev.evaluate("fake")
    assert task.scored is None


def test_evaluate_scores_complete_generations_file(monkeypatch, tmp_path):
    task = FakeTask(n_problems=2)
    ev = _evaluator_for(task, monkeypatch, tmp_path, [["x = 0"], ["x = 1"]])

    assert ev.evaluate("fake") == {"pass@1": 1.0}
    assert task.scored == ([["x = 0"], ["x = 1"]], ["assert 0", "assert 1"])
