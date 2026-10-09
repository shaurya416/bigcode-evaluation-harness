import json
import tempfile

from bigcode_eval.tasks import apps, multiple


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
