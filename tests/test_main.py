import itertools
import os
import sys

# main.py is a script at the repository root, not part of the installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from bigcode_eval.tasks import ALL_TASKS


def test_task_names_follow_the_order_of_the_tasks_argument():
    # --load_generations_intermediate_paths gives one file per task, matched by position
    names = ["humaneval", "mbpp", "apps-introductory"]

    for order in itertools.permutations(names):
        assert main.pattern_match(list(order), ALL_TASKS) == list(order)


def test_task_names_from_a_wildcard_keep_registry_order_without_duplicates():
    task_names = main.pattern_match(["multiple-j*", "multiple-js"], ALL_TASKS)

    assert task_names == ["multiple-java", "multiple-jl", "multiple-js"]
