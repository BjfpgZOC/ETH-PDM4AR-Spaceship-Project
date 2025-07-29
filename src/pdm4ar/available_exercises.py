from pdm4ar.exercises_def import *
from pdm4ar.exercises_def.structures import Exercise
from typing import Mapping, Callable
from frozendict import frozendict

available_exercises: Mapping[str, Callable[[], Exercise]] = frozendict(
    {
        "11": get_exercise11,
    }
)
