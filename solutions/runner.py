from enum import Enum
from typing import Callable

from aocd import get_data


class Part(Enum):
    A = "A"
    B = "B"


class Runner:
    __solutions: dict[tuple[int, int, Part], Callable[[str], str]]

    def __init__(self) -> None:
        self.__solutions = {}

    def __fetch_data(self, year: int, day: int) -> str:
        return get_data(year=year, day=day)

    def solution(
        self, year: int, day: int, part: Part
    ) -> Callable[[Callable[[str], str]], None]:
        def register_solution(solution: Callable[[str], str]) -> None:
            self.__solutions[(year, day, part)] = solution

        return register_solution

    def run(self, year: int, day: int, part: Part) -> None:
        self.run_custom_input(year, day, part, self.__fetch_data(year, day))

    def run_custom_input(self, year: int, day: int, part: Part, input: str) -> None:
        solution = self.__solutions.get((year, day, part))
        if solution:
            answer = solution(input)
            print(f"Year {year} Day {day} Part {part.value}: {answer}")
        else:
            print(f"No solution for Year {year} Day {day} Part {part.value}")

    def run_all(self) -> None:
        for year, day, part in self.__solutions:
            self.run(year, day, part)


r = Runner()
