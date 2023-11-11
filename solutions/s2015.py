from collections import defaultdict
from hashlib import md5
from typing import Callable

from .runner import r, Part, T


def instr_to_val(instr: str) -> int:
    return 1 if instr == "(" else -1


@r.solution(2015, 1, Part.A)
def s_2015_1_a(solution_input: str) -> int:
    return sum([instr_to_val(instr) for instr in solution_input])


@r.solution(2015, 1, Part.B)
def s_2015_1_b(solution_input: str) -> int:
    floor = 0
    for index, instr in enumerate(solution_input):
        floor += instr_to_val(instr)
        if floor == -1:
            return index + 1
    return -1


def gift_dims(input: str) -> tuple[int, int, int]:
    l, w, h = [int(dim) for dim in input.split("x")]
    return (l, w, h)


def wrapping_paper(dims: tuple[int, int, int]) -> int:
    l, w, h = dims
    side1 = l * w
    side2 = w * h
    side3 = h * l
    return 2 * (side1 + side2 + side3) + min(side1, side2, side3)


def ribbon(dims: tuple[int, int, int]) -> int:
    dim1, dim2, dim3 = sorted(dims)
    return (2 * (dim1 + dim2)) + (dim1 * dim2 * dim3)


@r.solution(2015, 2, Part.A)
def s_2015_2_a(solution_input: str) -> int:
    return sum([wrapping_paper(gift_dims(i)) for i in solution_input.split("\n")])


@r.solution(2015, 2, Part.B)
def s_2015_2_b(solution_input: str) -> int:
    return sum([ribbon(gift_dims(i)) for i in solution_input.split("\n")])


def move_santa(pos: tuple[int, int], move: str) -> tuple[int, int]:
    if move == "^":
        return (pos[0], pos[1] + 1)
    elif move == "<":
        return (pos[0] - 1, pos[1])
    elif move == ">":
        return (pos[0] + 1, pos[1])
    else:
        return (pos[0], pos[1] - 1)


@r.solution(2015, 3, Part.A)
def s_2015_3_a(solution_input: str) -> int:
    positions = [(0, 0)]
    for move in solution_input:
        positions.append(move_santa(positions[-1], move))
    return len(set(positions))


@r.solution(2015, 3, Part.B)
def s_2015_3_b(solution_input: str) -> int:
    santa_positions = [(0, 0)]
    robo_positions = [(0, 0)]
    for index, move in enumerate(solution_input):
        if index % 2 == 0:
            santa_positions.append(move_santa(santa_positions[-1], move))
        else:
            robo_positions.append(move_santa(robo_positions[-1], move))
    return len(set(santa_positions + robo_positions))


def digest_starts_with(start: str, string: str) -> int:
    for num in range(0, 9_999_999):
        digest = md5(f"{string}{num}".encode()).hexdigest()
        if digest.startswith(start):
            return num
    return -1


@r.solution(2015, 4, Part.A)
def s_2015_4_a(solution_input: str) -> int:
    return digest_starts_with("00000", solution_input)


@r.solution(2015, 4, Part.B)
def s_2015_4_b(solution_input: str) -> int:
    return digest_starts_with("000000", solution_input)


def nice_string_a(string: str) -> bool:
    vowel_count = 0
    twins = False
    for index, letter in enumerate(string):
        if index > 0:
            last = string[index - 1]
            if (last + letter) in ["ab", "cd", "pq", "xy"]:
                return False
            if last == letter and not twins:
                twins = True
        if letter in ["a", "e", "i", "o", "u"]:
            vowel_count += 1
    return vowel_count >= 3 and twins


def nice_string_b(string: str) -> bool:
    pairs: defaultdict[str, set[int]] = defaultdict(set)
    pair = False
    split = False
    for index, letter in enumerate(string):
        if pair and split:
            return True
        if index > 1:
            last_two = string[index - 2 : index]
            if last_two[0] == letter and not split:
                split = True
        if index < len(string) - 1:
            key = letter + string[index + 1]
            for i in pairs[key]:
                if i + 2 <= index and not pair:
                    pair = True
            else:
                pairs[key].add(index)

    return pair and split


@r.solution(2015, 5, Part.A)
def s_2015_5_a(solution_input: str) -> int:
    return sum([1 for string in solution_input.split("\n") if nice_string_a(string)])


@r.solution(2015, 5, Part.B)
def s_2015_5_b(solution_input: str) -> int:
    return sum([1 for string in solution_input.split("\n") if nice_string_b(string)])


def light_grid(size: int, default: T) -> list[list[T]]:
    grid = []
    for _ in range(size):
        grid += [[default] * size]
    return grid


def parse_light_cmd(cmd: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
    parts = cmd.split(" ")
    tr = [int(pos) for pos in parts[-1].split(",")]
    bl = [int(pos) for pos in parts[-3].split(",")]
    action = " ".join(parts[0:2]) if len(parts) == 5 else parts[0]
    return (action, (bl[0], bl[1]), (tr[0], tr[1]))


def modify_grid(
    grid: list[list[T]],
    action: str,
    bl: tuple[int, int],
    tr: tuple[int, int],
    mod_fun: Callable[[list[list[T]], str, int, int], None],
):
    for x in range(bl[0], tr[0] + 1):
        for y in range(bl[1], tr[1] + 1):
            mod_fun(grid, action, x, y)


def toggle_mod(grid: list[list[bool]], action: str, x: int, y: int) -> None:
    if action == "turn on":
        grid[y][x] = True
    elif action == "turn off":
        grid[y][x] = False
    else:
        grid[y][x] = not grid[y][x]


def var_mod(grid: list[list[int]], action: str, x: int, y: int) -> None:
    if action == "turn on":
        grid[y][x] += 1
    elif action == "turn off":
        grid[y][x] = max(0, grid[y][x] - 1)
    else:
        grid[y][x] += 2


@r.solution(2015, 6, Part.A)
def s_2015_6_a(solution_input: str) -> int:
    grid = light_grid(1000, False)
    for line in solution_input.split("\n"):
        action, bl, tr = parse_light_cmd(line)
        modify_grid(grid, action, bl, tr, toggle_mod)
    return sum([sum([1 for cell in row if cell]) for row in grid])


@r.solution(2015, 6, Part.B)
def s_2015_6_b(solution_input: str) -> int:
    grid = light_grid(1000, 0)
    for line in solution_input.split("\n"):
        action, bl, tr = parse_light_cmd(line)
        modify_grid(grid, action, bl, tr, var_mod)
    return sum([sum(row) for row in grid])
