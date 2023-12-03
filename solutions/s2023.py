from re import compile

from .runner import r, Part


def calibration_sum(solution_input: str, spelled: bool) -> int:
    digits = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    total = 0
    for line in solution_input.split("\n"):
        numbers = []
        position = 0
        while position < len(line):
            if line[position].isnumeric():
                numbers.append(int(line[position]))
            elif spelled:
                for digit in digits:
                    if line.startswith(digit, position):
                        numbers.append(digits[digit])
                        break
            position += 1
        total += numbers[0] * 10 + numbers[-1]
    return total


@r.solution(2023, 1, Part.A)
def s_2023_1_a(solution_input: str) -> int:
    return calibration_sum(solution_input, False)


@r.solution(2023, 1, Part.B)
def s_2023_1_b(solution_input: str) -> int:
    return calibration_sum(solution_input, True)


def cube_games(solution_input: str, use_ids: bool) -> int:
    ids = 0
    powers = 0
    limits = {
        "red": 12,
        "green": 13,
        "blue": 14,
    }
    for game in solution_input.split("\n"):
        mins: dict[str, int] = {}
        i, info = game.split(":")
        id = int(i.split(" ")[1])
        sets = info.split(";")
        possible = True
        for set in sets:
            if use_ids and not possible:
                break
            cubes = set.split(",")
            for cube in cubes:
                if use_ids and not possible:
                    break
                n, color = cube.strip().split(" ")
                number = int(n)
                if use_ids and color in limits and limits[color] < number:
                    possible = False
                    break
                if not use_ids and (color not in mins or mins[color] < number):
                    mins[color] = number
        if possible:
            ids += id
        if not use_ids:
            power = 1
            for value in mins.values():
                power *= value
            powers += power
    return ids if use_ids else powers


@r.solution(2023, 2, Part.A)
def s_2023_2_a(solution_input: str) -> int:
    return cube_games(solution_input, True)


@r.solution(2023, 2, Part.B)
def s_2023_2_b(solution_input: str) -> int:
    return cube_games(solution_input, False)


def gear_ratios(solution_input: str, part_a: bool):
    total = 0
    gear_ratios = 0
    regex = compile("\d+")
    lines = solution_input.split("\n")
    symbols: set[tuple[int, int]] = set()
    gears: dict[tuple[int, int], list[int]] = {}
    for row, line in enumerate(lines):
        for col, cell in enumerate(line):
            if (not cell.isdigit()) and cell != ".":
                symbols.add((row, col))
            if cell == "*":
                gears[(row, col)] = []
    for row, line in enumerate(lines):
        numbers = regex.finditer(line)
        for number in numbers:
            value = int(number[0])
            valid = False
            for pos in range(*number.span()):
                if valid:
                    break
                for symbol in symbols:
                    if valid:
                        break
                    d_x = abs(symbol[0] - row) <= 1
                    d_y = abs(symbol[1] - pos) <= 1
                    adj = d_x and d_y
                    valid = valid or adj
                    if symbol in gears and adj:
                        gears[symbol].append(value)
            if valid:
                total += value
    for adjacent in gears.values():
        if len(adjacent) == 2:
            gear_ratios += adjacent[0] * adjacent[1]
    return total if part_a else gear_ratios


@r.solution(2023, 3, Part.A)
def s_2023_3_a(solution_input: str):
    return gear_ratios(solution_input, True)


@r.solution(2023, 3, Part.B)
def s_2023_3_b(solution_input: str):
    return gear_ratios(solution_input, False)
