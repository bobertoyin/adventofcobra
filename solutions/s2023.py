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
