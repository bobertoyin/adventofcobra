from .runner import r, Part

from hashlib import md5


def instr_to_val(instr: str) -> int:
    return 1 if instr == "(" else -1


@r.solution(2015, 1, Part.A)
def s_2015_1_a(solution_input: str) -> str:
    return str(sum([instr_to_val(instr) for instr in solution_input]))


@r.solution(2015, 1, Part.B)
def s_2015_1_b(solution_input: str) -> str:
    floor = 0
    for index, instr in enumerate(solution_input):
        floor += instr_to_val(instr)
        if floor == -1:
            return str(index + 1)
    return ""


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
def s_2015_2_a(solution_input: str) -> str:
    return str(sum([wrapping_paper(gift_dims(i)) for i in solution_input.split("\n")]))


@r.solution(2015, 2, Part.B)
def s_2015_2_b(solution_input: str) -> str:
    return str(sum([ribbon(gift_dims(i)) for i in solution_input.split("\n")]))


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
def s_2015_3_a(solution_input: str) -> str:
    positions = [(0, 0)]
    for move in solution_input:
        positions.append(move_santa(positions[-1], move))
    return str(len(set(positions)))


@r.solution(2015, 3, Part.B)
def s_2015_3_b(solution_input: str) -> str:
    santa_positions = [(0, 0)]
    robo_positions = [(0, 0)]
    for index, move in enumerate(solution_input):
        if index % 2 == 0:
            santa_positions.append(move_santa(santa_positions[-1], move))
        else:
            robo_positions.append(move_santa(robo_positions[-1], move))
    return str(len(set(santa_positions + robo_positions)))


@r.solution(2015, 4, Part.A)
def s_2015_4_a(solution_input: str) -> str:
    for num in range(0, 999_999):
        digest = md5(f"{solution_input}{num}".encode()).hexdigest()
        if digest.startswith("00000"):
            return str(num)
    return ""


@r.solution(2015, 4, Part.B)
def s_2015_4_b(solution_input: str) -> str:
    for num in range(0, 9_999_999):
        digest = md5(f"{solution_input}{num}".encode()).hexdigest()
        if digest.startswith("000000"):
            return str(num)
    return ""
