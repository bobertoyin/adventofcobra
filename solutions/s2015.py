from .runner import r, Part


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
