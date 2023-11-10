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
        if floor == -1 : return str(index + 1)
    return str(-1)