from collections import defaultdict, deque
from copy import deepcopy
from itertools import permutations
from json import loads
from hashlib import md5
from typing import Any, Callable

from .runner import r, Part, T

json = int | str | bool | list[Any] | dict[str, Any]


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


def parse_circuit_cmd(cmd: str) -> tuple[list[str], str]:
    op, wire = [part.strip() for part in cmd.split("->")]
    return (op.split(" "), wire)


def eval_op(
    wires: dict[str, int],
    op: list[str],
    wire: str,
) -> tuple[list[str], str] | None:
    if len(op) == 1:
        lh = op[0]
        lh_search = int(lh) if lh.isdigit() else wires.get(lh)
        if lh_search is not None:
            wires[wire] = lh_search
            return None
    if len(op) == 2:
        lh = op[1]
        lh_search = int(lh) if lh.isdigit() else wires.get(lh)
        if lh_search is not None:
            wires[wire] = ~lh_search & 0xFFFF
            return None
    if len(op) == 3:
        lh, operator, rh = op
        operator = operator.lower()
        lh_search = int(lh) if lh.isdigit() else wires.get(lh)
        rh_search = int(rh) if rh.isdigit() else wires.get(rh)
        if lh_search is not None and rh_search is not None:
            if operator == "and":
                wires[wire] = lh_search & rh_search
            if operator == "or":
                wires[wire] = lh_search | rh_search
            if operator == "lshift":
                wires[wire] = lh_search << rh_search
            if operator == "rshift":
                wires[wire] = lh_search >> rh_search
            return None
    return (op, wire)


@r.solution(2015, 7, Part.A)
def s_2015_7_a(solution_input: str) -> int:
    wires: dict[str, int] = {}
    deferred: deque[tuple[list[str], str]] = deque()
    for line in solution_input.split("\n"):
        op, wire = parse_circuit_cmd(line)
        defer = eval_op(wires, op, wire)
        if defer:
            deferred.append(defer)
    while len(deferred) > 0:
        op, wire = deferred.popleft()
        defer = eval_op(wires, op, wire)
        if defer:
            deferred.append(defer)
    return wires["a"]


@r.solution(2015, 7, Part.B)
def s_2015_7_b(solution_input: str) -> int:
    prior = s_2015_7_a(solution_input)
    wires: dict[str, int] = {}
    deferred: deque[tuple[list[str], str]] = deque()
    for line in solution_input.split("\n"):
        op, wire = parse_circuit_cmd(line)
        if wire == "b":
            op = [str(prior)]
        defer = eval_op(wires, op, wire)
        if defer:
            deferred.append(defer)
    while len(deferred) > 0:
        op, wire = deferred.popleft()
        defer = eval_op(wires, op, wire)
        if defer:
            deferred.append(defer)
    return wires["a"]


def escaped_length(string: str) -> int:
    string = string[1 : len(string) - 1]
    length = 0
    skip = 0
    for index, char in enumerate(string):
        if skip > 0:
            skip -= 1
        else:
            if char == "\\":
                skip = 3 if index < len(string) - 1 and string[index + 1] == "x" else 1
            length += 1
    return length


def encode_str(string: str) -> str:
    encoded = '"'
    for char in string:
        new_char = "\\" + char if char == '"' or char == "\\" else char
        encoded += new_char
    encoded += '"'
    return encoded


@r.solution(2015, 8, Part.A)
def s_2015_8_a(solution_input: str) -> int:
    return sum(
        [len(string) - escaped_length(string) for string in solution_input.split("\n")]
    )


@r.solution(2015, 8, Part.B)
def s_2015_8_b(solution_input: str) -> int:
    return sum(
        [len(encode_str(string)) - len(string) for string in solution_input.split("\n")]
    )


def parse_location_pair(line: str) -> tuple[str, str, int]:
    source_target, distance = line.split(" = ")
    source, target = source_target.split(" to ")
    return (source, target, int(distance))


def all_complete_path_lengths(graph: dict[str, dict[str, int]]) -> list[int]:
    lengths: list[int] = []
    for path in permutations(graph.keys(), len(graph.keys())):
        length = path_length(graph, path)
        if length is not None:
            lengths.append(length)
    return lengths


def path_length(graph: dict[str, dict[str, int]], path: tuple[str, ...]) -> int | None:
    length = 0
    last = path[0]
    for node in path[1:]:
        if node in graph[last]:
            length += graph[last][node]
            last = node
        else:
            return None
    return length


@r.solution(2015, 9, Part.A)
def s_2015_9_a(solution_input: str) -> int:
    graph: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for line in solution_input.split("\n"):
        source, target, distance = parse_location_pair(line)
        graph[source][target] = distance
        graph[target][source] = distance
    lengths = all_complete_path_lengths(graph)
    return min(lengths)


@r.solution(2015, 9, Part.B)
def s_2015_9_b(solution_input: str) -> int:
    graph: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for line in solution_input.split("\n"):
        source, target, distance = parse_location_pair(line)
        graph[source][target] = distance
        graph[target][source] = distance
    lengths = all_complete_path_lengths(graph)
    return max(lengths)


def look_and_say(string: str) -> str:
    last = string[0]
    last_count = 0
    encoded = ""
    for char in string:
        if last == char:
            last_count += 1
        else:
            encoded += str(last_count) + last
            last = char
            last_count = 1
    encoded += str(last_count) + last
    return encoded


def look_and_say_multi(string: str, n: int) -> str:
    end = string
    for _ in range(n):
        end = look_and_say(end)
    return end


@r.solution(2015, 10, Part.A)
def s_2015_10_a(solution_input: str) -> int:
    return len(look_and_say_multi(solution_input, 40))


@r.solution(2015, 10, Part.B)
def s_2015_10_b(solution_input: str) -> int:
    return len(look_and_say_multi(solution_input, 50))


def increment_encoded_password(password: list[int]) -> None:
    index = len(password) - 1
    finished = False
    while not finished:
        if password[index] == 25:
            password[index] = 0
            index -= 1
        else:
            password[index] += 1
            finished = True


def valid_encoded_password(password: list[int]) -> bool:
    has_invalid_letters = 8 in password or 14 in password or 11 in password
    has_stair = False
    previous_chars: list[int] = []
    pairs = set()
    for char in password:
        if len(previous_chars) > 0:
            if char == previous_chars[-1]:
                pairs.add(char)
        if len(previous_chars) > 1:
            if (
                char - previous_chars[-1] == 1
                and previous_chars[-1] - previous_chars[-2] == 1
            ):
                has_stair = True
        previous_chars.append(char)
    return not has_invalid_letters and has_stair and len(pairs) > 1


def encode_password(password: str) -> list[int]:
    encoding = {char: val for val, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
    return [encoding[char] for char in password]


def decode_password(password: list[int]) -> str:
    decoding = {val: char for val, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
    return "".join([decoding[char] for char in password])


@r.solution(2015, 11, Part.A)
def s_2015_11_a(solution_input: str) -> str:
    password = encode_password(solution_input)
    while not valid_encoded_password(password):
        increment_encoded_password(password)
    return decode_password(password)


@r.solution(2015, 11, Part.B)
def s_2015_11_b(solution_input: str) -> str:
    password = encode_password(s_2015_11_a(solution_input))
    increment_encoded_password(password)
    while not valid_encoded_password(password):
        increment_encoded_password(password)
    return decode_password(password)


def sum_json_nums(input_json: json, ignore_red: bool) -> int:
    if type(input_json) == int:
        return input_json
    elif type(input_json) == list:
        l: list[json] = input_json
        return sum([sum_json_nums(item, ignore_red) for item in l])
    elif type(input_json) == dict:
        d: dict[str, json] = input_json
        if ignore_red and "red" in d.values():
            return 0
        return sum([sum_json_nums(value, ignore_red) for value in d.values()])
    else:
        return 0


@r.solution(2015, 12, Part.A)
def s_2015_12_a(solution_input: str) -> int:
    return sum_json_nums(loads(solution_input), False)


@r.solution(2015, 12, Part.B)
def s_2015_12_b(solution_input: str) -> int:
    return sum_json_nums(loads(solution_input), True)


def parse_seat_rel(rel: str) -> tuple[str, str, int]:
    person, target = rel.replace(".", "").split(" would ")
    effect, neighbor = target.split(" happiness units by sitting next to ")
    eff_sign, eff_val = effect.split(" ")
    val = int(eff_val)
    val *= -1 if eff_sign == "lose" else 1
    return (person, neighbor, val)


def score_arrangement(arr: list[str], rels: dict[str, dict[str, int]]) -> int:
    score = 0
    for index, person in enumerate(arr):
        left = index - 1 if index > 0 else len(arr) - 1
        right = index + 1 if index < len(arr) - 1 else 0
        score += rels[person][arr[left]]
        score += rels[person][arr[right]]
    return score


def best_arr(solution_input: str, add_self: bool) -> int:
    rels: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for rel in solution_input.split("\n"):
        person, neighbor, effect = parse_seat_rel(rel)
        rels[person][neighbor] = effect
    if add_self:
        for other in deepcopy(list(rels.keys())):
            rels["Robert"][other] = 0
            rels[other]["Robert"] = 0
    return max(
        [
            score_arrangement(list(arr), rels)
            for arr in permutations(rels.keys(), len(rels.keys()))
        ]
    )


@r.solution(2015, 13, Part.A)
def s_2015_13_a(solution_input: str) -> int:
    return best_arr(solution_input, False)


@r.solution(2015, 13, Part.B)
def s_2015_13_b(solution_input: str) -> int:
    return best_arr(solution_input, True)
