from collections import deque, defaultdict, Counter
from functools import cache
from itertools import combinations
from math import lcm
from re import compile
from typing import Callable

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


def scratchcards(solution_input: str, part_a: bool):
    regex = compile("\d+")
    total = 0
    copy_transform = defaultdict(list)
    copy_list: deque[int] = deque()
    copy_final = []
    for card in solution_input.split("\n"):
        a, b = card.split(" | ")
        card_info, winning = a.split(": ")
        card_num = int(regex.search(card_info).group())
        winning = {int(num[0]) for num in regex.finditer(winning)}
        nums = [int(num[0]) for num in regex.finditer(b)]
        win_nums = [num for num in nums if num in winning]
        if len(win_nums) > 0:
            if part_a:
                total += 2 ** (len(win_nums) - 1)
            else:
                for i in range(1, len(win_nums) + 1):
                    copy_transform[card_num].append(card_num + i)
        copy_list.append(card_num)
    while len(copy_list) > 0:
        next_copy = copy_list.popleft()
        copy_final.append(next_copy)
        copy_list.extend(copy_transform[next_copy])
    return total if part_a else len(copy_final)


@r.solution(2023, 4, Part.A)
def s_2023_4_a(solution_input: str):
    return scratchcards(solution_input, True)


@r.solution(2023, 4, Part.B)
def s_2023_4_b(solution_input: str):
    return scratchcards(solution_input, False)


def fertilize_seeds(solution_input: str, part_a: bool) -> int:
    seeds = []
    mappings = {}
    source_name = None
    dest_name = None
    seed_ranges = []
    cache = defaultdict(dict)
    locations = set()
    for line in solution_input.split("\n"):
        if line == "":
            source_name = None
            dest_name = None
        elif line.startswith("seeds"):
            if part_a:
                for seed in line.split(" ")[1:]:
                    seeds.append(int(seed))
            else:
                line = line.split(" ")[1:]
                for i in range(0, len(line), 2):
                    start = int(line[i])
                    r = int(line[i + 1])
                    seed_ranges.append((start, r))
        elif "map" in line:
            map_name = line.split(" ")[0]
            map_name_split = map_name.split("-")
            source_name = map_name_split[0]
            dest_name = map_name_split[2]
            mappings[source_name] = {"new_item": dest_name, "ranges": []}
            for item in mappings:
                if mappings[item]["new_item"] == source_name:
                    mappings[source_name]["old_item"] = item
        else:
            dest, source, r = [int(n) for n in line.split(" ")]
            mappings[source_name]["ranges"].append(
                {
                    "source": source,
                    "dest": dest,
                    "range": r,
                }
            )

    if part_a:
        for seed in seeds:
            if seed not in cache["seed"]:
                item = "seed"
                value = seed
                while item != "location":
                    if value not in cache[item]:
                        ranges = mappings[item]["ranges"]
                        item = mappings[item]["new_item"]
                        for range_item in ranges:
                            source = range_item["source"]
                            dest = range_item["dest"]
                            r = range_item["range"]
                            if source <= value and value < source + r:
                                new_value = dest + (value - source)
                                cache[item][value] = new_value
                                value = new_value
                                break
                    else:
                        value = cache[item][value]
                        item = "location"
                cache["seed"][seed] = value
                locations.add(value)
        return min(locations)
    else:
        location = 0
        while True:
            value = location
            item = "humidity"
            while item is not None:
                for range_item in mappings[item]["ranges"]:
                    source = range_item["source"]
                    dest = range_item["dest"]
                    r = range_item["range"]
                    if dest <= value and value < dest + r:
                        value = source + (value - dest)
                        break
                item = mappings[item].get("old_item")
            for start, length in seed_ranges:
                if start <= value and value < start + length:
                    return location
            location += 1


@r.solution(2023, 5, Part.A)
def s_2023_5_a(solution_input: str) -> int:
    return fertilize_seeds(solution_input, True)


@r.solution(2023, 5, Part.B)
def s_2023_5_b(solution_input: str) -> int:
    return fertilize_seeds(solution_input, False)


def ferry_race(solution_input: str, part_a: bool) -> int:
    times = []
    distances = []
    ways = 1
    num_re = compile("\d+")
    for line in solution_input.split("\n"):
        if "Time" in line:
            if part_a:
                times += [int(n[0]) for n in num_re.finditer(line)]
            else:
                times.append(int("".join([n[0] for n in num_re.finditer(line)])))
        else:
            if part_a:
                distances += [int(n[0]) for n in num_re.finditer(line)]
            else:
                distances.append(int("".join([n[0] for n in num_re.finditer(line)])))
    races = zip(times, distances)
    for time, record in races:
        race_ways = 0
        for hold in range(0, time + 1):
            distance = (time - hold) * hold
            if distance > record:
                race_ways += 1
        ways *= race_ways
    return ways


@r.solution(2023, 6, Part.A)
def s_2023_6_a(solution_input: str) -> int:
    return ferry_race(solution_input, True)


@r.solution(2023, 6, Part.B)
def s_2023_6_b(solution_input: str) -> int:
    return ferry_race(solution_input, False)


def camel_cards(solution_input: str, part_a: bool) -> int:
    kinds = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
    }
    rankings = []
    for l in solution_input.split("\n"):
        hand, bid = l.split(" ")
        bid = int(bid)
        cards = Counter(hand)
        replace = None
        for card in sorted(
            cards, key=lambda c: (cards[c], card_strength(c, part_a)), reverse=True
        ):
            if card != "J":
                replace = card
                break
        if replace and not part_a:
            new_hand = hand.replace("J", replace)
            cards = Counter(new_hand)
        a = cards.most_common(1)[0]
        t = ""
        if a[1] == 5:
            t = 1
        elif a[1] == 4:
            t = 2
        else:
            a, b = cards.most_common(2)
            if a[1] == 3 and b[1] == 2:
                t = 3
            elif a[1] == 3 and b[1] == 1:
                t = 4
            elif a[1] == 2 and b[1] == 2:
                t = 5
            elif a[1] == 2:
                t = 6
            else:
                t = 7
        kinds[t].append((hand, bid))
    for kind in sorted(kinds.keys(), reverse=True):
        for duo in sorted(kinds[kind], key=lambda p: hand_strength(p[0], part_a)):
            rankings.append(duo)
    return sum([(index + 1) * duo[1] for index, duo in enumerate(rankings)])


def hand_strength(hand: str, part_a: bool) -> list[int]:
    return [card_strength(card, part_a) for card in hand]


def card_strength(card: str, part_a: bool) -> int:
    values = {
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11 if part_a else 1,
        "T": 10,
    }
    if card.isnumeric():
        return int(card)
    else:
        return values[card]


@r.solution(2023, 7, Part.A)
def s_2023_7_a(solution_input: str) -> int:
    return camel_cards(solution_input, True)


@r.solution(2023, 7, Part.B)
def s_2023_7_b(solution_input: str) -> int:
    return camel_cards(solution_input, False)


def traverse_wasteland(solution_input: str, part_a: bool) -> int:
    instructions = []
    network = {}
    for i, l in enumerate(solution_input.split("\n")):
        if i == 0:
            instructions += list(l)
        elif "=" in l:
            src, choice = l.split(" = ")
            network[src] = tuple(choice.replace("(", "").replace(")", "").split(", "))
    if part_a:
        location = "AAA"
        steps = 0
        current_instr = 0
        while location != "ZZZ":
            location = network[location][0 if instructions[current_instr] == "L" else 1]
            if current_instr == len(instructions) - 1:
                current_instr = 0
            else:
                current_instr += 1
            steps += 1
        return steps
    else:
        locations = [n for n in network if n.endswith("A")]
        steps = []
        for l in locations:
            loc = l
            step = 0
            current_instr = 0
            while not loc.endswith("Z"):
                loc = network[loc][0 if instructions[current_instr] == "L" else 1]
                if current_instr == len(instructions) - 1:
                    current_instr = 0
                else:
                    current_instr += 1
                step += 1
            steps.append(step)
        return lcm(*steps)


@r.solution(2023, 8, Part.A)
def s_2023_8_a(solution_input: str) -> int:
    return traverse_wasteland(solution_input, True)


@r.solution(2023, 8, Part.B)
def s_2023_8_b(solution_input: str) -> int:
    return traverse_wasteland(solution_input, False)


def oasis_history(solution_input: str, part_a: bool) -> int:
    total = 0
    for l in solution_input.split("\n"):
        histories = []
        histories.append([int(n) for n in l.split(" ")])
        while True:
            last_history = histories[-1]
            next_history = [
                last_history[i + 1] - last_history[i]
                for i in range(len(last_history) - 1)
            ]
            histories.append(next_history)
            if sum(next_history) == 0 and len(set(next_history)) == 1:
                break
        new_histories = []
        for history in reversed(histories):
            new_history = history
            if len(new_histories) > 0:
                new_val = history[-1 if part_a else 0]
                if part_a:
                    new_val += new_histories[-1][-1]
                    new_history.append(new_val)
                else:
                    new_val -= new_histories[-1][0]
                    new_history = [new_val] + new_history
            else:
                if part_a:
                    new_history.append(0)
                else:
                    new_history = [0] + new_history
            new_histories.append(new_history)
        total += new_histories[-1][-1 if part_a else 0]
    return total


@r.solution(2023, 9, Part.A)
def s_2023_9_a(solution_input: str) -> int:
    return oasis_history(solution_input, True)


@r.solution(2023, 9, Part.B)
def s_2023_9_b(solution_input: str) -> int:
    return oasis_history(solution_input, False)


def pipe_maze(solution_input: str, part_a: bool) -> int:
    grid = [list(l) for l in solution_input.split("\n")]
    for r_i, row in enumerate(grid):
        for c_i, cell in enumerate(row):
            if cell == "S":
                start = (r_i, c_i)
    left = grid[start[0]][start[1] - 1] in ["-", "L", "F"]
    right = grid[start[0]][start[1] + 1] in ["-", "J", "7"]
    top = grid[start[0] - 1][start[1]] in ["|", "F", "7"]
    bottom = grid[start[0] - 1][start[1]] in ["|", "L", "J"]
    if left and right:
        start_piece = "-"
    elif top and bottom:
        start_piece = "|"
    elif left and top:
        start_piece = "J"
    elif left and bottom:
        start_piece = "7"
    elif right and top:
        start_piece = "L"
    else:
        start_piece = "F"
    grid[start[0]][start[1]] = start_piece
    visited = {start: 0}
    horizon = deque([start])
    while len(horizon) > 0:
        r, c = horizon.popleft()
        neighbors = []
        pipe = grid[r][c]
        if pipe == "-":
            neighbors += [(r, c - 1), (r, c + 1)]
        elif pipe == "|":
            neighbors += [(r - 1, c), (r + 1, c)]
        elif pipe == "L":
            neighbors += [(r - 1, c), (r, c + 1)]
        elif pipe == "J":
            neighbors += [(r - 1, c), (r, c - 1)]
        elif pipe == "F":
            neighbors += [(r + 1, c), (r, c + 1)]
        elif pipe == "7":
            neighbors += [(r + 1, c), (r, c - 1)]
        for neighbor in neighbors:
            r_n, c_n = neighbor
            if (
                len(grid) > r_n >= 0
                and len(grid[r_n]) > c_n >= 0
                and neighbor not in visited
            ):
                visited[neighbor] = visited[(r, c)] + 1
                horizon.append(neighbor)
    if part_a:
        return max(visited.values())
    else:
        count = 0
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if (r, c) not in visited:
                    x_ray = (r - c, 0)
                    edges = 0
                    while x_ray != (r, c):
                        if x_ray in visited:
                            pipe = grid[x_ray[0]][x_ray[1]]
                            if pipe in ["7", "L"]:
                                edges += 2
                            else:
                                edges += 1
                        x_ray = (x_ray[0] + 1, x_ray[1] + 1)
                    if edges % 2 == 1:
                        count += 1
        return count


@r.solution(2023, 10, Part.A)
def s_2023_10_a(solution_input: str) -> int:
    return pipe_maze(solution_input, True)


@r.solution(2023, 10, Part.B)
def s_2023_10_b(solution_input: str) -> int:
    return pipe_maze(solution_input, False)


def cosmic_expansion(solution_input: str, part_a: bool) -> int:
    grid = [list(l) for l in solution_input.split("\n")]
    expanded_rows = []
    expanded = []
    for row in grid:
        if "#" not in row:
            expanded_rows.append(["x" for _ in row])
        else:
            expanded_rows.append(row)
    cols = set()
    for c in range(len(grid[0])):
        col = [row[c] for row in grid]
        if "#" not in col:
            cols.add(c)
    for row in expanded_rows:
        new_row = []
        for i, c in enumerate(row):
            if i in cols:
                new_row.append("x")
            else:
                new_row.append(c)
        expanded.append(new_row)
    galaxies = set()
    for r_i, r in enumerate(expanded):
        for c_i, c in enumerate(r):
            if c == "#":
                galaxies.add((r_i, c_i))
    total = 0
    pairs = set(combinations(galaxies, 2))
    grow = 2 if part_a else 1_000_000
    for src in galaxies:
        visited = {src: 0}
        horizon = deque([src])
        while len(horizon) > 0:
            r, c = horizon.popleft()
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if r < len(expanded) - 1:
                neighbors.append((r + 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            if c < len(expanded[r]) - 1:
                neighbors.append((r, c + 1))
            for n in neighbors:
                if n not in visited:
                    visited[n] = visited[(r, c)] + (
                        grow if expanded[n[0]][n[1]] == "x" else 1
                    )
                    horizon.append(n)
        for tgt in galaxies:
            if (src, tgt) in pairs:
                total += visited[tgt]
    return total


@r.solution(2023, 11, Part.A)
def s_2023_11_a(solution_input: str) -> int:
    return cosmic_expansion(solution_input, True)


@r.solution(2023, 11, Part.B)
def s_2023_11_b(solution_input: str) -> int:
    return cosmic_expansion(solution_input, False)


def hot_springs(solution_input: str, part_a: bool) -> int:
    counts = []
    for l in solution_input.split("\n"):
        springs, condition = l.split(" ")
        condition = [int(n) for n in condition.split(",")]
        if part_a:
            counts.append(eval_springs(springs, "", tuple(condition)))
        else:
            expanded_springs = "?".join([springs] * 5)
            expanded_condition = condition * 5
            counts.append(eval_springs(expanded_springs, "", tuple(expanded_condition)))
    return sum(counts)


@cache
def eval_springs(springs: str, last: str, conditions: tuple[int]) -> int:
    if len(springs) == 0:
        return sum(conditions) == 0
    else:
        spring = springs[0]
        rest = springs[1 : len(springs)]
        if spring == ".":
            if last != "#":
                return eval_springs(rest, spring, conditions)
            else:
                if conditions[0] != 0:
                    return 0
                return eval_springs(rest, spring, conditions[1 : len(conditions)])
        elif spring == "#":
            if len(conditions) == 0:
                return 0
            condition = conditions[0]
            rest_conditions = list(conditions[1 : len(conditions)])
            if condition == 0:
                return 0
            return eval_springs(rest, spring, tuple([condition - 1] + rest_conditions))
        else:
            return eval_springs(
                springs.replace("?", "#", 1), last, conditions
            ) + eval_springs(springs.replace("?", ".", 1), last, conditions)


@r.solution(2023, 12, Part.A)
def s_2023_12_a(solution_input: str) -> int:
    return hot_springs(solution_input, True)


@r.solution(2023, 12, Part.B)
def s_2023_12_b(solution_input: str) -> int:
    return hot_springs(solution_input, False)


def lava_island_reflections(solution_input: str, part_a: bool) -> int:
    grids = [[]]
    scores = {}
    for l in solution_input.split("\n"):
        if l == "":
            grids[-1].pop()
            grids.append([])
        else:
            grids[-1].append(list("x".join(list(l))))
            grids[-1].append(["x"] * len(grids[-1][-1]))
    grids[-1].pop()
    for i, grid in enumerate(grids):
        if part_a:
            score = grid_reflection(grid, reflects)
            if score:
                scores[i] = score
        else:
            score = grid_reflection(grid, diff_count)
            if score:
                scores[i] = score
    return sum(scores.values())


def grid_reflection(
    grid: list[list[str]], func: Callable[[list[list[str]], int], bool]
) -> int | None:
    pos = [i for i, c in enumerate(grid[0]) if c == "x"]
    reflects_x = False
    for p in pos:
        reflects_x = func(grid, p)
        if reflects_x:
            return len("".join(grid[0][0:p]).replace("x", ""))
    if not reflects_x:
        rotated = [[row[c_num] for row in grid] for c_num in range(len(grid[0]))]
        pos = [i for i, c in enumerate(rotated[0]) if c == "x"]
        reflects_y = False
        for p in pos:
            reflects_y = func(rotated, p)
            if reflects_y:
                return len("".join(rotated[0][0:p]).replace("x", "")) * 100


def diff_count(grid: list[list[str]], p: int) -> bool:
    differences = 0
    left = p - 1
    right = p + 1
    while left >= 0 and right < len(grid[0]) and differences < 2:
        col_left = [row[left] for row in grid]
        col_right = [row[right] for row in grid]
        for l, r in zip(col_left, col_right):
            if l != r:
                differences += 1
        left -= 1
        right += 1
    return differences == 1


def reflects(grid: list[list[str]], p: int) -> bool:
    for row in grid:
        left = p - 1
        right = p + 1
        while left >= 0 and right < len(row):
            col_left = [row[left] for row in grid]
            col_right = [row[right] for row in grid]
            if col_left != col_right:
                return False
            left -= 1
            right += 1
    return True


@r.solution(2023, 13, Part.A)
def s_2023_13_a(solution_input: str) -> int:
    return lava_island_reflections(solution_input, True)


@r.solution(2023, 13, Part.B)
def s_2023_13_b(solution_input: str) -> int:
    return lava_island_reflections(solution_input, False)
