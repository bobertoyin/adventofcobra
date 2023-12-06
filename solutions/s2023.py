from collections import deque, defaultdict
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
