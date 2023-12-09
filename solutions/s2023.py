from collections import deque, defaultdict, Counter
from math import lcm
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
