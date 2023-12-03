# Advent of Cobra: Python Solutions for Advent of Code

My incredibly cursed setup and solutions for [Advent of Code](https://adventofcode.com).

## Usage

```console
$ pip3 install -r requirements.txt
$ export AOC_SESSION=a_really_long_session_token
$ python3 main.py -h
usage: adventofcobra [-h] year day part
$ python3 main.py 2001 12 A
Year 2001 Day 12 Part A: your answer :D
```

## Is it really an FAQ if nobody asked?

> What is `runner.py`?

`runner.py` contains code that essentially wraps the [`advent-of-code-data`](https://github.com/wimglenn/advent-of-code-data) library and auto-runs solution functions that are marked using a decorator.

> Why is your implementation of an auto-runner and your general code layout so janky?

Something something how Python modules are loaded? Something something how I implemented the auto-runner? I don't really know right now.

> Why not have your auto-runner support example inputs and auto-submitting?

AFAIK fetching example inputs is only supported in `advent-of-code-data` via the CLI, as oppposed to the library (which is what I'm using). [`advent-of-code-data`] *does* support auto-submitting but I chose not to use it due to the inherent time-out penalty that comes from submitting incorrect solutions.

> Why is your code formatted and type annotated?

I honestly don't know.
