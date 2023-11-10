from argparse import ArgumentParser

from solutions import r, Part

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="adventofcobra",
        description="Run Advent of Code solutions",
    )
    parser.add_argument("year", type=int)
    parser.add_argument("day", type=int)
    parser.add_argument("part", type=Part)
    args = parser.parse_args()

    r.run(**vars(args))
