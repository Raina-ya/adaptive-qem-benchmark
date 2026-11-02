"""Regenerate only artifacts affected by deterministic readout correction."""

from src.generate_days08_12 import generate_day09, generate_day11, generate_day12, prepare


def main() -> None:
    prepare()
    generate_day09(write_figure=False)
    generate_day11(write_figure=False)
    generate_day12(write_derived=False)


if __name__ == "__main__":
    main()
