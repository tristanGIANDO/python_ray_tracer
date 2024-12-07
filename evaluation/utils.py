import csv
from pathlib import Path
from typing import Any


def create_csv_file(csv_file: Path, columns: list[str]) -> None:
    if csv_file.exists():
        return

    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def populate_csv_file(csv_file: Path, row: list[Any]) -> None:
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} does not exist")

    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)
