"""Read atom-resolved Mulliken and Löwdin charges from CHARGE.lobster."""

from math import isfinite


class LobsterChargeParser:
    def __init__(self):
        self.atom_numbers = []
        self.symbols = []
        self.mulliken = []
        self.loewdin = []

    def parse(self, file_path):
        with open(file_path, encoding="utf-8-sig") as stream:
            lines = stream.readlines()

        header = " ".join(lines[:3]).upper()
        if "MULLIKEN" not in header or not any(name in header for name in ("LOEWDIN", "LÖWDIN")):
            raise ValueError("Expected a CHARGE.lobster file with Mulliken and Löwdin charges.")

        numbers, symbols, mulliken, loewdin = [], [], [], []
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            if not fields or not fields[0].isdigit():
                continue
            try:
                number, symbol, mulliken_charge, loewdin_charge = fields
                number = int(number)
                mulliken_charge = float(mulliken_charge)
                loewdin_charge = float(loewdin_charge)
                if number != len(numbers) + 1 or not symbol.isalpha():
                    raise ValueError
                if not all(isfinite(value) for value in (mulliken_charge, loewdin_charge)):
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"Invalid CHARGE.lobster atom row at line {line_number}.") from error
            numbers.append(number)
            symbols.append(symbol)
            mulliken.append(mulliken_charge)
            loewdin.append(loewdin_charge)

        if not numbers:
            raise ValueError("CHARGE.lobster contains no atomic charges.")
        self.atom_numbers = numbers
        self.symbols = symbols
        self.mulliken = mulliken
        self.loewdin = loewdin
