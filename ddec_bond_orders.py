"""Read atom-pair bond orders produced by the Chargemol cleanup scripts."""

from dataclasses import dataclass
import math
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_SYMBOL = r"[A-Z][a-z]?"
_FORMATS = (
    # chargemol_cleanup_bo.awk, free format: Co(1)-O(2): 0.5432
    re.compile(rf"({_SYMBOL})\s*\(\s*(\d+)\s*\)\s*-\s*({_SYMBOL})\s*\(\s*(\d+)\s*\)\s*:\s*({_NUMBER})"),
    # chargemol_cleanup_bo.awk, CSV/whitespace: Co; 1; O; 2; 0.5432
    re.compile(rf"({_SYMBOL})\s+(\d+)\s+({_SYMBOL})\s+(\d+)\s+({_NUMBER})"),
)
_CORRECTED = re.compile(rf"(\d+)\s*\(\s*({_SYMBOL})\s*\)\s+(\d+)\s*\(\s*({_SYMBOL})\s*\)\s+({_NUMBER})")
_INDICES = re.compile(rf"(\d+)\s+(\d+)\s+({_NUMBER})")


@dataclass(frozen=True)
class DDECBondOrder:
    atom1: int  # One-based atom numbers, as in the file and fragment selector.
    atom2: int
    value: float


class DDECBondOrders:
    def __init__(self, bonds, symbols):
        self.bonds = tuple(bonds)
        self.symbols = symbols

    @classmethod
    def read(cls, path):
        directed = {}
        symbols = {}
        with open(path, encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                text = line.split("#", 1)[0].strip()
                if not text:
                    continue
                normalized = re.sub(r"[;,]", " ", text)
                row = None
                for pattern in _FORMATS:
                    match = pattern.fullmatch(normalized)
                    if match:
                        symbol1, atom1, symbol2, atom2, value = match.groups()
                        row = int(atom1), int(atom2), value, symbol1, symbol2
                        break
                if row is None:
                    match = _CORRECTED.fullmatch(normalized)
                    if match:
                        atom1, symbol1, atom2, symbol2, value = match.groups()
                        row = int(atom1), int(atom2), value, symbol1, symbol2
                if row is None:
                    match = _INDICES.fullmatch(normalized)
                    if match:
                        atom1, atom2, value = match.groups()
                        row = int(atom1), int(atom2), value, None, None
                if row is None:
                    # Permit column headings, but never silently skip malformed data.
                    if not re.search(r"\d", text) and any(
                            word in text.lower() for word in ("atom", "bond", "order")):
                        continue
                    raise ValueError(
                        f"Cannot read bond-order row {line_number}: {text}\n"
                        "Expected element; atom number; element; atom number; bond order, "
                        "or atom1 atom2 bond_order.")
                atom1, atom2, value, symbol1, symbol2 = row
                value = float(value.replace("D", "E").replace("d", "e"))
                if min(atom1, atom2) < 1 or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid atom number or bond order on row {line_number}.")
                for atom, symbol in ((atom1, symbol1), (atom2, symbol2)):
                    if symbol is not None:
                        if atom in symbols and symbols[atom] != symbol:
                            raise ValueError(f"Conflicting elements for atom {atom} on row {line_number}.")
                        symbols[atom] = symbol
                # Cleanup output can contain several periodic-image contributions.
                directed[atom1, atom2] = directed.get((atom1, atom2), 0.0) + value
        if not directed:
            raise ValueError("No atom-pair bond orders were found in this file.")
        bonds = []
        for atom1, atom2 in sorted({tuple(sorted(pair)) for pair in directed}):
            value = directed.get((atom1, atom2))
            reverse = directed.get((atom2, atom1))
            if value is None:
                value = reverse
            elif reverse is not None and atom1 != atom2:
                if not math.isclose(value, reverse, abs_tol=5e-5, rel_tol=1e-4):
                    raise ValueError(f"Conflicting bond orders for atoms {atom1} and {atom2}.")
                value = (value + reverse) / 2
            bonds.append(DDECBondOrder(atom1, atom2, value))
        return cls(bonds, symbols)

    def validate_structure(self, symbols):
        count = len(symbols)
        if any(max(bond.atom1, bond.atom2) > count for bond in self.bonds):
            raise ValueError("DDEC atom numbers exceed the number of atoms in the current structure.")
        for atom, symbol in self.symbols.items():
            if symbols[atom - 1] != symbol:
                raise ValueError(f"DDEC element for atom {atom} ({symbol}) does not match the current structure.")

    def between_fragments(self, fragment1=(), fragment2=()):
        first, second = set(fragment1), set(fragment2)
        result = []
        for bond in self.bonds:
            if bond.atom1 == bond.atom2 or bond.value == 0:
                continue
            if not first:
                result.append(bond)
            elif second:
                if ((bond.atom1 in first and bond.atom2 in second)
                        or (bond.atom2 in first and bond.atom1 in second)):
                    result.append(bond)
            elif (bond.atom1 in first) != (bond.atom2 in first):
                result.append(bond)
        return result


def parse_atom_ranges(text):
    atoms = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", item)
        if not match:
            raise ValueError(f"Invalid atom selection: {item}")
        start = int(match[1])
        end = int(match[2] or start)
        if start < 1 or end < start:
            raise ValueError(f"Invalid atom range: {item}")
        atoms.update(range(start, end + 1))
    return sorted(atoms)
