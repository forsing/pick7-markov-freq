#!/usr/bin/env python3
"""
Loto 7/39 
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple


NUMBER_MIN, NUMBER_MAX, DRAW_SIZE = 1, 39, 7
MARKOV_WEIGHT, FREQUENCY_WEIGHT, PAIR_WEIGHT = 0.6, 0.4, 0.25
SEED = 39  # Fixed project seed. Prediction selection itself is deterministic.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
DEFAULT_LOTO_FILE = os.path.join(DATA_DIR, "loto7_4672_k67_loto_2958.csv")
DEFAULT_PLUS_FILE = os.path.join(DATA_DIR, "loto7_4672_k67_loto_plus_1714.csv")


class Loto739Predictor:
    """Draw-to-draw Markov model for one 7/39 series."""

    def __init__(self) -> None:
        self.draws: List[Tuple[int, ...]] = []
        self._prediction_cache: Dict[str, Tuple[Tuple[int, ...], float]] = {}

    def load_csv(self, filepath: str) -> None:
        """Read seven numbers from each CSV row; no date/header is required."""
        parsed: List[Tuple[int, ...]] = []
        with open(filepath, "r", newline="", encoding="utf-8-sig") as source:
            for line_number, row in enumerate(csv.reader(source), 1):
                cells = [cell.strip() for cell in row if cell.strip()]
                if not cells:
                    continue
                if len(cells) != DRAW_SIZE:
                    raise ValueError(f"row {line_number}: expected exactly 7 columns, got {len(cells)}")
                try:
                    draw = tuple(int(cell) for cell in cells)
                except ValueError as exc:
                    raise ValueError(f"row {line_number}: all values must be integers") from exc
                if any(number < NUMBER_MIN or number > NUMBER_MAX for number in draw):
                    raise ValueError(f"row {line_number}: numbers must be between 1 and 39")
                if len(set(draw)) != DRAW_SIZE:
                    raise ValueError(f"row {line_number}: a draw must contain 7 different numbers")
                parsed.append(tuple(sorted(draw)))

        if not parsed:
            raise ValueError("CSV contains no valid Loto 7/39 draws")
        self.draws = parsed
        self._prediction_cache.clear()
        print(f"[INFO] Loaded {len(self.draws)} draws from {filepath}")

    def _frequency_probabilities(self) -> Dict[int, float]:
        """Occurrence probability over every row in the complete CSV."""
        frequency = defaultdict(float)
        for draw in self.draws:
            for number in draw:
                frequency[number] += 1.0
        total_frequency = sum(frequency.values())
        return {
            number: frequency[number] / total_frequency
            for number in range(NUMBER_MIN, NUMBER_MAX + 1)
        }

    def _markov_probabilities(self) -> Dict[int, float]:
        """Probability from transitions out of the most recent draw."""
        if not self.draws:
            raise RuntimeError("Load CSV data before predicting")

        # A transition connects every number in draw t with every number in draw t+1.
        transitions = defaultdict(lambda: defaultdict(float))
        for index, current_draw in enumerate(self.draws[:-1]):
            for current in current_draw:
                for following in self.draws[index + 1]:
                    transitions[current][following] += 1.0

        markov = defaultdict(float)
        for current in self.draws[-1]:
            targets = transitions[current]
            total_targets = sum(targets.values())
            if total_targets:
                for number in range(NUMBER_MIN, NUMBER_MAX + 1):
                    markov[number] += targets[number] / total_targets
        total_markov = sum(markov.values())
        if total_markov:
            markov_prob = {n: markov[n] / total_markov for n in range(NUMBER_MIN, NUMBER_MAX + 1)}
        else:
            markov_prob = {n: 1 / NUMBER_MAX for n in range(NUMBER_MIN, NUMBER_MAX + 1)}

        return markov_prob

    def _number_probabilities(self, model: str) -> Dict[int, float]:
        markov = self._markov_probabilities()
        if model == "markov":
            return markov
        if model != "combined":
            raise ValueError("model must be 'markov' or 'combined'")

        frequency = self._frequency_probabilities()
        combined = {
            number: MARKOV_WEIGHT * markov[number]
            + FREQUENCY_WEIGHT * frequency[number]
            for number in range(NUMBER_MIN, NUMBER_MAX + 1)
        }
        total = sum(combined.values())
        return {number: combined[number] / total for number in combined}

    def _pair_log_lifts(self) -> Dict[Tuple[int, int], float]:
        """Return smoothed pair dependencies calculated from all historical rows."""
        number_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        for draw in self.draws:
            for number in draw:
                number_counts[number] += 1
            for pair in combinations(draw, 2):
                pair_counts[pair] += 1

        total_draws = len(self.draws)
        smoothing = 0.5
        lifts: Dict[Tuple[int, int], float] = {}
        for left, right in combinations(range(NUMBER_MIN, NUMBER_MAX + 1), 2):
            observed = (pair_counts[(left, right)] + smoothing) / (total_draws + smoothing)
            expected = ((number_counts[left] + smoothing) / (total_draws + smoothing)) * (
                (number_counts[right] + smoothing) / (total_draws + smoothing)
            )
            lifts[(left, right)] = math.log(observed / expected)
        return lifts

    @staticmethod
    def _pair_key(left: int, right: int) -> Tuple[int, int]:
        return (left, right) if left < right else (right, left)

    def _rank_all_combinations(self) -> Dict[str, Tuple[Tuple[int, ...], float]]:
        """Score all 15,380,937 valid combinations once for both output models."""
        markov = self._number_probabilities("markov")
        combined = self._number_probabilities("combined")
        markov_log = [0.0] + [math.log(markov[number]) for number in range(1, NUMBER_MAX + 1)]
        combined_log = [0.0] + [math.log(combined[number]) for number in range(1, NUMBER_MAX + 1)]

        pair_score = [[0.0] * (NUMBER_MAX + 1) for _ in range(NUMBER_MAX + 1)]
        for (left, right), lift in self._pair_log_lifts().items():
            pair_score[left][right] = pair_score[right][left] = PAIR_WEIGHT * lift

        best_markov_draw: Tuple[int, ...] = ()
        best_combined_draw: Tuple[int, ...] = ()
        best_markov_score = best_combined_score = float("-inf")

        # Incremental partial scores avoid recalculating all 21 pair terms at each leaf.
        for a in range(1, 34):
            markov_a, combined_a = markov_log[a], combined_log[a]
            for b in range(a + 1, 35):
                markov_b = markov_a + markov_log[b] + pair_score[a][b]
                combined_b = combined_a + combined_log[b] + pair_score[a][b]
                for c in range(b + 1, 36):
                    markov_c = markov_b + markov_log[c] + pair_score[a][c] + pair_score[b][c]
                    combined_c = combined_b + combined_log[c] + pair_score[a][c] + pair_score[b][c]
                    for d in range(c + 1, 37):
                        markov_d = markov_c + markov_log[d] + pair_score[a][d] + pair_score[b][d] + pair_score[c][d]
                        combined_d = combined_c + combined_log[d] + pair_score[a][d] + pair_score[b][d] + pair_score[c][d]
                        for e in range(d + 1, 38):
                            markov_e = markov_d + markov_log[e] + pair_score[a][e] + pair_score[b][e] + pair_score[c][e] + pair_score[d][e]
                            combined_e = combined_d + combined_log[e] + pair_score[a][e] + pair_score[b][e] + pair_score[c][e] + pair_score[d][e]
                            for f in range(e + 1, 39):
                                markov_f = markov_e + markov_log[f] + pair_score[a][f] + pair_score[b][f] + pair_score[c][f] + pair_score[d][f] + pair_score[e][f]
                                combined_f = combined_e + combined_log[f] + pair_score[a][f] + pair_score[b][f] + pair_score[c][f] + pair_score[d][f] + pair_score[e][f]
                                for g in range(f + 1, 40):
                                    markov_score = markov_f + markov_log[g] + pair_score[a][g] + pair_score[b][g] + pair_score[c][g] + pair_score[d][g] + pair_score[e][g] + pair_score[f][g]
                                    combined_score = combined_f + combined_log[g] + pair_score[a][g] + pair_score[b][g] + pair_score[c][g] + pair_score[d][g] + pair_score[e][g] + pair_score[f][g]
                                    if markov_score > best_markov_score:
                                        best_markov_score = markov_score
                                        best_markov_draw = (a, b, c, d, e, f, g)
                                    if combined_score > best_combined_score:
                                        best_combined_score = combined_score
                                        best_combined_draw = (a, b, c, d, e, f, g)

        return {
            "markov": (best_markov_draw, best_markov_score),
            "combined": (best_combined_draw, best_combined_score),
        }

    def predict(self, model: str) -> Tuple[Tuple[int, ...], float]:
        """Return the highest-scoring whole combination for the chosen model."""
        if model not in ("markov", "combined"):
            raise ValueError("model must be 'markov' or 'combined'")
        if not self._prediction_cache:
            self._prediction_cache = self._rank_all_combinations()
        return self._prediction_cache[model]

def run_dataset(name: str, filename: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 12} {name} {'=' * 12}")
    predictor = Loto739Predictor()
    predictor.load_csv(filename)
    for model, title in (
        ("markov", "MARKOV ONLY"),
        ("combined", "MARKOV + FREQUENCY"),
    ):
        print(f"\n--- NEXT {name} PREDICTION: {title} ---")
        draw, score = predictor.predict(model)
        numbers = " - ".join(f"{number:02d}" for number in draw)
        print(f"  {numbers}  (joint model score: {score:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loto 7/39 analyser with separate Markov-only and combined predictions."
    )
    parser.add_argument("--loto-file", default=DEFAULT_LOTO_FILE)
    parser.add_argument("--plus-file", default=DEFAULT_PLUS_FILE)
    parser.add_argument("--dataset", choices=("both", "loto", "plus"), default="both")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("both", "loto"):
        datasets.append(("LOTO 7/39", args.loto_file))
    if args.dataset in ("both", "plus"):
        datasets.append(("LOTO PLUS 7/39", args.plus_file))
    for name, filename in datasets:
        if not os.path.isfile(filename):
            print(f"[ERROR] {name} CSV not found: {filename}")
            continue
        try:
            run_dataset(name, filename, args)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[ERROR] {name}: {exc}")


if __name__ == "__main__":
    main()



"""
ANALIZA:

- koristi ceo Loto CSV: 2.958 izvlačenja
- koristi ceo Loto Plus CSV: 1.714 izvlačenja
- rangira svih 15.380.937 kombinacija 7/39 kao cele skupove
- računa svih 21 odnosa unutar svake kombinacije

Formula je:
- Markov deo: verovatnoća broja na osnovu prelaza iz prethodnog izvlačenja u sledeće.
- Kombinovani deo: 60% Markov + 40% ukupna frekvencija kroz ceo CSV.
- Zajednički deo: za svaku kombinaciju dodaje se odnos svih 21 parova, 
  prema tome koliko su se zajedno pojavljivali u punoj istoriji u odnosu na očekivanje.

Važna tehnička stvar: 
oznaka „Markov only” trenutno nije potpuno čist Markov, 
jer i taj rezultat koristi zajednički skor parova. 
Razlika je što ne koristi frekvencijski deo. 
Score je logaritamski; manje negativan rezultat je jači samo unutar istog modela i istog CSV-a.



Kod kombinacije od 7 brojeva postoje svi ovi parovi:
- prvi sa ostalih 6
- drugi sa narednih 5
- treći sa naredna 4
- četvrti sa naredna 3
- peti sa naredna 2
- šesti sa poslednjim 1
Ukupno:
6 + 5 + 4 + 3 + 2 + 1 = 21

Model proverava koliko je svaki od tih 21 parova kroz celu istoriju bio zastupljen zajedno, 
u odnosu na ono što bi se očekivalo iz pojedinačnih pojavljivanja brojeva.
"""




"""
RUN:

============ LOTO 7/39 ============
[INFO] Loaded 2958 draws from /Users/4c/Desktop/GHQ/data/loto7_4672_k67_loto_2958.csv

--- NEXT LOTO 7/39 PREDICTION: MARKOV ONLY ---
  08 - 11 - 22 - 26 - 33 - 34 - 38  (joint model score: -25.579002)

--- NEXT LOTO 7/39 PREDICTION: MARKOV + FREQUENCY ---
  08 - 11 - 22 - 23 - 26 - 33 - 34  (join model score: -25.635686)

============ LOTO PLUS 7/39 ============
[INFO] Loaded 1714 draws from /Users/4c/Desktop/GHQ/data/loto7_4672_k67_loto_plus_1714.csv

--- NEXT LOTO PLUS 7/39 PREDICTION: MARKOV ONLY ---
  07 - 09 - 14 - 23 - 26 - 27 - 38  (joint model score: -25.436365)

--- NEXT LOTO PLUS 7/39 PREDICTION: MARKOV + FREQUENCY ---
  07 - 08 - 11 - 23 - 27 - 35 - 37  (joint model score: -25.504187)

"""
