from __future__ import annotations

import argparse
from pathlib import Path

from text_pipeline import run_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing dataset hotel bookings untuk submission SMSML.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[1] / "namadataset_raw" / "hotel_bookings.csv"),
        help="Path dataset hotel_bookings.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "namadataset_preprocessing"),
        help="Folder output data hasil preprocessing.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi test set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preprocessing(args.input, args.output, test_size=args.test_size)
    print("Preprocessing selesai.")
    print(f"Train: {result.train_path}")
    print(f"Test: {result.test_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Preprocessor: {result.preprocessor_path}")


if __name__ == "__main__":
    main()
