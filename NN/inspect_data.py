from __future__ import annotations
import argparse
from pathlib import Path
import awkward as ak
import uproot
from config import (
    EVENT_FEATURES,
    MISSING_SENTINELS,
    PFO_FEATURES,
    TAU_CATEGORICAL_FEATURES,
    TAU_CONTINUOUS_FEATURES,
    TRACK_FEATURES,
    TREE_NAME,
)

DEFAULT_INPUT = (
    "/home/rbaba/tauspin-ss/"
    "MakeNtuple/outputs/H/H_chunk_000.root"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the structure and first event of a tau-spin ntuple."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to the input ROOT file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input ROOT file was not found: {input_path}"
        )

    with uproot.open(input_path) as root_file:
        print(f"Input file: {input_path}")
        print(f"Top-level objects: {root_file.keys()}")

        if TREE_NAME not in root_file:
            raise KeyError(
                f"TTree '{TREE_NAME}' was not found in {input_path}"
            )

        tree = root_file[TREE_NAME]
        branch_names = set(tree.keys())

        print(f"TTree: {TREE_NAME}")
        print(f"Entries: {tree.num_entries}")
        print(f"Number of branches: {len(branch_names)}")

        requested_features = (
            EVENT_FEATURES
            + TAU_CONTINUOUS_FEATURES
            + TAU_CATEGORICAL_FEATURES
            + TRACK_FEATURES
            + PFO_FEATURES
        )

        missing_features = [
            name
            for name in requested_features
            if name not in branch_names
        ]

        if missing_features:
            print("\nFeatures missing from the TTree:")
            for name in missing_features:
                print(f"  - {name}")
        else:
            print("\nAll features listed in config.py exist.")

        inspection_branches = [
            "eventNumber",
            "mcChannelNumber",
            "label",
            "met_et",
            "met_phi",
            "met_sumet",
            "tau_pt",
            "tau_eta",
            "tau_phi",
            "tau_charge",
            "track_pt",
            "track_tauIndex",
            "pfo_pt",
            "pfo_tauIndex",
        ]

        available_inspection_branches = [
            name
            for name in inspection_branches
            if name in branch_names
        ]

        arrays = tree.arrays(
            available_inspection_branches,
            entry_start=0,
            entry_stop=1,
            library="ak",
        )

        print("\nFirst event:")
        for branch_name in available_inspection_branches:
            value = arrays[branch_name][0]
            print(f"  {branch_name}: {ak.to_list(value)}")

        print("\nMissing-value summary:")

        for branch_name in requested_features:
            values = tree[branch_name].array(library="ak")
            flat_values = ak.flatten(values, axis=None)
            total_count = len(flat_values)

            if total_count == 0:
                print(f"  {branch_name}: no stored values")
                continue

            missing_count = sum(
                int(ak.sum(flat_values == sentinel))
                for sentinel in MISSING_SENTINELS
            )
            missing_fraction = missing_count / total_count

            print(
                f"  {branch_name}: "
                f"{missing_count}/{total_count} missing "
                f"({missing_fraction:.2%})"
            )


if __name__ == "__main__":
    main()
