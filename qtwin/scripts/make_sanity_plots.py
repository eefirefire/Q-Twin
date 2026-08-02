"""
Sanity-check plots from raw_timeseries_master.csv.

Priority: the two real negative-control chips (No.7, No.29 from 21 Mar) first,
since Eva's NC review is blocked on these. Then grouped overlay plots by
condition for the rest of the dataset.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

STAGE_COLORS = {"CHI": "tab:blue", "probe": "tab:orange", "target": "tab:green"}


def plot_nc_chips(master: pd.DataFrame):
    for chip_id in ["21Mar_No.7", "21Mar_No.29"]:
        chip_data = master[master["chip_id"] == chip_id]
        if chip_data.empty:
            print(f"  [WARN] no data for {chip_id}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for (stage, replicate), g in chip_data.groupby(["stage", "replicate"]):
            g = g.sort_values("Relative_time")
            ax.plot(
                g["Relative_time"],
                g["Resonance_Frequency"],
                color=STAGE_COLORS.get(stage, "gray"),
                linestyle="-" if replicate == 1 else "--",
                label=f"{stage} r{replicate}",
                alpha=0.85,
            )
        ax.set_title(f"Negative control — {chip_id}")
        ax.set_xlabel("Relative time (s)")
        ax.set_ylabel("Resonance frequency (Hz)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"NC_{chip_id.replace('.', '_')}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_grouped_conditions(master: pd.DataFrame):
    probe = master[master["stage"] == "probe"].copy()
    probe["group_key"] = probe["condition_label"].astype(str) + " (" + probe["date_folder"].astype(str) + ")"

    for group_key, g in probe.groupby("group_key"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for chip_id, chip_g in g.groupby("chip_id"):
            for replicate, rep_g in chip_g.groupby("replicate"):
                rep_g = rep_g.sort_values("Relative_time")
                ax.plot(rep_g["Relative_time"], rep_g["Resonance_Frequency"], alpha=0.7, label=f"{chip_id} r{replicate}")
        ax.set_title(f"Probe stage — {group_key}")
        ax.set_xlabel("Relative time (s)")
        ax.set_ylabel("Resonance frequency (Hz)")
        if len(ax.lines) <= 10:
            ax.legend(fontsize=7)
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() else "_" for c in group_key)
        out = FIG_DIR / f"grouped_{safe_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


def main():
    master = pd.read_csv(DATA_DIR / "raw_timeseries_master.csv")

    print("Priority: NC chips No.7 and No.29 ...")
    plot_nc_chips(master)

    print("Grouped condition plots ...")
    plot_grouped_conditions(master)


if __name__ == "__main__":
    main()
