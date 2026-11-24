"""Logging adapter outside the shared SAC package."""
import csv
import math
from pathlib import Path

LOSS_KEYS = ("actor_loss", "critic_loss", "q1_mean", "q_target_mean", "entropy", "alpha_loss", "alpha")


def plot_curves(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(run_dir)
    path = root / "train.csv"
    if not path.exists():
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    x = [float(r["env_step"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, keys, title in ((axes[0,0], ["return_avg20"], "Training return (last 20 episodes)"),
                            (axes[0,1], ["success_avg20"], "Training success (last 20 episodes)"),
                            (axes[1,0], ["actor_loss", "critic_loss"], "SAC losses"),
                            (axes[1,1], ["alpha"], "Entropy coefficient")):
        for key in keys:
            ax.plot(x, [float(r.get(key) or "nan") for r in rows], label=key)
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0,1].set_ylim(-.05, 1.05)
    temporary = root / "learning_curves.tmp.png"
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    temporary.replace(root / "learning_curves.png")


class TrainingMonitor:
    def __init__(self, logger, run_dir, optional_keys=()):
        # optional_keys: metrics the agent does not produce in this configuration
        # (e.g. alpha_loss when the entropy coefficient is fixed); they are still
        # reserved as columns but excluded from the finiteness check.
        self.logger, self.run_dir = logger, run_dir
        self.required = tuple(k for k in LOSS_KEYS if k not in optional_keys)

    def log(self, stream, row):
        row = dict(row)
        if stream == "train":
            # Shared CSVLogger fixes its header at the first row, before warm-up
            # may have produced any losses. Reserve all columns from the start.
            for key in LOSS_KEYS:
                row.setdefault(key, float("nan"))
            if row["grad_updates"] > 0 and not all(math.isfinite(row[k]) for k in self.required):
                raise FloatingPointError("Non-finite SAC metrics")
        self.logger.log(stream, row)
        if stream == "train":
            plot_curves(self.run_dir)

    def close(self):
        self.logger.close()
