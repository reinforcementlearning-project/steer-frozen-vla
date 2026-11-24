import csv
from rl.logger import CSVLogger
from scripts import training_monitor


def test_loss_columns_survive_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(training_monitor, "plot_curves", lambda _: None)
    monitor = training_monitor.TrainingMonitor(CSVLogger(str(tmp_path)), tmp_path)
    monitor.log("train", {"env_step": 100, "grad_updates": 0})
    row = {"env_step": 300, "grad_updates": 45}
    row.update({key: .25 for key in training_monitor.LOSS_KEYS})
    monitor.log("train", row)
    monitor.close()
    with (tmp_path / "train.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["critic_loss"] == "nan"
    assert rows[1]["actor_loss"] == "0.25"
    assert rows[1]["critic_loss"] == "0.25"
