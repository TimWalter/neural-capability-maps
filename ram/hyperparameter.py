from pathlib import Path

import argparse
import optuna

from train import train


def objective(trial: optuna.Trial) -> float:
    print(f"[TRIAL {trial.number}]")
    kwargs.update({
        "hyperparameter": {
            "dim_encoding": trial.suggest_int("dim_encoding", 16, 1024, step=16),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
            "drop_prob": trial.suggest_float("drop_prob", 0.0, 1.0),
            "dim_decoder": trial.suggest_int("dim_decoder", 128, 4096, step=64),
            "num_decoder_layer": trial.suggest_int("num_decoder_layer", 1, 12),

        },
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    })
    return train(**kwargs, pretrain=-1, early_stopping=-1, trial=trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--study_name", type=str, default="RAM-Hyperparameter")

    parser.add_argument("--training_set_path", type=str, default="train")
    parser.add_argument("--validation_set_path", type=str, default="val")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--validation_interval", type=int, default=100_000)
    parser.add_argument("--stop_after", type=int, default=1_000_000)
    args = parser.parse_args()

    storage_path = Path(__file__).parent.parent / "data" / args.study_name
    storage_path.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(study_name=args.study_name,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage=f"sqlite:///{storage_path.resolve()}/hyperparameter.sqlite3",
                                load_if_exists=True)
    args.group = args.study_name
    del args.study_name
    kwargs = vars(args)
    study.optimize(objective, n_trials=100, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
