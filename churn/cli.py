"""Command line interface: ``python -m churn.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import GeneratorConfig, churn_rate_by, generate_customers
from .evaluate import Campaign
from .pipeline import coefficient_report
from .train import train


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _data_config(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        n_customers=args.customers,
        target_churn_rate=args.churn_rate,
        seed=args.seed,
    )


def cmd_data(args: argparse.Namespace) -> int:
    """Generate a customer table and describe the churn drivers."""
    customers = generate_customers(_data_config(args))
    print(f"{len(customers)} customers, churn rate {customers['churned'].mean():.4f}\n")
    for column in ("contract", "payment_method", "internet_service"):
        print(f"churn rate by {column}")
        print(churn_rate_by(customers, column).to_string(index=False), "\n")
    missing = customers.isna().sum()
    print("missing values per column")
    print(missing[missing > 0].to_string(), "\n")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(destination, index=False)
    print(f"wrote {destination}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train, calibrate, choose a threshold and save the artefact."""
    out_dir = Path(args.out)
    campaign = Campaign(
        offer_cost=args.offer_cost,
        margin_saved=args.margin_saved,
        acceptance_rate=args.acceptance_rate,
    )
    frame = pd.read_csv(args.data) if args.data else None
    result = train(
        data_config=_data_config(args),
        campaign=campaign,
        seed=args.seed,
        calibrate=not args.no_calibration,
        frame=frame,
    )

    rows = result.artifact.metadata["rows"]
    print(
        f"split: train={rows['train']} validation={rows['validation']} test={rows['test']}\n"
    )
    print("candidate models on the validation split")
    print(result.comparison.to_string(index=False), "\n")
    print(f"champion: {result.artifact.name}")
    print(f"threshold chosen on validation: {result.artifact.threshold:.4f}\n")

    print("held-out test metrics")
    for key, value in result.artifact.metadata["test_metrics"].items():
        print(f"  {key:<28}{value}")
    print("\ncampaign value at the committed threshold (test split)")
    for key, value in result.artifact.metadata["test_campaign_value"].items():
        print(f"  {key:<28}{value}")
    print()

    print("risk deciles on the test split")
    print(result.lift.to_string(index=False), "\n")
    print("calibration on the test split")
    print(result.calibration.to_string(index=False), "\n")

    _write(result.comparison, out_dir / "model_comparison.csv")
    _write(result.calibration, out_dir / "calibration.csv")
    _write(result.lift, out_dir / "lift_deciles.csv")
    _write(result.expected_value, out_dir / "expected_value.csv")

    report_path = out_dir / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {result.artifact.save(out_dir / 'model.joblib')}")
    return 0


def cmd_coefficients(args: argparse.Namespace) -> int:
    """Fit the interpretable baseline and print its standardised coefficients."""
    from .pipeline import TARGET, build_model, prepare_features

    customers = generate_customers(_data_config(args))
    model = build_model("logistic", seed=args.seed)
    model.fit(prepare_features(customers), customers[TARGET].to_numpy(dtype=int))
    table = coefficient_report(model)
    print("logistic coefficients on standardised inputs (largest effects first)")
    print(table.head(args.top).to_string(index=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI service with uvicorn."""
    import uvicorn

    from .service import create_app

    application = create_app(artifact_path=args.artifact)
    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
    return 0


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--customers", type=int, default=20_000)
    parser.add_argument("--churn-rate", dest="churn_rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=29)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn", description="Calibrated churn prediction with a deployable service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="generate and describe customers")
    _add_data_arguments(data_parser)
    data_parser.add_argument("--out", default="data/customers.csv")
    data_parser.set_defaults(func=cmd_data)

    train_parser = subparsers.add_parser("train", help="train and save the artefact")
    _add_data_arguments(train_parser)
    train_parser.add_argument("--data", help="optional CSV to train on instead of simulating")
    train_parser.add_argument("--offer-cost", dest="offer_cost", type=float, default=12.0)
    train_parser.add_argument("--margin-saved", dest="margin_saved", type=float, default=220.0)
    train_parser.add_argument(
        "--acceptance-rate", dest="acceptance_rate", type=float, default=0.35
    )
    train_parser.add_argument(
        "--no-calibration", dest="no_calibration", action="store_true"
    )
    train_parser.add_argument("--out", default="artifacts")
    train_parser.set_defaults(func=cmd_train)

    coefficients_parser = subparsers.add_parser(
        "coefficients", help="inspect the logistic baseline"
    )
    _add_data_arguments(coefficients_parser)
    coefficients_parser.add_argument("--top", type=int, default=15)
    coefficients_parser.set_defaults(func=cmd_coefficients)

    serve_parser = subparsers.add_parser("serve", help="serve the artefact over HTTP")
    serve_parser.add_argument("--artifact", default="artifacts/model.joblib")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
