"""
Representation Engineering Pipeline — Single Entry Point

Runs the full sycophancy detection pipeline:
1. Prepare contrastive dataset
2. Measure baseline model sycophancy rate
3. Layer sweep (diff-in-means direction extraction)
4. Linear probe at optimal layer
5. Random baseline comparison
6. Activation steering demo

Usage:
    python run_repeng.py
    python run_repeng.py --model Qwen/Qwen2.5-1.5B-Instruct --batch_size 8
"""

import argparse
import json
import os
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run representation engineering pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model identifier")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for activation extraction")
    parser.add_argument("--csv_path", type=str, default="data/sycophancy_nlp_survey_clean.csv",
                        help="Path to the Anthropic sycophancy CSV")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for output artifacts")
    parser.add_argument("--max_train", type=int, default=500,
                        help="Max number of training pairs")
    parser.add_argument("--max_test", type=int, default=200,
                        help="Max number of testing pairs")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="Steering strength coefficient")
    parser.add_argument("--skip_steering", action="store_true",
                        help="Skip the steering demo (faster)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    all_metrics = {}
    t_start = time.time()

    # ── Step 1: Prepare contrastive dataset ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Preparing contrastive dataset")
    print("=" * 60)

    from repeng.data_prep import prepare_data, load_contrastive_data

    prepare_data(args.csv_path, "data", max_train_samples=args.max_train, max_test_samples=args.max_test)

    train_texts, train_labels = load_contrastive_data(split="train", data_dir="data")
    test_texts, test_labels = load_contrastive_data(split="test", data_dir="data")

    print(f"  Train: {len(train_texts)} samples ({sum(train_labels)} sycophantic)")
    print(f"  Test:  {len(test_texts)} samples ({sum(test_labels)} sycophantic)")

    train_labels_np = np.array(train_labels)
    test_labels_np = np.array(test_labels)

    # ── Step 2: Load model ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading model")
    print("=" * 60)

    from repeng.extract import load_model, get_num_layers

    model, tokenizer = load_model(args.model)
    n_layers = get_num_layers(model)
    print(f"  Model: {args.model}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")

    # ── Step 3: Baseline sycophancy rate ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Measuring baseline model sycophancy rate")
    print("=" * 60)

    from repeng.baselines import measure_sycophancy_rate, random_baseline

    rate_results = measure_sycophancy_rate(model, tokenizer, args.csv_path, max_samples=200)
    all_metrics["sycophancy_rate"] = rate_results
    print(f"  Model sycophancy rate: {rate_results['sycophancy_rate']:.1f}%")
    print(f"  ({rate_results['sycophantic_picks']}/{rate_results['total']} sycophantic choices)")

    # ── Step 4: Layer sweep ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Running layer sweep (diff-in-means)")
    print("=" * 60)

    from repeng.layer_sweep import run_layer_sweep, plot_layer_sweep

    sweep_results, best_layer, best_direction, all_X_train, all_X_test = run_layer_sweep(
        model, tokenizer,
        train_texts, train_labels_np,
        test_texts, test_labels_np,
        batch_size=args.batch_size
    )

    all_metrics["layer_sweep"] = sweep_results
    all_metrics["best_layer"] = best_layer
    best_auroc = max(r["auroc"] for r in sweep_results)
    print(f"\n  Best layer: {best_layer} (AUROC: {best_auroc:.4f})")

    # Save direction vector
    vec_path = os.path.join(args.output_dir, "sycophancy_vector.npy")
    np.save(vec_path, best_direction)
    print(f"  Direction vector saved to {vec_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, "layer_sweep.png")
    plot_layer_sweep(sweep_results, plot_path)
    print(f"  Layer sweep plot saved to {plot_path}")

    # ── Step 5: Linear probe at optimal layer ────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 5: Training linear probe at layer {best_layer}")
    print("=" * 60)

    from repeng.linear_probe import (
        train_probe, evaluate_probe, compare_directions, plot_confusion_matrix
    )

    X_train = all_X_train[best_layer]
    X_test = all_X_test[best_layer]

    clf = train_probe(X_train, train_labels_np)
    probe_metrics = evaluate_probe(clf, X_test, test_labels_np)
    all_metrics["linear_probe"] = probe_metrics

    print(f"  Accuracy: {probe_metrics['accuracy']:.4f}")
    print(f"  AUROC:    {probe_metrics['auroc']:.4f}")
    print(f"  F1:       {probe_metrics['f1']:.4f}")

    # Compare probe weights vs diff-in-means direction
    probe_weights = clf.coef_[0]
    cosine_sim = compare_directions(probe_weights, best_direction)
    all_metrics["direction_cosine_similarity"] = cosine_sim
    print(f"  Cosine sim (probe vs diff-in-means): {cosine_sim:.4f}")

    # Confusion matrix
    y_pred = clf.predict(X_test)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(test_labels_np, y_pred, cm_path)
    print(f"  Confusion matrix saved to {cm_path}")

    # ── Step 6: Random baseline ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Computing random baseline")
    print("=" * 60)

    baseline_metrics = random_baseline(test_labels_np)
    all_metrics["random_baseline"] = baseline_metrics
    print(f"  Random accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Random AUROC:    {baseline_metrics['auroc']:.4f}")

    # ── Step 7: Steering demo ────────────────────────────────────────
    if not args.skip_steering:
        print("\n" + "=" * 60)
        print("STEP 7: Running activation steering demo")
        print("=" * 60)

        from repeng.steer import run_steering_demo

        steer_path = os.path.join(args.output_dir, "steering_examples.txt")
        run_steering_demo(
            model, tokenizer, best_direction, best_layer,
            output_path=steer_path, alpha=args.alpha
        )
        print(f"  Steering examples saved to {steer_path}")
    else:
        print("\n  [Skipping steering demo (--skip_steering)]")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model:                  {args.model}")
    print(f"  Layers:                 {n_layers}")
    print(f"  Model sycophancy rate:  {rate_results['sycophancy_rate']:.1f}%")
    print(f"  Best layer:             {best_layer}")
    print(f"  Diff-in-means AUROC:    {best_auroc:.4f}")
    print(f"  Linear probe accuracy:  {probe_metrics['accuracy']:.4f}")
    print(f"  Linear probe AUROC:     {probe_metrics['auroc']:.4f}")
    print(f"  Direction cos-sim:      {cosine_sim:.4f}")
    print(f"  Random baseline AUROC:  {baseline_metrics['auroc']:.4f}")
    print(f"  Total time:             {elapsed:.1f}s")
    print("=" * 60)

    # Save all metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_metrics, default=convert))
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  All metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
