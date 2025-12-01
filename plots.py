import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_pickle(path: Path):
    if not path.exists():
        print(f"Skipping plot; missing file: {path}")
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def save_or_show(fig_path: Path, show: bool):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curve(total_rewards, out_dir: Path, show: bool):
    if total_rewards is None:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    save_or_show(out_dir / "learning_curve.png", show)


def plot_losses(loss_values, out_dir: Path, show: bool):
    if loss_values is None:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.grid(True)
    save_or_show(out_dir / "loss_over_time.png", show)


def plot_epsilon_decay(epsilon_values, out_dir: Path, show: bool):
    if not epsilon_values:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon")
    plt.xlabel("Training Step")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Decay Over Time")
    plt.legend()
    plt.grid(True)
    save_or_show(out_dir / "epsilon_decay.png", show)


def plot_action_counts(action_counts, out_dir: Path, show: bool):
    if action_counts is None:
        return
    action_names = {
        0: "NOOP",
        1: "FIRE",
        3: "RIGHT",
        4: "LEFT",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
    }
    sorted_action_codes = sorted(action_counts.keys())
    actions = [action_names[code] for code in sorted_action_codes]
    counts = [action_counts[code] for code in sorted_action_codes]

    plt.figure(figsize=(10, 6))
    plt.bar(actions, counts, color="skyblue")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Distribution During Training")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_or_show(out_dir / "action_distribution.png", show)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics for the Atari agent.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Directory containing training artifact .pkl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write plot .png files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of just saving them.",
    )
    args = parser.parse_args()

    total_rewards = load_pickle(args.log_dir / "total_rewards.pkl")
    loss_values = load_pickle(args.log_dir / "loss_values.pkl")
    epsilon_values = load_pickle(args.log_dir / "epsilon_values.pkl")
    action_counts = load_pickle(args.log_dir / "action_counts.pkl")

    plot_learning_curve(total_rewards, args.out_dir, args.show)
    plot_losses(loss_values, args.out_dir, args.show)
    plot_epsilon_decay(epsilon_values, args.out_dir, args.show)
    plot_action_counts(action_counts, args.out_dir, args.show)


if __name__ == "__main__":
    main()
