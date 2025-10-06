import matplotlib.pyplot as plt


def plot_results(all_rewards, stage_rewards, stages, episodes_per_stage):
    plt.figure(figsize=(10, 5))

    # ① 全エピソードの報酬推移
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, label="Total reward per episode", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Trend (All Episodes)")
    plt.grid(True)
    plt.legend()

    # ② ステージごとの平均報酬
    plt.subplot(1, 2, 2)
    plt.bar([str(s) for s in stages], stage_rewards, color="orange")
    plt.xlabel("Goal distance (Stage length)")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Stage")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("main.png")


