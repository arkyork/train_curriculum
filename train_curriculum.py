from DQNAgent import DQNAgent
from LineWorld import LineWorld
from utils import plot_results
def train_curriculum():
    stages = [5, 10, 20, 50]  # 簡単→難しい
    agent = DQNAgent()
    episodes_per_stage = 1000
    
    # 🔹 グラフ用に記録
    stage_rewards = []  # 各ステージの平均報酬
    all_rewards = []    # 全エピソードの推移
    
    for length in stages:
        env = LineWorld(length)
        rewards = []

        print(f"\n=== カリキュラム Stage: ゴール距離 = {length} ===")


        for epi in range(episodes_per_stage):
            state = env.reset()
            done = False
            total_reward = 0
            eps = max(0.1, 0.9 - epi * 0.001)  # ε-greedy減衰

            while not done:
                action = agent.actions(state, eps)
                next_state, reward, done = env.step(action)
                agent.store(state, action, reward, next_state, done)
                agent.one_step()
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            all_rewards.append(total_reward)

            if (epi + 1) % 50 == 0:
                avg_reward = sum(rewards[-50:]) / 50
                print(f"Episode {epi+1}: total_reward = {total_reward:.2f}, avg(last50)={avg_reward:.2f}")
        # ステージごとにターゲットネット更新
        agent.update_target()
        
        # ステージ平均を記録
        avg_stage_reward = sum(rewards) / len(rewards)
        stage_rewards.append(avg_stage_reward)
    
    # グラフ描画
    plot_results(all_rewards, stage_rewards, stages, episodes_per_stage)

if __name__ == "__main__":
    train_curriculum()
