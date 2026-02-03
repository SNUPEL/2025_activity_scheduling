import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from KA_sequencing import kahn_topological_sort_lh
from UT_calculate_variance import calculate_variance
from RL_resource_utilization import calculate_resource_profile
from UT_time_constraint import get_valid_start_times, adjust_ls_values
from RL_Baseline import baseline_greedy
from RL_training_data import generate_random_activities
from RL_Baseline import baseline_greedy



def normalize_advantage(advantages, epsilon=1e-8):
    if len(advantages) < 2: return advantages[-1]
    advantages = torch.tensor(advantages, dtype=torch.float)
    mean = advantages.mean()
    std = advantages.std()
    if std.item() < epsilon: return advantages[-1] - mean
    return ((advantages[-1] - mean) / (std + epsilon)).item() * 2.0


def train_RL(pointer_net, device, epochs=30, lr=1e-4, entropy_coef=0.001, batch_size=32):
    optimizer = optim.AdamW(pointer_net.parameters(), lr=lr)

    policy_var_history = []
    base_var_history = []
    advantage_history = []
    entropy_history = []

    advantage_buffer = []
    normalize_window = 50

    for ep in range(epochs):
        # 배치 데이터 생성
        batch_projects = [generate_random_activities() for _ in range(batch_size)]
        batch_log_probs = []
        batch_entropies = []
        batch_advantages = []

        for project_data in batch_projects:
            priority_sequence = kahn_topological_sort_lh(project_data)
            if priority_sequence is None:
                continue

            adjust_ls_values(project_data, priority_sequence)
            current_schedule = {}
            log_probs = []
            entropies = []

            for act in priority_sequence:
                vtimes = get_valid_start_times(act, current_schedule, project_data)
                if not vtimes:
                    print("No valid")
                    continue

                # 각 가능한 시작 시간에 대한 리소스 프로필 계산
                resource_profiles = calculate_resource_profile(project_data, current_schedule, act, vtimes,
                                                               fixed_length=50)
                resource_profiles = torch.tensor(resource_profiles, dtype=torch.float).unsqueeze(0).to(device)

                info = project_data[act]
                successor_count = sum(1 for _, data in project_data.items() if act in data["Predecessor"])

                feats = []
                for t_ in vtimes:
                    feats.append([
                        info["Duration"],
                        info["Resource"],
                        len(info["Predecessor"]),
                        successor_count,
                        float(len(priority_sequence) - len(current_schedule)) / len(priority_sequence),
                        float(t_)
                    ])
                feats = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)

                logits = pointer_net(feats, resource_profiles)
                probs = F.softmax(logits, dim=-1)

                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)

                dist = torch.distributions.Categorical(probs=probs)

                entropy = dist.entropy()
                entropies.append(entropy)

                action_index = dist.sample()
                log_p = dist.log_prob(action_index)
                log_probs.append(log_p)

                chosen_t = vtimes[action_index.item()]
                current_schedule[act] = chosen_t

            final_var = calculate_variance(current_schedule, project_data)

            greedy_schedule = {}
            for act in priority_sequence:
                vt = get_valid_start_times(act, greedy_schedule, project_data)
                if vt:
                    t_, inc = baseline_greedy(act, vt, greedy_schedule, project_data)
                    greedy_schedule[act] = t_
            base_var = calculate_variance(greedy_schedule, project_data)

            raw_advantage = base_var - final_var
            advantage_buffer.append(raw_advantage)
            if len(advantage_buffer) > normalize_window:
                advantage_buffer.pop(0)

            normalized_advantage = normalize_advantage(advantage_buffer)

            batch_log_probs.extend(log_probs)
            batch_entropies.extend(entropies)
            batch_advantages.extend([normalized_advantage] * len(log_probs))

        # 배치 단위 학습
        if batch_log_probs:
            batch_policy_loss = torch.stack([-lp * adv for lp, adv in zip(batch_log_probs, batch_advantages)]).mean()
            batch_entropy_loss = torch.stack(batch_entropies).mean()
            total_loss = batch_policy_loss - entropy_coef * batch_entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pointer_net.parameters(), max_norm=1.0)
            optimizer.step()

            policy_var_history.append(final_var)
            base_var_history.append(base_var)
            advantage_history.append(raw_advantage)
            entropy_history.append(batch_entropy_loss.item())

        if ep % 10 == 0:
            print(f"[Train] Epoch {ep}, final_var={final_var:.4f}, base_var={base_var:.4f}, "
                  f"raw_advantage={raw_advantage:.4f}, norm_advantage={normalized_advantage:.4f}, "
                  f"entropy={batch_entropy_loss.item():.4f}")

        # 결과 데이터를 DataFrame으로 변환
        results_df = pd.DataFrame({
            'Epoch': list(range(len(policy_var_history))),
            'Policy_Variance': policy_var_history,
            'Baseline_Variance': base_var_history,
            'Advantage': advantage_history,
            'Entropy': entropy_history
        })

        # 엑셀 파일로 저장
        excel_filename = f"training_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        results_df.to_excel(excel_filename, index=False)

    # 그래프 그리기
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # 첫 번째 그래프: Policy and Baseline Variance
    ax1.plot(policy_var_history, label="Policy Variance")
    ax1.plot(base_var_history, label="Baseline Variance")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Training Convergence")
    ax1.legend()
    ax1.grid(True)

    # 두 번째 그래프: Advantage
    ax2.plot(advantage_history, label="Advantage", color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Advantage")
    ax2.set_title("Training Advantage Over Time")
    ax2.legend()
    ax2.grid(True)

    # 세 번째 그래프: Entropy
    ax3.plot(entropy_history, label="Entropy", color='purple')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Entropy")
    ax3.set_title("Training Entropy Over Time")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n[Train] 학습 종료. 최종 모델로 테스트해봅시다.")

    return policy_var_history, base_var_history, advantage_history, entropy_history