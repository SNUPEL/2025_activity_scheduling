import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 필요한 모듈들 가져오기
from sequencing import kahn_topological_sort_lh
from core.Stage_2_variance import calculate_variance
from core.Stage_3_resource_profile import calculate_resource_profile
from core.Stage_4_time_constraints import get_valid_start_times
from core.Stage_5_greedy_baseline import baseline_greedy
from utils.Training_utils import generate_random_activities, adjust_ls_values


def normalize_advantage(advantages, epsilon=1e-8):
    if len(advantages) < 2: return advantages[-1]
    advantages = torch.tensor(advantages, dtype=torch.float)
    mean = advantages.mean()
    std = advantages.std()
    if std.item() < epsilon: return advantages[-1] - mean
    return ((advantages[-1] - mean) / (std + epsilon)).item() * 2.0


def train_RL(pointer_net, device, epochs=30, lr=1e-4, entropy_coef=0.001, batch_size=32):
    optimizer = optim.AdamW(pointer_net.parameters(), lr=lr)
    policy_var_history, base_var_history, advantage_history, entropy_history = [], [], [], []
    advantage_buffer, normalize_window = [], 50

    for ep in range(epochs):
        batch_projects = [generate_random_activities() for _ in range(batch_size)]
        batch_log_probs, batch_entropies, batch_advantages = [], [], []

        for project_data in batch_projects:
            priority_sequence = kahn_topological_sort_lh(project_data)
            if priority_sequence is None: continue
            adjust_ls_values(project_data, priority_sequence)
            current_schedule, log_probs, entropies = {}, [], []

            for act in priority_sequence:
                vtimes = get_valid_start_times(act, current_schedule, project_data)
                if not vtimes: continue
                resource_profiles = torch.tensor(
                    calculate_resource_profile(project_data, current_schedule, act, vtimes),
                    dtype=torch.float).unsqueeze(0).to(device)
                info = project_data[act]
                successor_count = sum(1 for _, data in project_data.items() if act in data["Predecessor"])
                feats = torch.tensor([[info["Duration"], info["Resource"], len(info["Predecessor"]), successor_count,
                                       float(len(priority_sequence) - len(current_schedule)) / len(priority_sequence),
                                       float(t_)] for t_ in vtimes], dtype=torch.float).unsqueeze(0).to(device)

                logits = pointer_net(feats, resource_profiles)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action_index = dist.sample()
                log_probs.append(dist.log_prob(action_index))
                entropies.append(dist.entropy())
                current_schedule[act] = vtimes[action_index.item()]

            final_var = calculate_variance(current_schedule, project_data)
            greedy_schedule = {}
            for act in priority_sequence:
                vt = get_valid_start_times(act, greedy_schedule, project_data)
                if vt:
                    t_, _ = baseline_greedy(act, vt, greedy_schedule, project_data)
                    greedy_schedule[act] = t_
            base_var = calculate_variance(greedy_schedule, project_data)

            raw_advantage = base_var - final_var
            advantage_buffer.append(raw_advantage)
            if len(advantage_buffer) > normalize_window: advantage_buffer.pop(0)
            batch_log_probs.extend(log_probs);
            batch_entropies.extend(entropies);
            batch_advantages.extend([normalize_advantage(advantage_buffer)] * len(log_probs))

        if batch_log_probs:
            loss = torch.stack(
                [-lp * adv for lp, adv in zip(batch_log_probs, batch_advantages)]).mean() - entropy_coef * torch.stack(
                batch_entropies).mean()
            optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(pointer_net.parameters(), 1.0);
            optimizer.step()

        if ep % 10 == 0: print(f"Epoch {ep}, Var: {final_var:.4f}, Base: {base_var:.4f}")

    torch.save(pointer_net.state_dict(), 'trained_model_att.pth')