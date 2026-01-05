import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict, deque, OrderedDict
import numpy as np
import random
import time
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import itertools

## 조선소 데이터 적용을 위한 전처리 ##
class activitys:
    def __init__(self, shipname=None, block=None, item=None, workname=None, shop=None
                 , lead_time_1=None, lead_range=None, lead_area=None, lead_weight=None ,startdate =None, enddate =None,
                 startBuffer = None, endBuffer = None, activity_name=None,block_group=None, block_sequence=None,
                 block_pair=None):
        self.shipname = shipname
        self.block = block
        self.item = item
        self.workname = workname
        self.shop = shop
        self.lead_time_1 =lead_time_1
        self.lead_range = lead_range
        self.lead_weight=lead_weight
        self.lead_area = lead_area
        self.startdate = startdate
        self.enddate = enddate
        self.startBuffer = startBuffer
        self.endBuffer = endBuffer
        self.activity_name =activity_name
        self.block_group = block_group
        self.block_sequence = block_sequence
        self.block_pair = block_pair

def import_schedule(filepath):
    df = pd.read_excel(filepath, sheet_name='선행도장ACT정보',  engine='openpyxl' )
    df_SHOP = df[(df['SHOP'] == 'OP1')|(df['SHOP'] == 'SP4')|(df['SHOP'] == 'SP7')|(df['SHOP'] == 'SP1')]
    # df_SHOP = df[(df['SHOP'] == 'SP7')]
    activity_dict = OrderedDict()
    block_groups = set()  # 고유한 블록 순서 그룹을 저장할 집합

    for i, activity_class in df_SHOP.iterrows():
        activity_dict[str(activity_class['품목']) + str(activity_class['호선ZZACTCODE'])] \
                = activitys(shipname=activity_class['호선']
                            , block=activity_class['탑재블록'],
                            item=activity_class['품목'],
                            workname=activity_class['작업내역']
                            , shop=activity_class['SHOP'],
                            lead_time_1=activity_class['초기완료일'] - activity_class['초기시작일']
                            , lead_range=activity_class['공기조정범위']
                            , lead_area=activity_class['소지면적']
                            , startdate=(activity_class['초기시작일']),
                            startBuffer=(activity_class['선행버퍼'])
                            , enddate=(activity_class['초기완료일']),
                            endBuffer=(activity_class['후행버퍼']),
                            activity_name=(activity_class['품목'])
                            , block_group=(activity_class['블록순서그룹'])
                            , block_sequence=(activity_class['블록순서'])
                            , block_pair=(activity_class['짝블록']))
        # 블록 순서 그룹 추가
        block_groups.add(activity_class['블록순서그룹'])
    # 블록 순서 그룹의 개수 출력
    print(f"Number of unique block groups: {len(block_groups)}")
    return activity_dict


def convert_to_project_data(activity_dict):
    project_data = {}
    print("Converting data...")
    try:
        # 먼저 기본 데이터 변환
        for key, activity in activity_dict.items():
            es = activity.startdate
            ls = activity.enddate
            resource = activity.lead_area / activity.lead_time_1
            project_data[key] = {
                'ES': es,
                'LS': ls,
                'Duration': activity.lead_time_1,
                'Resource': resource,
                'Predecessor': set()
            }

        # 시퀀스별로 활동 그룹화
        seq_activities = {}
        for key, activity in activity_dict.items():
            group = activity.block_group
            seq = activity.block_sequence

            if group not in seq_activities:
                seq_activities[group] = {}

            if seq not in seq_activities[group]:
                seq_activities[group][seq] = []

            seq_activities[group][seq].append(key)

        # 각 그룹 내에서 시퀀스 기반 선행자 설정
        for group, sequences in seq_activities.items():
            sorted_seqs = sorted(sequences.keys())

            for i in range(1, len(sorted_seqs)):
                current_seq = sorted_seqs[i]
                prev_seq = sorted_seqs[i - 1]

                # 현재 시퀀스의 모든 활동에 대해
                for current_key in sequences[current_seq]:
                    # 이전 시퀀스의 모든 활동을 선행자로 추가
                    for prev_key in sequences[prev_seq]:
                        project_data[current_key]['Predecessor'] = project_data[current_key]['Predecessor'].union(
                            {prev_key})
                        print(f"Added predecessor: {prev_key} -> {current_key}")

        # set을 tuple로 변환 (CP solver 요구사항)
        for key in project_data:
            project_data[key]['Predecessor'] = tuple(project_data[key]['Predecessor'])

        print("Conversion successful. Number of activities:", len(project_data))
        return project_data
    except Exception as e:
        print("Error in conversion:", str(e))
        import traceback
        traceback.print_exc()
        return None

############################################
# 1. 학습 데이터셋 (TRAIN)
############################################

## 정규 분포를 따른 데이터 셋 ##
def generate_random_activities(n=30, mu=1.6, sigma=0.5):
    activities = OrderedDict()

    # numpy의 로그정규분포를 사용하여 작업 시간 생성
    for i in range(n):
        name = chr(65 + (i % 26)) + str(i // 26 + 1)
        es = 0
        ls = 80

        # 유니폼 분포 대신 로그정규분포를 사용하여 작업 시간 생성
        duration = np.random.lognormal(mean=mu, sigma=sigma)
        # 적절한 값으로 반올림 (필요에 따라 조정 가능)
        duration = max(1, round(duration))

        resource = random.uniform(5.0, 11.0)  # 리소스는 유니폼 분포 유지

        predecessors = []
        if i > 0:
            for j in range(max(0, i - 3), i):
                if random.random() < 0.25:
                    pred_name = chr(65 + (j % 26)) + str(j // 26 + 1)
                    predecessors.append(pred_name)

        activities[name] = {
            "ES": es,
            "LS": ls,
            "Duration": duration,
            "Resource": resource,
            "Predecessor": tuple(predecessors)
        }
    return activities

############################################
# 2. 우선 순위 알고리즘(Kahn's Algorithm)
############################################

def kahn_topological_sort_lh(project_data):
    activities = list(project_data.keys())
    graph = defaultdict(set)
    for act in activities:
        for pred in project_data[act]["Predecessor"]:
            graph[pred].add(act)

    in_degree = defaultdict(int)
    for node in activities:
        in_degree[node] = 0
    for preds in graph.values():
        for node in preds:
            in_degree[node] += 1

    # 우선순위 큐 대신 리스트를 사용하고 수동으로 정렬

    # (in-degree가 0인 노드들을 작업기간*자원요구량 기준으로 정렬)
    zero_in_degree = [node for node in activities if in_degree[node] == 0]

    # 작업기간*자원요구량이 큰 순서대로 정렬
    zero_in_degree.sort(key=lambda node: -project_data[node]["Duration"] * project_data[node]["Resource"])

    queue = deque(zero_in_degree)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        # 새로 in-degree가 0이 된 노드들을 위한 임시 리스트
        new_zero_in_degree = []

        for succ in graph[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                new_zero_in_degree.append(succ)

        # 새로운 노드들을 작업기간*자원요구량 기준으로 정렬하여 큐에 추가
        new_zero_in_degree.sort(key=lambda node: -project_data[node]["Duration"] * project_data[node]["Resource"])
        queue.extend(new_zero_in_degree)

    if len(order) != len(activities):
        print("Error: cycle detected")
        return None
    return order
############################################
# 3. Reinforcement Learning 학습 코드
############################################
def adjust_ls_values(project_data, priority_sequence):
    """Finish-to-Finish 관계로 LS를 재조정"""
    for activity in reversed(priority_sequence):
        info = project_data[activity]
        duration = info["Duration"]
        current_ls = info["LS"]
        successors = []
        for act, data in project_data.items():
            if activity in data["Predecessor"]:
                successors.append(act)
        if successors:
            min_succ_finish = min(project_data[s]["LS"] + project_data[s]["Duration"] for s in successors)
            new_ls = min(current_ls, min_succ_finish - duration)
            project_data[activity]["LS"] = new_ls
            # print(f"Adjusted LS for Activity {activity}: {current_ls} -> {new_ls}")

# 최종의 시작 시간을 결정할 수 있도록 LS 수정(추론)
def adjust_ls_values_infer(project_data, priority_sequence, activity_dict):
    """시퀀스 번호를 고려하여 Finish-to-Finish 관계로 LS를 재조정"""
    # 액티비티별 시퀀스 번호 추출
    sequence_map = {}
    for act in priority_sequence:
        if act in activity_dict:
            sequence_map[act] = activity_dict[act].block_sequence

    # 시퀀스 번호별로 액티비티 그룹화
    sequence_groups = {}
    for act, seq in sequence_map.items():
        if seq not in sequence_groups:
            sequence_groups[seq] = []
        sequence_groups[seq].append(act)

    # 시퀀스 번호 기준 정렬 (큰 번호부터)
    sorted_seqs = sorted(sequence_groups.keys(), reverse=True)

    # 각 시퀀스 그룹 내에서 LS 조정
    for seq in sorted_seqs:
        activities = sequence_groups[seq]

        for activity in activities:
            info = project_data[activity]
            duration = info["Duration"]
            current_ls = info["LS"]

            # 시퀀스 번호가 더 큰(다음 시퀀스) 액티비티만 후행자로 고려
            successors = []
            for act, data in project_data.items():
                if activity in data["Predecessor"] and act in sequence_map:
                    # 시퀀스 번호가 현재보다 큰 경우만 고려
                    if sequence_map[act] > seq:
                        successors.append(act)

            if successors:
                min_succ_finish = min(project_data[s]["LS"] + project_data[s]["Duration"] for s in successors)
                new_ls = min(current_ls, min_succ_finish - duration)

                # ES보다 작아지지 않도록 보호
                if new_ls < info["ES"]:
                    new_ls = info["ES"]
                    print(f"Warning: LS for {activity} would be less than ES. Setting LS = ES.")

                project_data[activity]["LS"] = new_ls

def calculate_variance(schedule, project_data):
    if not schedule:
        return 0.0

    # CP Optimizer와 동일한 방식: LS + Duration
    max_t = max(int(data['LS']) + int(data['Duration'])
                for data in project_data.values())

    resource_usage = np.zeros(max_t)
    for act, st in schedule.items():
        dur = project_data[act]["Duration"]
        res = project_data[act]["Resource"]
        for t in range(int(st), int(st + dur)):
            if t < max_t:  # 안전 체크 추가
                resource_usage[t] += res

    avg = np.mean(resource_usage)
    var = np.mean((resource_usage - avg) ** 2)
    return var

def get_valid_start_times(activity, current_schedule, project_data):
    info = project_data[activity]
    base_es = info["ES"]
    ls = info["LS"]
    duration = info["Duration"]
    earliest_start = base_es
    for pred in info["Predecessor"]:
        if pred in current_schedule:
            pred_finish = current_schedule[pred] + project_data[pred]["Duration"]
            earliest_start = max(earliest_start, pred_finish - duration)

    # 부동소수점을 정수로 변환
    earliest_start = int(earliest_start)
    ls = int(ls)

    valid_times = []
    for t in range(earliest_start, ls + 1):
        if t >= 0:
            valid_times.append(t)
    return valid_times


class SimplePointerNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_heads=8, resource_dim=50):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 리소스 프로필을 위한 1D CNN 또는 선형 레이어
        self.resource_embed = nn.Sequential(
            nn.Linear(resource_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # 통합된 특성 차원 처리
        self.combined_fc = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

        # Multi head attention 활용 #
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.context = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, resource_profiles):
        batch_size, num_options = x.size(0), x.size(1)

        # 기본 특성 임베딩
        enc_h = self.embedding(x)  # (배치, 시간옵션, hidden_dim)

        # 리소스 프로필 처리 - 전체 패턴 활용 (평균 대신!)
        resource_h = self.resource_embed(resource_profiles)  # (배치, 시간옵션, hidden_dim//2)

        # 활동 특성과 리소스 특성 결합
        combined_h = torch.cat([enc_h, resource_h], dim=2)
        combined_h = self.combined_fc(combined_h)

        # 단일 어텐션으로 모든 옵션 간의 관계 학습
        ### 어텐션 활용 ###
        attn_out, _ = self.attn(combined_h, combined_h, combined_h)
        enc_h = combined_h + attn_out
        enc_h = self.fc(enc_h)

        # # MLP 활용 #
        # enc_h = self.fc(combined_h)

        # 최종 점수 계산
        query = self.context.expand(batch_size, -1, -1)
        logits = torch.bmm(query, enc_h.transpose(1, 2)).squeeze(1)

        return logits

def calculate_resource_profile(project_data, current_schedule, act, vtimes, fixed_length=50):
    """동적 관심 구간만 추출하여 고정 길이로 패딩"""
    if not vtimes:
        return np.array([])

    duration = project_data[act]["Duration"]
    resource = project_data[act]["Resource"]

    # 관심 구간 계산: 최소 시작시간 ~ 최대 종료시간
    min_start = min(vtimes)
    max_end = max(vtimes) + duration
    window_size = int(max_end - min_start)

    profiles = []

    for start_time in vtimes:
        # 관심 구간만큼의 프로필 생성
        profile = np.zeros(window_size)

        # 현재 스케줄의 리소스 사용 반영 (관심 구간 내에서만)
        for scheduled_act, st in current_schedule.items():
            act_dur = project_data[scheduled_act]["Duration"]
            act_res = project_data[scheduled_act]["Resource"]

            # 관심 구간과 겹치는 부분만 계산
            overlap_start = max(int(st), min_start)
            overlap_end = min(int(st + act_dur), max_end)

            for t in range(overlap_start, overlap_end):
                if min_start <= t < max_end:
                    profile[t - min_start] += act_res

        # 현재 활동의 리소스 사용 추가
        act_start_in_window = int(start_time - min_start)
        act_end_in_window = int(start_time + duration - min_start)

        for t in range(act_start_in_window, act_end_in_window):
            if 0 <= t < window_size:
                profile[t] += resource

        # 고정 길이로 패딩/트렁케이션
        if len(profile) < fixed_length:
            # 패딩: 뒤쪽을 0으로 채움
            padded_profile = np.pad(profile, (0, fixed_length - len(profile)), 'constant')
        else:
            # 트렁케이션: 앞쪽 fixed_length만 사용
            padded_profile = profile[:fixed_length]

        profiles.append(padded_profile)

    return np.array(profiles)
# 학습을 위한 Baseline: B&K Heuristic
def baseline_greedy(activity, valid_times, schedule, project_data):
    best_t = None
    best_increase = 999999
    base_var = calculate_variance(schedule, project_data)
    for t in valid_times:
        temp_schedule = dict(schedule)
        temp_schedule[activity] = t
        new_var = calculate_variance(temp_schedule, project_data)
        increase = new_var - base_var
        if increase < best_increase:
            best_increase = increase
            best_t = t
    return best_t, best_increase


# Advantage 정규화
def normalize_advantage(advantages, epsilon=1e-8):
    if len(advantages) < 2:  # 데이터가 충분하지 않은 경우
        return advantages[-1]

    advantages = torch.tensor(advantages, dtype=torch.float)
    mean = advantages.mean()

    # 안전한 표준편차 계산
    std = advantages.std()
    if std.item() < epsilon:  # 표준편차가 너무 작은 경우
        return advantages[-1] - mean

    return ((advantages[-1] - mean) / (std + epsilon)).item() * 2.0  # scale factor 추가


def train_RL(pointer_net, epochs=30, lr=1e-4, entropy_coef=0.001, batch_size=32):
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

###########################################################
# 테스트 함수 (Sample 추론)
###########################################################

# 일반 순서를 매겨서 각각 추론하는 방식
def inference(project_data, pointer_net,device):
    pointer_net.eval()
    best_schedule = None
    best_greedy = None
    best_var = float('inf')
    n_sampling = 10

    for _ in range(n_sampling):
        priority_sequence = kahn_topological_sort_lh(project_data)
        if priority_sequence is None:
            continue

        adjust_ls_values(project_data, priority_sequence)
        current_schedule = {}
        schedule_greedy = {}

        # 그리디 스케줄링
        for act in priority_sequence:
            vt = get_valid_start_times(act, schedule_greedy, project_data)
            if vt:
                t_, _ = baseline_greedy(act, vt, schedule_greedy, project_data)
                schedule_greedy[act] = t_

        # 강화학습 스케줄링
        for act in priority_sequence:
            vtimes = get_valid_start_times(act, current_schedule, project_data)
            if not vtimes:
                continue

            # 각 가능한 시작 시간에 대한 리소스 프로필 계산
            resource_profiles = calculate_resource_profile(
                project_data, current_schedule, act, vtimes)
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

            with torch.no_grad():
                logits = pointer_net(feats, resource_profiles)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action_index = dist.sample()
                temp_t = vtimes[action_index.item()]
                current_schedule[act] = temp_t

        # 분산 계산
        final_var = calculate_variance(current_schedule, project_data)
        if final_var < best_var:
            best_var = final_var
            best_schedule = current_schedule.copy()
            best_greedy = schedule_greedy.copy()

    greedy_var = calculate_variance(best_greedy, project_data)
    return best_schedule, best_greedy, best_var, greedy_var

# 빔서치를 활용한 그룹 별 추론
def inference_by_groups(project_data, pointer_net, activity_dict, beam_width=10):

    start_time = time.time()
    pointer_net.eval()

    # 그룹별로 활동 분류
    groups = defaultdict(list)
    for key, activity in activity_dict.items():
        if key in project_data:  # project_data에 있는 활동만 포함
            groups[activity.block_group].append(key)

    # 그룹별 총 자원 요구량 계산
    group_resources = {}
    for group_id, group_activities in groups.items():
        total_resource = sum(project_data[act]["Resource"] * project_data[act]["Duration"]
                             for act in group_activities)
        group_resources[group_id] = total_resource

    # 자원 요구량이 큰 순서대로 그룹 ID 정렬
    sorted_groups = sorted(group_resources.keys(),
                           key=lambda g: group_resources[g], reverse=True)

    print("그룹 처리 순서 (자원 요구량 기준):")
    for g in sorted_groups:
        print(f"Group {g}: {group_resources[g]:.2f} 자원 단위")

    # 정책 네트워크 결과 스케줄 초기화
    final_schedule = {}

    # 자원 요구량 순으로 그룹 처리
    for group_id in sorted_groups:
        group_activities = groups[group_id]
        num_activities = len(group_activities)

        # 액티비티 개수에 따라 빔 너비 동적 조정 (선택적)
        # local_beam_width = min(beam_width, max(2, num_activities))
        local_beam_width = beam_width


        print(f"Processing group {group_id} with {num_activities} activities - using beam width {local_beam_width}")

        # 그룹 내 활동만 포함하는 서브셋 생성
        group_data = {}
        for key in group_activities:
            group_data[key] = project_data[key].copy()

            # 그룹 내 선행자만 유지
            filtered_preds = tuple(p for p in project_data[key]["Predecessor"]
                                   if p in group_activities)
            group_data[key]["Predecessor"] = filtered_preds

        # Kahn 알고리즘을 사용한 위상 정렬
        priority_sequence = kahn_topological_sort_lh(group_data)
        if priority_sequence is None:
            print(f"Error: cycle detected in group {group_id}")
            continue

        adjust_ls_values_infer(group_data, priority_sequence,activity_dict)

        # 빔 서치 초기화: 빈 스케줄로 시작
        # 각 빔은 (스케줄 사전, 현재까지의 분산) 튜플로 저장
        beams = [({}, 0.0)]  # 초기 빔은 빈 스케줄

        # 우선순위 시퀀스에 따라 각 액티비티를 순차적으로 처리
        for act_idx, act in enumerate(priority_sequence):
            # 모든 현재 빔에 대해 가능한 다음 액티비티 시작 시간 탐색
            new_beams = []

            for beam_schedule, _ in beams:
                # 이전 그룹 스케줄과 현재 빔 스케줄 합치기
                combined_schedule = {**final_schedule, **beam_schedule}
                vtimes = get_valid_start_times(act, combined_schedule, group_data)

                if not vtimes:
                    continue  # 유효한 시작 시간이 없으면 이 빔은 무시

                # 리소스 프로필 계산
                resource_profiles = calculate_resource_profile(
                    project_data, combined_schedule, act, vtimes)
                resource_profiles = torch.tensor(resource_profiles, dtype=torch.float).unsqueeze(0).to(device)

                # 특성 벡터 생성
                info = project_data[act]
                successor_count = sum(1 for a, data in group_data.items()
                                      if act in data["Predecessor"])

                feats = []
                for t_ in vtimes:
                    feats.append([
                        info["Duration"],
                        info["Resource"],
                        len(info["Predecessor"]),
                        successor_count,
                        float(len(priority_sequence) - len(beam_schedule)) / len(priority_sequence),
                        float(t_)
                    ])
                feats = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)

                # 모델로 모든 가능한 시작 시간의 점수 계산
                with torch.no_grad():
                    logits = pointer_net(feats, resource_profiles)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                # 각 가능한 시작 시간을 평가하고 상위 후보 유지
                # 여기서는 모든 가능한 시작 시간을 고려하되, 점수가 높은 순서대로 정렬
                candidates = []
                for i, t_ in enumerate(vtimes):
                    # 새 스케줄 생성
                    new_schedule = beam_schedule.copy()
                    new_schedule[act] = t_

                    # 현재 스케줄의 분산 계산
                    combined_new_schedule = {**final_schedule, **new_schedule}
                    var = calculate_variance(combined_new_schedule, project_data)

                    # 후보 저장 (분산이 낮을수록, 확률이 높을수록 좋음)
                    # 점수 = 1/분산 + 확률
                    score = (1.0 / (var + 1e-6)) + probs[i]  # 분산의 역수와 확률 결합
                    candidates.append((new_schedule, var, score))

                # 후보들을 점수 기준으로 정렬 (높은 점수가 좋음)
                candidates.sort(key=lambda x: x[2], reverse=True)

                # 상위 후보만 추가 (빔 너비 제한)
                for candidate in candidates[:local_beam_width]:
                    new_beams.append((candidate[0], candidate[1]))  # (스케줄, 분산) 저장

            # 모든 빔의 모든 후보 중에서 분산이 가장 낮은 순으로 정렬
            new_beams.sort(key=lambda x: x[1])

            # 상위 빔 너비 개만 유지
            beams = new_beams[:local_beam_width]

            # 진행 상황 출력 (선택적)
            if act_idx % 5 == 0 or act_idx == len(priority_sequence) - 1:
                print(f"  Processed {act_idx + 1}/{len(priority_sequence)} activities, "
                      f"best variance so far: {beams[0][1] if beams else float('inf'):.4f}")

        # 최종적으로 가장 좋은 빔 선택 (분산이 가장 낮은 것)
        if beams:
            best_schedule, best_var = beams[0]
            print(f"Group {group_id} completed with variance: {best_var:.4f}")

            # 현재 그룹의 최종 결과를 전체 스케줄에 병합
            final_schedule.update(best_schedule)
        else:
            print(f"Warning: No valid schedules found for group {group_id}")

    end_time = time.time()
    exe_time = end_time - start_time
    print(f"Total execution time: {exe_time:.2f} seconds")

    # 전체 프로젝트에 대한 그리디 알고리즘 실행
    print("Running greedy algorithm on entire project...")
    schedule_greedy = {}

    # 전체 프로젝트에 대한 위상 정렬
    full_priority_sequence = kahn_topological_sort_lh(project_data)

    # LS 값 조정 (전체 프로젝트 데이터에 대해)
    adjust_ls_values_infer(project_data, full_priority_sequence, activity_dict)

    # 조정된 LS 값으로 그리디 스케줄링
    for act in full_priority_sequence:
        vt = get_valid_start_times(act, schedule_greedy, project_data)
        if vt:
            t_, _ = baseline_greedy(act, vt, schedule_greedy, project_data)
            schedule_greedy[act] = t_
        else:
            print(f"Cannot schedule activity {act}: no valid start times")

    # 최종 분산 계산
    final_var = calculate_variance(final_schedule, project_data)
    greedy_var = calculate_variance(schedule_greedy, project_data)

    print(f"Final results: Policy variance = {final_var:.4f}, Greedy variance = {greedy_var:.4f}")
    print(f"Improvement: {greedy_var - final_var:.4f} ({(greedy_var - final_var) / greedy_var * 100:.2f}%)")

    return final_schedule, schedule_greedy, final_var, greedy_var,exe_time

def validate_schedule(project_data, schedule):
    violations = []

    # 1. ES 제약조건 검증
    for activity, start_time in schedule.items():
        if start_time < project_data[activity]['ES']:
            violations.append(f"ES Violation - Activity {activity}: "
                              f"Scheduled at {start_time} but ES is {project_data[activity]['ES']}")

    # 2. LS 제약조건 검증
    for activity, start_time in schedule.items():
        if start_time > project_data[activity]['LS']:
            violations.append(f"LS Violation - Activity {activity}: "
                              f"Scheduled at {start_time} but LS is {project_data[activity]['LS']}")

    # 3. Finish-to-Finish 선후행 관계 검증
    for activity, start_time in schedule.items():
        activity_finish = start_time + project_data[activity]['Duration']

        # 현재 액티비티가 선행작업인 경우를 찾아서 검증
        for successor, successor_data in project_data.items():
            if successor in schedule and activity in successor_data['Predecessor']:
                successor_finish = schedule[successor] + project_data[successor]['Duration']

                # 선행작업의 종료시점이 후행작업의 종료시점보다 늦은 경우
                if activity_finish > successor_finish:
                    violations.append(f"FF Precedence Violation - Activity {activity} "
                                      f"finishes at {activity_finish} but its successor {successor} "
                                      f"finishes at {successor_finish}")

    # 4. 전체 일정이 모든 액티비티를 포함하는지 검증
    scheduled_activities = set(schedule.keys())
    all_activities = set(project_data.keys())
    missing_activities = all_activities - scheduled_activities
    if missing_activities:
        violations.append(f"Missing Activities: {missing_activities}")

    # 5. 리소스 사용량이 음수가 되지 않는지 검증
    max_time = 0
    if schedule:  # 스케줄이 비어있지 않은 경우에만 계산
        max_time = max(start_time + project_data[activity]['Duration']
                       for activity, start_time in schedule.items())
        max_time = int(max_time) + 1  # 부동소수점을 정수로 변환하고 버퍼 추가
    resource_usage = [0] * max_time  # 인덱스 에러 방지를 위해 충분한 길이 확보

    for activity, start_time in schedule.items():
        duration = project_data[activity]['Duration']
        resource = project_data[activity]['Resource']
        for t in range(int(start_time), int(start_time + duration)):  # 부동소수점 에러 방지를 위해 int 변환
            resource_usage[t] += resource
            if resource_usage[t] < 0:
                violations.append(f"Negative Resource Usage at time {t}")

    return len(violations) == 0, violations


# 검증 결과를 보기 좋게 출력하는 함수
def print_validation_results(is_valid, violations):
    """
    검증 결과를 보기 좋게 출력하는 함수

    Args:
        is_valid (bool): 전체 검증 결과
        violations (list): 위반된 제약조건 목록
    """
    print("\n=== Schedule Validation Results ===")
    if is_valid:
        print("✅ All constraints are satisfied!")
    else:
        print("❌ Some constraints are violated:")
        for i, violation in enumerate(violations, 1):
            print(f"{i}. {violation}")
    print("================================\n")

###############################################
# 실제 실행: 학습
###############################################

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pointer_net = SimplePointerNet(input_dim=6, hidden_dim=32)
#     pointer_net = pointer_net.to(device)
#     print("=== TRAIN ===")
#     train_RL(pointer_net, epochs=1000, lr=1e-4, batch_size=10)
#     torch.save(pointer_net.state_dict(), 'trained_model_att.pth')


###############################################
# 실제 실행: 추론
###############################################
if __name__ == "__main__":

    # NVIDIA GPU 상태 확인을 위해 터미널에서 'nvidia-smi' 명령어 실행
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 (학습된 모델 파일 경로)
    pointer_net = SimplePointerNet(input_dim=6, hidden_dim=128)
    pointer_net.load_state_dict(torch.load('trained_model_att.pth', map_location=device))

    # 모델을 GPU로 이동
    pointer_net = pointer_net.to(device)
    print(f"모델 디바이스: {next(pointer_net.parameters()).device}")
    print(torch.__version__)

    # 추론 실행
    activity_dict = import_schedule('미래형과제 act데이터 전달_추가요소_231106_exp.xlsx')
    project_data = convert_to_project_data(activity_dict)
    # project_data = OrderedDict({
    #             "A1": {"ES": 0, "LS": 4, "Duration": 4, "Resource": 1, "Predecessor": ()},
    #             "A2": {"ES": 3, "LS": 8, "Duration": 6, "Resource": 1, "Predecessor": ("A1",)},
    #             "A3": {"ES": 5, "LS": 9, "Duration": 5, "Resource": 1, "Predecessor": ("A1",)},
    #             "A4": {"ES": 8, "LS": 15, "Duration": 5, "Resource": 1, "Predecessor": ("A2","A3",)},
    #             "A5": {"ES": 4, "LS": 7, "Duration": 4, "Resource": 1, "Predecessor": ("A1",)},
    #             "A6": {"ES": 6, "LS": 13, "Duration": 6, "Resource": 1, "Predecessor": ("A5",)},
    #             "A7": {"ES": 2, "LS": 8, "Duration": 6, "Resource": 1, "Predecessor": ("A1",)},
    #             "A8": {"ES": 6, "LS": 15, "Duration": 6, "Resource": 1, "Predecessor": ("A7",)},
    #             "A9": {"ES": 12, "LS": 20, "Duration": 4, "Resource": 1, "Predecessor": ("A8",)},
    #             "A10": {"ES": 17, "LS": 30, "Duration": 4, "Resource": 1, "Predecessor": ("A4", "A6", "A9",)},
    #
    # })
    schedule, schedule_greedy, final_var, greedy_var, *_ = inference_by_groups(project_data, pointer_net, activity_dict,
                                                                               5)
    # schedule, schedule_greedy, final_var, greedy_var, *_ = inference(project_data, pointer_net,device)

    # 결과 출력
    print("\n스케줄링 결과:")
    # print("\n[정책 네트워크]")
    # for act, start_time in schedule.items():
    #     print(f"Activity {act}: Start at {start_time}")
    print(f"분산: {final_var:.4f}")

    print("\n[그리디 휴리스틱]")
    # for act, start_time in schedule_greedy.items():
    #     print(f"Activity {act}: Start at {start_time}")
    print(f"분산: {greedy_var:.4f}")

    print(f"\n개선도: {greedy_var - final_var:.4f}")

    # 실행 후 GPU 메모리 사용량 출력 (선택 사항)
    if torch.cuda.is_available():
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    # 검증 단계 추가
    print("\n정책 네트워크 스케줄 검증:")
    is_valid_policy, violations_policy = validate_schedule(project_data, schedule)
    print_validation_results(is_valid_policy, violations_policy)

    print("\n그리디 휴리스틱 스케줄 검증:")
    is_valid_greedy, violations_greedy = validate_schedule(project_data, schedule_greedy)
    print_validation_results(is_valid_greedy, violations_greedy)


    def save_daily_workload_to_excel(schedule, project_data, filename="daily_workload.xlsx"):
        # 최대 시간 계산
        max_time = max(int(start_time + project_data[act]["Duration"])
                       for act, start_time in schedule.items())

        # 일별 리소스 사용량 계산
        daily_workload = [0] * max_time
        for act, start_time in schedule.items():
            duration = project_data[act]["Duration"]
            resource = project_data[act]["Resource"]
            for day in range(int(start_time), int(start_time + duration)):
                if day < max_time:
                    daily_workload[day] += resource

        # DataFrame 생성
        df = pd.DataFrame({
            'Day': list(range(max_time)),
            'Resource_Usage': daily_workload
        })

        # 엑셀 저장
        df.to_excel(filename, index=False)
        print(f"일별 부하 데이터가 {filename}에 저장되었습니다.")

    # 정책 네트워크와 그리디 결과 모두 저장
    save_daily_workload_to_excel(schedule, project_data, "policy_daily_workload.xlsx")
    save_daily_workload_to_excel(schedule_greedy, project_data, "greedy_daily_workload.xlsx")