import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
from collections import defaultdict

import time


from KA_sequencing import kahn_topological_sort_lh
from UT_calculate_variance import calculate_variance
from RL_resource_utilization import calculate_resource_profile
from UT_time_constraint import get_valid_start_times, adjust_ls_values_infer, adjust_ls_values
from RL_Baseline import baseline_greedy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
