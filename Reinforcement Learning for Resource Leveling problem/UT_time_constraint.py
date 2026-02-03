from collections import defaultdict

def adjust_ls_values(project_data, priority_sequence):
    """Finish-to-Finish 관계로 LS를 재조정 (학습용)"""
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