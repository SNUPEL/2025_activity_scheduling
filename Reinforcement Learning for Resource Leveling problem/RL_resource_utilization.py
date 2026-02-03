import numpy as np

def calculate_resource_profile(project_data, current_schedule, act, vtimes, fixed_length=50):

    if not vtimes:
        return np.array([])

    duration = project_data[act]["Duration"]
    resource = project_data[act]["Resource"]

    # 관심 구간 계산: 가능한 최소 시작시간 ~ 최대 종료시간
    min_start = min(vtimes)
    max_end = max(vtimes) + duration
    window_size = int(max_end - min_start)

    profiles = []

    for start_time in vtimes:
        # 1. 해당 옵션(start_time)에서의 리소스 윈도우 생성
        profile = np.zeros(window_size)

        # 2. 이미 스케줄링이 완료된 작업들의 리소스 사용량 반영
        for scheduled_act, st in current_schedule.items():
            act_dur = project_data[scheduled_act]["Duration"]
            act_res = project_data[scheduled_act]["Resource"]

            # 현재 검토 중인 윈도우(min_start ~ max_end)와 겹치는 부분만 계산
            overlap_start = max(int(st), min_start)
            overlap_end = min(int(st + act_dur), max_end)

            for t in range(overlap_start, overlap_end):
                if min_start <= t < max_end:
                    profile[t - min_start] += act_res

        # 3. 현재 검토 중인 작업(act)이 이 시간(start_time)에 배치될 경우의 리소스 추가
        act_start_in_window = int(start_time - min_start)
        act_end_in_window = int(start_time + duration - min_start)

        for t in range(act_start_in_window, act_end_in_window):
            if 0 <= t < window_size:
                profile[t] += resource

        # 4. 신경망 입력을 위해 고정 길이로 패딩(Padding) 또는 트렁케이션(Truncation)
        if len(profile) < fixed_length:
            # 길이가 짧으면 뒤를 0으로 채움
            padded_profile = np.pad(profile, (0, fixed_length - len(profile)), 'constant')
        else:
            # 길이가 길면 앞부분만 사용
            padded_profile = profile[:fixed_length]

        profiles.append(padded_profile)

    return np.array(profiles)