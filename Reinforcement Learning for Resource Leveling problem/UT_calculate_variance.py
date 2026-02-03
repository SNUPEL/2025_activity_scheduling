import numpy as np

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