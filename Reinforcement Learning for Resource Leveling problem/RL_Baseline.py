from UT_calculate_variance import calculate_variance

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