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
        max_time = int(max_time) + 1
    resource_usage = [0] * max_time

    for activity, start_time in schedule.items():
        duration = project_data[activity]['Duration']
        resource = project_data[activity]['Resource']
        for t in range(int(start_time), int(start_time + duration)):  # 부동소수점 에러 방지를 위해 int 변환
            resource_usage[t] += resource
            if resource_usage[t] < 0:
                violations.append(f"Negative Resource Usage at time {t}")

    return len(violations) == 0, violations

def print_validation_results(is_valid, violations):

    print("\n=== Schedule Validation Results ===")
    if is_valid:
        print("✅ All constraints are satisfied!")
    else:
        print("❌ Some constraints are violated:")
        for i, violation in enumerate(violations, 1):
            print(f"{i}. {violation}")
    print("================================\n")