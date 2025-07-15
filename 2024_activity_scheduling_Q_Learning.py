import numpy as np
np.bool = np.bool_
from docplex.cp.model import *
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt

# Q-Learning Algorithm
class QLScheduler:
    def __init__(self, project_data: Dict):
        self.project_data = project_data
        self.priority_sequence = []
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.best_schedule = None
        self.best_variance = float('inf')
        self.variance_history = []


    # 결정 순서 알고리즘: Kahn's Algorithm #
    def determine_priorities_with_kahn(self):
        start_time = time.time()
        activities = list(self.project_data.keys())
        graph = defaultdict(set)

        # 그래프 구축
        for act, info in self.project_data.items():
            predecessors = info['Predecessor']
            if isinstance(predecessors, tuple):
                for pred in predecessors:
                    if pred != 0:
                        graph[pred].add(act)

        # 진입 차수(in-degree) 계산
        in_degree = defaultdict(int)
        for node in activities:
            in_degree[node] = 0
        for successors in graph.values():
            for node in successors:
                in_degree[node] += 1

        # 진입 차수가 0인 노드들로 시작
        queue = deque([node for node in activities if in_degree[node] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            # 후행 작업들의 진입 차수 감소
            for successor in graph[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        # 모든 노드가 처리되었는지 확인
        if len(order) != len(activities):
            print("Error: Graph contains a cycle")
            return False
        total_time = time.time()-start_time
        print(f"총 실행 시간: {total_time:.6f}초")
        self.priority_sequence = order
        return True

    # Latest start date 조정 함수 #
    def adjust_ls_values(self):
        for activity in reversed(self.priority_sequence):
            info = self.project_data[activity]
            duration = info['Duration']
            current_ls = info['LS']

            # 현재 activity를 predecessor로 가지는 후행 작업들 찾기
            successors = []
            for act, data in self.project_data.items():
                if isinstance(data['Predecessor'], tuple):
                    if activity in data['Predecessor']:
                        successors.append(act)

            # Finish-to-Finish 관계에 맞춘 구현
            if successors:
                # 후행 작업들의 종료시점(LS + duration) 중 가장 이른 시점
                min_successor_finish = min(self.project_data[succ]['LS'] + self.project_data[succ]['Duration']
                                           for succ in successors)
                # 현재 작업의 종료시점도 그보다 빨라야 함
                new_ls = min(current_ls, min_successor_finish - duration)
                self.project_data[activity]['LS'] = new_ls
                print(f"Adjusted LS for Activity {activity}: {current_ls} -> {new_ls}")

    # 분산 계산 함수 #
    def calculate_variance(self, schedule: Dict) -> float:
        if not schedule:
            return float('inf')

        # 모든 일정: LS + Duration
        max_time = max(int(data['LS']) + int(data['Duration'])
                       for data in self.project_data.values())

        # 자원 사용량 배열 초기화
        resource_usage = np.zeros(max_time)

        # 각 활동 별 자원 사용량 누적
        for activity, start_time in schedule.items():
            dur = self.project_data[activity]['Duration']
            res = self.project_data[activity]['Resource']
            for t in range(int(start_time), int(start_time + dur)):
                if t < max_time:
                    resource_usage[t] += res

        # 평균 및 분산 계산
        avg = np.mean(resource_usage)
        var = np.mean((resource_usage - avg) ** 2)

        return var

    # Reward 로 활용할 분산 변화량 계산 함수 #
    def calculate_contribution(self, schedule: Dict, activity: int) -> float:
        # 전체 분산 계산
        total_variance = self.calculate_variance(schedule)

        # 해당 액티비티를 제외한 분산 계산
        schedule_without_activity = {k: v for k, v in schedule.items() if k != activity}
        variance_without_activity = self.calculate_variance(schedule_without_activity)

        # 기여도 계산 (음수면 분산을 증가시킨 것, 양수면 감소시킨 것)
        if variance_without_activity == 0:
            return 0
        return (variance_without_activity - total_variance) / total_variance


    ## 가능 시작 시간 조정: finish to finish ##
    def get_valid_start_times(self, activity: int, current_schedule: Dict) -> List[int]:
        """주어진 액티비티의 가능한 시작 시간 목록 반환 (Finish to Finish 관계)"""
        info = self.project_data[activity]
        base_es = info['ES']
        ls = info['LS']
        duration = info['Duration']

        # 현재 작업의 가장 빠른 시작 시간 = base_es
        earliest_start = base_es

        # 선행 작업들의 종료 시간 확인
        for pred in info['Predecessor']:
            if pred != 0 and pred in current_schedule:
                pred_finish = current_schedule[pred] + self.project_data[pred]['Duration']
                earliest_start = max(earliest_start, pred_finish - duration)

        # 정수로 변환한 후 가능한 시작 시간 범위 반환(range에 의해 79를 일정에 포함하기 위해 int(ls)+1로 구현)
        return list(range(int(earliest_start), int(ls) + 1))

    # 입실론 그디리 정책을 활용한 시작 시간 선택 #
    def epsilon_greedy_action(self, state: int, valid_times: List[int], epsilon: float) -> int:
        """입실론-그리디 Policy 공식
        π(a|s) = ε/m + 1 - ε  if a* = argmax(Q(s,a))
                 ε/m          otherwise
        """
        m = len(valid_times)  # 가능한 액션(시작 시간)의 수

        # Q값들 중 최대값을 가진 액션 찾기
        q_values = {time: self.q_values[state][time] for time in valid_times}
        max_value = max(q_values.values())
        best_times = [t for t, v in q_values.items() if v == max_value]
        best_action = best_times[0]  # 최적의 액션 중 하나 선택

        # 각 액션에 대한 선택 확률 계산
        probabilities = []
        for time in valid_times:
            if time == best_action:
                prob = epsilon / m + (1 - epsilon)  # 최적 액션의 확률
            else:
                prob = epsilon / m  # 그 외 액션의 확률
            probabilities.append(prob)

        # 확률 정규화
        probabilities = np.array(probabilities) / sum(probabilities)

        # 확률에 따라 액션 선택
        return np.random.choice(valid_times, p=probabilities)

    # 학습(Q-Learning) 함수 #
    def learn_schedule(self, episodes: int = 5000, start_epsilon: float = 0.3, min_epsilon: float = 0.01,
                       gamma: float = 0.9):

        # epsilon decay rate 계산
        decay_rate = -np.log(min_epsilon / start_epsilon) / episodes

        # 에피소드 학습
        for episode in range(episodes):
            current_schedule = {}

            # epsilon 값을 시간에 따라 감소
            current_epsilon = max(min_epsilon, start_epsilon * np.exp(-decay_rate * episode))

            # 우선순위 순서대로 액티비티 시작 시간 스케줄링
            for activity in self.priority_sequence:
                valid_times = self.get_valid_start_times(activity, current_schedule)
                # 불가능한 액티비티의 경우
                if not valid_times:
                    print("Infeasible!")
                    continue
                else:
                    start_time = self.epsilon_greedy_action(activity, valid_times, current_epsilon)
                    current_schedule[activity] = start_time

            # 일정에 맞추어 Q-value 계산
            if current_schedule:
                # 현재 스케줄의 리소스 사용량 분산 계산 (목적함수)
                current_variance = self.calculate_variance(current_schedule)
                # 학습 진행 상황 추적을 위해 분산값 기록
                self.variance_history.append(current_variance)

                # 지금까지 발견한 최고 성능보다 좋으면 최적해 갱신
                if current_variance < self.best_variance:
                    self.best_variance = current_variance
                    self.best_schedule = current_schedule.copy()  # 최적 일정 저장

                # 우선순위 순서대로 모든 액티비티에 대해 Q-값 업데이트
                for activity in self.priority_sequence:
                    # 결정된 액티비티 대상 Q-Table 계산
                    if activity in current_schedule:
                        # Reward: 해당 액티비티가 전체 분산에 미친 기여도 계산
                        contribution = self.calculate_contribution(current_schedule, activity)
                        # 이번 에피소드에서 선택한 시작 시간
                        start_time = current_schedule[activity]
                        # 업데이트 전 현재 Q-값
                        current_q = self.q_values[activity][start_time]

                        # 다음 액티비티의 max Q값 계산
                        # 우선순위 순서에 따른 다음 순서의 액티비티 찾음
                        next_activity_idx = self.priority_sequence.index(activity) + 1
                        # 다음 액티비티가 존재하는 경우
                        if next_activity_idx < len(self.priority_sequence):
                            # 다음 순서의 액티비티 식별
                            next_activity = self.priority_sequence[next_activity_idx]
                            # 다음 액티비티가 시작할 수 있는 모든 가능한 시간들 계산
                            next_valid_times = self.get_valid_start_times(next_activity, current_schedule)
                            # 다음 액티비티의 가능한 시작 시간이 있는 경우
                            if next_valid_times:
                                # 다음 상태에서 가능한 최대 Q-값 계산
                                max_next_q = max(self.q_values[next_activity][t] for t in next_valid_times)
                                # Q-Learning 공식: Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
                                self.q_values[activity][start_time] = current_q + 0.1 * (
                                            contribution + gamma * max_next_q - current_q)
                            else:
                                # 다음 액티비티의 가능한 시간이 없는 경우, 즉시 보상만 적용
                                self.q_values[activity][start_time] = current_q + 0.1 * (contribution - current_q)
                        else:
                            # 마지막 액티비티인 경우 (터미널 상태), 미래 보상 없이 즉시 보상만 고려
                            self.q_values[activity][start_time] = current_q + 0.1 * (contribution - current_q)

            if episode % 1 == 0:
                print(
                    f"Episode {episode}: Current Schedule Size = {len(current_schedule)}/{len(self.priority_sequence)}, "
                    f"Best Variance = {self.best_variance:.2f}, "
                    f"Epsilon = {current_epsilon:.4f}")

    # 학습 곡선 출력 함수 #
    def plot_convergence_from_start(self):
        plt.figure(figsize=(15, 8))

        max_variance = max(self.variance_history)
        min_variance = min(self.variance_history)

        plt.plot(self.variance_history,
                 'b-',
                 linewidth=1,
                 label='Variance over Episodes')

        plt.title('Learning Progress: Resource Usage Variance')
        plt.xlabel('Episode')
        plt.ylabel('Variance')

        # y축 범위를 실제 최대값부터 최소값까지로 설정
        plt.ylim(min_variance * 0.9, max_variance * 1.1)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 학습 실행 함수 #
    def run(self):
        if self.determine_priorities_with_kahn():
            print("우선순위 결정 완료:")
            for i, act in enumerate(self.priority_sequence, 1):
                print(f"{i}순위: Activity {act}")

            print("\nAdjusting LS values based on precedence relationships...")
            self.adjust_ls_values()

            self.learn_schedule()
            self.plot_convergence_from_start()

            print("\n최종 스케줄:")
            for act in sorted(self.best_schedule.keys()):
                print(f"Activity {act}: 시작 시간 = {self.best_schedule[act]}, "
                      f"자원 사용량 = {self.project_data[act]['Resource']}")

            print(f"\n최종 분산: {self.best_variance:.2f}")
            return self.best_schedule
        else:
            print("우선순위를 결정할 수 없습니다.")
            return None


# Gantt chart 출력 함수 #
def draw_gantt_chart(project_data, schedule):
    plt.figure(figsize=(15, 8))
    yticks = []
    ytick_labels = []

    for i, (activity, start_time) in enumerate(sorted(schedule.items(), key=lambda x: x[1])):
        duration = project_data[activity]['Duration']
        plt.barh(i, duration, left=start_time, color='skyblue', edgecolor='black')
        yticks.append(i)
        ytick_labels.append(f'Activity {activity}')

        # 시작 및 종료 시간 표시
        plt.text(start_time + duration / 2, i, f'{start_time}-{start_time + duration}', va='center', ha='center')

    plt.yticks(yticks, ytick_labels)
    plt.xlabel('Time')
    plt.title('Gantt Chart for Project Schedule')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 실행 데이터 #
project_data={
                "A1": {"ES": 0, "LS": 79, "Duration": 8, "Resource": 9.4, "Predecessor": ('B1','B2',)},
                "B1": {"ES": 0, "LS": 79, "Duration": 6, "Resource": 6.4, "Predecessor": ()},
                "C1": {"ES": 0, "LS": 79, "Duration": 9, "Resource": 9.4, "Predecessor": ()},
                "A2": {"ES": 0, "LS": 79, "Duration": 8, "Resource": 7.8, "Predecessor": ()},
                "B2": {"ES": 0, "LS": 79, "Duration": 6, "Resource": 5.4, "Predecessor": ()},

    }


# 실행 코드 #
scheduler = QLScheduler(project_data)
final_schedule = scheduler.run()
solution = {}
for activity in sorted(final_schedule.keys()):
    solution[activity] = final_schedule[activity]
print("Validation input:", solution)
draw_gantt_chart(project_data, final_schedule)
