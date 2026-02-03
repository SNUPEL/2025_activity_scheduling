import numpy as np
import random
from collections import OrderedDict

def generate_random_activities(n=30, mu=1.6, sigma=0.5):
    activities = OrderedDict()


    for i in range(n):
        name = chr(65 + (i % 26)) + str(i // 26 + 1)
        # 학습 데이터 기준: PSPLIB 과 동일하게 es, ls 고정
        es = 0
        ls = 80

        # 로그정규분포를 사용한 작업 시간 생성
        duration = np.random.lognormal(mean=mu, sigma=sigma)
        duration = max(1, round(duration))

        # 자원 사용량은 유니폼 분포 활용
        resource = random.uniform(5.0, 11.0)

        # 선,후행 관계 생성
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