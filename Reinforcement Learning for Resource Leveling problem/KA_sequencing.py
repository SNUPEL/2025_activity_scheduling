from collections import defaultdict, deque

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