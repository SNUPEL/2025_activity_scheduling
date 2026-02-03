import pandas as pd
from collections import OrderedDict

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
    df_SHOP = df[(df['SHOP'] == 'SP4')]
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