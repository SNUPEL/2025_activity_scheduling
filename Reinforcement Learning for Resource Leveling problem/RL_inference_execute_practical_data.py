import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time

from RL_network import SimplePointerNet
from UT_data_preprocessing import import_schedule, convert_to_project_data
from RL_inference_main import inference_by_groups
from RL_inference_validate import validate_schedule, print_validation_results


def main(beam_width=5, hidden_dim=128, input_dim=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pointer_net = SimplePointerNet(input_dim=input_dim, hidden_dim=hidden_dim)

    model_path = 'trained_model_att.pth'
    if os.path.exists(model_path):
        # 가중치 로드
        pointer_net.load_state_dict(torch.load(model_path, map_location=device))
        pointer_net = pointer_net.to(device)
        pointer_net.eval()
    else:
        print(f"모델 파일 없음: {model_path}")
        return

    # Exel data 로드
    data_file = 'Practical_data_HD.xlsx'
    activity_dict = import_schedule(data_file)
    project_data = convert_to_project_data(activity_dict)

    print("스케줄링 추론 중...")
    start_time = time.time()

    # 빔 너비를 여기서 전달합니다.
    result = inference_by_groups(project_data, pointer_net, activity_dict, beam_width=beam_width)

    schedule, schedule_greedy, final_var, greedy_var, *_ = result

    end_time = time.time()

    # 6. 결과 출력
    print("\n" + "=" * 40)
    print(f"결과 요약 (Beam={beam_width})")
    print(f"RL 분산: {final_var:.4f}")
    print(f"Greedy 분산: {greedy_var:.4f}")
    print(f"실행 시간: {end_time - start_time:.2f}s")
    print("=" * 40)

    # 제약 만족 여부 검증
    is_valid, violations = validate_schedule(project_data, schedule)
    print_validation_results(is_valid, violations)


if __name__ == "__main__":
    # 여기서 원하는 값을 설정해서 실행하면 됩니다!
    main(
        beam_width=100,  # 빔 수 조절
        hidden_dim=32,  # 모델 차원 조절
        input_dim=6  # 입력 피처 수 조절
    )