import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import time
from collections import OrderedDict
from RL_network import SimplePointerNet
from RL_inference_main import inference
from RL_inference_validate import validate_schedule, print_validation_results


def main(project_data,hidden_dim=128, input_dim=6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointer_net = SimplePointerNet(input_dim=input_dim, hidden_dim=hidden_dim)

    # 학습된 모델 업로드
    model_path = 'trained_model_att.pth'
    if os.path.exists(model_path):
        pointer_net.load_state_dict(torch.load(model_path, map_location=device))
        pointer_net = pointer_net.to(device)
        pointer_net.eval()
    else:
        print(f"모델 파일 없음: {model_path}")
        return

    # 모든 프로젝트에 대한 스케줄링 수행
    print("스케줄링 추론 중...")
    start_time = time.time()
    result = inference(project_data, pointer_net, device)
    schedule, schedule_greedy, final_var, greedy_var, *_ = result
    end_time = time.time()
    print(schedule)

    print("\n스케줄링 결과:")
    print(f"분산: {final_var:.4f}")
    print("\n[그리디 휴리스틱]")
    print(f"분산: {greedy_var:.4f}")
    print(f"\n개선도: {greedy_var - final_var:.4f}")
    print(f"\n계산시간: {end_time - start_time:.4f}")

    # 제약 만족 여부 검증
    is_valid, violations = validate_schedule(project_data, schedule)
    print_validation_results(is_valid, violations)

if __name__ == "__main__":
    # 프로젝트 데이터 input: PSPLIB 형식(현재 데이터: J30_1)
    project_data = OrderedDict({
        "A1": {"ES": 0, "LS": 0, "Duration": 0, "Resource": 0, "Predecessor": ()},
        "A2": {"ES": 0, "LS": 21, "Duration": 8, "Resource": 4, "Predecessor": ("A1",)},
        "A3": {"ES": 0, "LS": 24, "Duration": 4, "Resource": 10, "Predecessor": ("A1",)},
        "A4": {"ES": 0, "LS": 18, "Duration": 6, "Resource": 3, "Predecessor": ("A1",)},
        "A5": {"ES": 6, "LS": 24, "Duration": 3, "Resource": 3, "Predecessor": ("A4",)},
        "A6": {"ES": 8, "LS": 29, "Duration": 8, "Resource": 8, "Predecessor": ("A2",)},
        "A7": {"ES": 4, "LS": 28, "Duration": 5, "Resource": 4, "Predecessor": ("A3",)},
        "A8": {"ES": 4, "LS": 28, "Duration": 9, "Resource": 1, "Predecessor": ("A3",)},
        "A9": {"ES": 6, "LS": 24, "Duration": 2, "Resource": 6, "Predecessor": ("A4",)},
        "A10": {"ES": 6, "LS": 24, "Duration": 7, "Resource": 1, "Predecessor": ("A4",)},
        "A11": {"ES": 8, "LS": 29, "Duration": 9, "Resource": 5, "Predecessor": ("A2",)},
        "A12": {"ES": 13, "LS": 37, "Duration": 2, "Resource": 7, "Predecessor": ("A8",)},
        "A13": {"ES": 4, "LS": 28, "Duration": 6, "Resource": 4, "Predecessor": ("A3",)},
        "A14": {"ES": 15, "LS": 39, "Duration": 3, "Resource": 8, "Predecessor": ("A9", "A12")},
        "A15": {"ES": 8, "LS": 29, "Duration": 9, "Resource": 3, "Predecessor": ("A2",)},
        "A16": {"ES": 13, "LS": 31, "Duration": 10, "Resource": 5, "Predecessor": ("A10",)},
        "A17": {"ES": 18, "LS": 42, "Duration": 6, "Resource": 8, "Predecessor": ("A13", "A14")},
        "A18": {"ES": 10, "LS": 34, "Duration": 5, "Resource": 7, "Predecessor": ("A13",)},
        "A19": {"ES": 13, "LS": 37, "Duration": 3, "Resource": 1, "Predecessor": ("A8",)},
        "A20": {"ES": 17, "LS": 38, "Duration": 7, "Resource": 10, "Predecessor": ("A5", "A11", "A18")},
        "A21": {"ES": 23, "LS": 41, "Duration": 2, "Resource": 6, "Predecessor": ("A16",)},
        "A22": {"ES": 24, "LS": 48, "Duration": 7, "Resource": 2, "Predecessor": ("A16", "A17", "A18")},
        "A23": {"ES": 31, "LS": 55, "Duration": 2, "Resource": 5, "Predecessor": ("A20", "A22")},
        "A24": {"ES": 33, "LS": 57, "Duration": 3, "Resource": 9, "Predecessor": ("A19", "A23")},
        "A25": {"ES": 24, "LS": 45, "Duration": 3, "Resource": 4, "Predecessor": ("A10", "A15", "A20")},
        "A26": {"ES": 17, "LS": 38, "Duration": 7, "Resource": 4, "Predecessor": ("A11",)},
        "A27": {"ES": 13, "LS": 37, "Duration": 8, "Resource": 7, "Predecessor": ("A7", "A8")},
        "A28": {"ES": 25, "LS": 43, "Duration": 3, "Resource": 8, "Predecessor": ("A21", "A27")},
        "A29": {"ES": 16, "LS": 40, "Duration": 7, "Resource": 7, "Predecessor": ("A19",)},
        "A30": {"ES": 36, "LS": 60, "Duration": 2, "Resource": 7, "Predecessor": ("A6", "A24", "A25")},
        "A31": {"ES": 28, "LS": 46, "Duration": 2, "Resource": 2, "Predecessor": ("A26", "A28")},
        "A32": {"ES": 38, "LS": 62, "Duration": 0, "Resource": 0, "Predecessor": ("A29", "A30", "A31")}})

    main(project_data,hidden_dim=32,input_dim=6)