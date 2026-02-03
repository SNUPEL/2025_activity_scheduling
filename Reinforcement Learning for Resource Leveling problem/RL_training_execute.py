import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from RL_network import SimplePointerNet
from RL_training_main import train_RL

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointer_net = SimplePointerNet(input_dim=6, hidden_dim=32)
    pointer_net = pointer_net.to(device)
    print("=== TRAIN ===")
    train_RL(pointer_net, device, epochs=200, lr=1e-4, batch_size=10)
    torch.save(pointer_net.state_dict(), 'trained_model_att.pth')