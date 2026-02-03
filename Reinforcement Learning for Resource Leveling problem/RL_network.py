import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePointerNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_heads=8, resource_dim=50):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 리소스 프로필을 위한 1D CNN 또는 선형 레이어
        self.resource_embed = nn.Sequential(
            nn.Linear(resource_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # 통합된 특성 차원 처리
        self.combined_fc = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

        # Multi head attention 활용 #
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.context = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, resource_profiles):
        batch_size, num_options = x.size(0), x.size(1)

        # 기본 특성 임베딩
        enc_h = self.embedding(x)  # (배치, 시간옵션, hidden_dim)

        # 리소스 프로필 처리 - 전체 패턴 활용 (평균 대신!)
        resource_h = self.resource_embed(resource_profiles)  # (배치, 시간옵션, hidden_dim//2)

        # 활동 특성과 리소스 특성 결합
        combined_h = torch.cat([enc_h, resource_h], dim=2)
        combined_h = self.combined_fc(combined_h)

        # 단일 어텐션으로 모든 옵션 간의 관계 학습
        ### 어텐션 활용 ###
        attn_out, _ = self.attn(combined_h, combined_h, combined_h)
        enc_h = combined_h + attn_out
        enc_h = self.fc(enc_h)

        # 최종 점수 계산
        query = self.context.expand(batch_size, -1, -1)
        logits = torch.bmm(query, enc_h.transpose(1, 2)).squeeze(1)

        return logits