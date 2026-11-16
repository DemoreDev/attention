"""
Transformer decoder-only 

Implementado sem usar funções prontas do pytorch: LayerNorm, encoding posicional senoidal, 
multi-head self-attention com máscara causal, FFN, Add & Norm (Post-LN), 
bloco do decoder e modelo completo.
Do PyTorch: nn.Linear, nn.Embedding e nn.Dropout.

Convenção de nomes dos tensores (dimensões):
    B = batch, T = comprimento da sequência, D = d_model, H = n_heads, d_k = D / H

Uso (mostra o número de parâmetros das 3 configurações e roda um forward de teste):
    python src/model.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Hiperparâmetros da arquitetura (os nomes seguem o paper)."""
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int = 80
    dropout: float = 0.1


class LayerNorm(nn.Module):
    """LayerNorm à mão: normaliza cada vetor (cada token) pela própria média e variância,
       usando operações de tensor do PyTorch

        LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # γ começa em 1 e β em 0: no início do treino a camada é só a normalização.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        mean = x.mean(dim=-1, keepdim=True)
        
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def sinusoidal_positional_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Matriz PE (max_seq_len, d_model) do paper. Fixa, sem parâmetros treináveis.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)                 # 0, 2, 4, ..., d_model - 2
    # 1 / 10000^(2i / d_model), uma frequência por par de dimensões (das rápidas às lentas).
    frequencies = 1.0 / (10000.0 ** (two_i / d_model))                       # (d_model / 2,)

    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(positions * frequencies)  # dimensões pares
    pe[:, 1::2] = torch.cos(positions * frequencies)  # dimensões ímpares
    return pe


def causal_mask(seq_len: int) -> torch.Tensor:
    """Máscara (T, T) com True nas posições FUTURAS (j > i), que devem ser bloqueadas.

        F T T T
        F F T T
        F F F T
        F F F F
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention

    O paper descreve H matrizes W_i^Q de tamanho (D, d_k) por camada (uma por head).
    Aqui as H matrizes ficam lado a lado em uma única nn.Linear(D, D): o resultado
    é o mesmo, mas calcula todas as heads de uma vez (mais eficiente). Depois a
    saída é "fatiada" em H pedaços de tamanho d_k, um por head.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projeções Q, K, V e a projeção final W^O. Sem bias, como nas fórmulas do paper.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, d_k): separa as heads para calcular a atenção de cada uma em paralelo."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Cada token gera sua Query, Key e Value, em cada head.
        q = self.split_heads(self.w_q(x))  # (B, H, T, d_k)
        k = self.split_heads(self.w_k(x))  # (B, H, T, d_k)
        v = self.split_heads(self.w_v(x))  # (B, H, T, d_k)

        # 2. Compatibilidade entre queries e keys: C = Q·Kᵀ / √d_k  → (B, H, T, T)
        #    A divisão por √d_k evita scores grandes demais, que saturariam a softmax.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # 3. Máscara causal: posições futuras recebem -inf, que vira peso 0 após a softmax.
        #    A mesma máscara (T, T) vale para todos os exemplos do batch e todas as heads.
        scores = scores.masked_fill(mask, float("-inf"))

        # 4. Softmax em cada linha: pesos de atenção de cada query (cada linha soma 1).
        weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # 5. Média ponderada dos values: O = S·V  → (B, H, T, d_k)
        out = weights @ v

        # 6. Concatena as heads de volta (B, T, D) e aplica W^O, que combina as heads.
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class FeedForward(nn.Module):
    """FFN aplicada a cada token separadamente: max(0, z·W1 + b1)·W2 + b2."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # expande D → d_ff (= 4·D)
        self.linear2 = nn.Linear(d_ff, d_model)  # volta d_ff → D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class DecoderBlock(nn.Module):
    """Uma camada do decoder-only, com Add & Norm em Post-LN (como no paper):

        Z = LayerNorm(X + Dropout(MaskedMHA(X)))
        H = LayerNorm(Z + Dropout(FFN(Z)))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config.d_model, config.n_heads)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = LayerNorm(config.d_model)

        self.ffn = FeedForward(config.d_model, config.d_ff)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Primeira subcamada + Add & Norm (dropout → residual → LayerNorm).
        z = self.norm1(x + self.dropout1(self.attention(x, mask)))
        # Segunda subcamada + Add & Norm.
        return self.norm2(z + self.dropout2(self.ffn(z)))


class DecoderOnlyTransformer(nn.Module):
    """Modelo completo: embeddings + posição → N blocos → logits (pesos compartilhados com E)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Matriz de embeddings E (|V|, D). Inicializada com desvio D^(-1/2): depois da
        # multiplicação por √D, cada elemento tem desvio ≈ 1, na mesma escala do PE
        # (valores em [-1, 1]). Com o padrão do PyTorch (desvio 1), os embeddings
        # abafariam a informação de posição.
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.d_model ** -0.5)

        # PE fixo: registrado como buffer (acompanha o modelo, mas não é treinado).
        # persistent=False: não é salvo no checkpoint, pois é recalculado no construtor.
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(config.max_seq_len, config.d_model),
            persistent=False,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids (B, T) → logits (B, T, |V|). logits[:, t] são os scores do token t+1."""
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"sequência de {T} tokens excede max_seq_len={self.config.max_seq_len}")

        # Embedding (lookup) com escala √D, soma da posição e dropout.
        x = self.embedding(input_ids) * math.sqrt(self.config.d_model)  # (B, T, D)
        x = x + self.positional_encoding[:T]                              # PE (T, D) somado a cada exemplo
        x = self.dropout(x)

        # A máscara causal é criada uma vez por forward e usada por todas as camadas.
        mask = causal_mask(T).to(input_ids.device)
        for block in self.blocks:
            x = block(x, mask)

        # Weight tying: a projeção de saída é a própria E transposta (sem bias),
        # logits = H · Eᵀ. Cada logit é o produto escalar entre a representação da
        # posição e o embedding de um token candidato.
        return x @ self.embedding.weight.T

    def count_parameters(self) -> int:
        """Número de parâmetros treináveis. E é contada uma vez só (compartilhada com a saída)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main() -> None:
    # Configurações dos experimentos (as definitivas ficarão em configs/expN.json).
    vocab_size = 44
    configs = {
        "exp1": ModelConfig(vocab_size, d_model=32, n_layers=2, n_heads=4, d_ff=128),
        "exp2": ModelConfig(vocab_size, d_model=64, n_layers=3, n_heads=4, d_ff=256),
        "exp3": ModelConfig(vocab_size, d_model=128, n_layers=4, n_heads=4, d_ff=512),
    }
    for name, config in configs.items():
        model = DecoderOnlyTransformer(config)
        print(f"{name}: {model.count_parameters():>9,} parâmetros")

    # Forward de teste: batch de 2 sequências com 35 tokens aleatórios.
    model = DecoderOnlyTransformer(configs["exp1"])
    input_ids = torch.randint(0, vocab_size, (2, 35))
    logits = model(input_ids)
    print(f"\nentrada {tuple(input_ids.shape)} → logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
