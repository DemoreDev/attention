"""
Testes essenciais: cobrem bugs silenciosos, que não geram erro mas estragariam o treino.

Rodar (na raiz do projeto):
    pytest            # resumo
    pytest -v         # um resultado por teste
"""

import pytest
import torch

from dataset import IGNORE_INDEX, build_sequence
from model import DecoderOnlyTransformer, LayerNorm, ModelConfig
from tokenizer import CharTokenizer

SOURCE = "5 de Março de 1935"
TARGET = "1935-03-05"


@pytest.fixture
def tokenizer() -> CharTokenizer:
    """Tokenizer pequeno, montado só com os textos do exemplo (não depende de data/)."""
    return CharTokenizer.from_texts([SOURCE, TARGET])


# 1. Tokenizer ---------------------------------------------------------------

def test_tokenizer_ida_e_volta_e_especiais_indivisiveis(tokenizer):
    # Texto → IDs → texto devolve exatamente o original (um ID por caractere).
    ids = tokenizer.text_to_ids(SOURCE)
    assert len(ids) == len(SOURCE)
    assert tokenizer.ids_to_text(ids) == SOURCE

    # Tokens especiais são UM token cada, inseridos pelo ID...
    assert tokenizer.ids_to_text([tokenizer.bos_id, tokenizer.eos_id]) == "<bos><eos>"
    # ...e nunca saem do texto: "<bos>" escrito como texto não é reconhecido.
    with pytest.raises(ValueError):
        tokenizer.text_to_ids("<bos>")


# 2. Dataset: shift e máscara da loss ----------------------------------------

def test_shift_e_labels_ignorados(tokenizer):
    input_ids, labels = build_sequence(SOURCE, TARGET, tokenizer)
    n_source = len(SOURCE)

    # Entrada: <bos> + source + <sep> + target (sem o <eos>).
    assert input_ids[0] == tokenizer.bos_id
    assert input_ids[n_source + 1] == tokenizer.sep_id
    assert tokenizer.ids_to_text(input_ids[n_source + 2:]) == TARGET
    assert len(input_ids) == len(labels)

    # Labels: tudo até o <sep> (inclusive) ignorado; depois, exatamente target + <eos>.
    assert labels[:n_source + 1] == [IGNORE_INDEX] * (n_source + 1)
    assert labels[n_source + 1:] == tokenizer.text_to_ids(TARGET) + [tokenizer.eos_id]

    # Alinhamento do shift: a posição do <sep> prevê o 1º caractere ISO.
    assert input_ids[n_source + 1] == tokenizer.sep_id
    assert labels[n_source + 1] == tokenizer.text_to_ids(TARGET[0])[0]


# 3. Máscara causal (e dispensa da máscara de padding) -----------------------

def test_mascara_causal_ignora_posicoes_futuras():
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(ModelConfig(vocab_size=20, d_model=32, n_layers=2, n_heads=4, d_ff=64))
    model.eval()  # sem dropout, para a comparação ser determinística

    original = torch.randint(4, 20, (1, 12))
    # Substitui as posições a partir da 7 por <pad> (id 0): é o que o padding à direita faz.
    padded = original.clone()
    padded[0, 7:] = 0

    with torch.no_grad():
        out_original, out_padded = model(original), model(padded)

    # As posições 0..6 não podem enxergar o futuro: saídas idênticas.
    assert torch.allclose(out_original[0, :7], out_padded[0, :7], atol=1e-6)
    # Sanidade: a partir da posição 7 as entradas mudaram, então as saídas também.
    assert not torch.allclose(out_original[0, 7:], out_padded[0, 7:])


# 4. LayerNorm à mão ---------------------------------------------------------

def test_layernorm_igual_a_referencia_do_pytorch():
    torch.manual_seed(0)
    x = torch.randn(4, 10, 32) * 3 + 1  # média e escala diferentes de 0 e 1

    ours = LayerNorm(32)
    reference = torch.nn.LayerNorm(32, eps=1e-5)
    # Mesmos γ e β (não triviais) nas duas implementações.
    with torch.no_grad():
        ours.gamma.copy_(torch.randn(32))
        ours.beta.copy_(torch.randn(32))
        reference.weight.copy_(ours.gamma)
        reference.bias.copy_(ours.beta)

    assert torch.allclose(ours(x), reference(x), atol=1e-5)
