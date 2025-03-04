import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import koi
import random
# from koi.ctc import Log, semiring
# from koi.ctc import logZ_cu, logZ_cu_sparse
# from utils.utils_func import get_logger
#
# LOGGER = get_logger(__name__)
from models.conformer.encoder import ConformerBlock

from mamba_ssm import Mamba, Mamba2
from models.linformer import Linformer

try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.modules.mlp import GatedMlp
    from flash_attn.ops.triton.layer_norm import RMSNorm
except ImportError:
    print(
        "please install flash-attn to use the transformer module: "
        "`pip install flash-attn --no-build-isolation`"
    )


state_len = 5
n_base = 4
alphabet = ['N', 'A', 'C', 'G', 'T']

def init(seed, device, deterministic=True):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu": return
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)
    assert(torch.cuda.is_available())

idx = torch.cat([
    torch.arange(n_base**(state_len))[:, None],
    torch.arange(
        n_base**(state_len)
    ).repeat_interleave(n_base).reshape(n_base, -1).T
], dim=1).to(torch.int32)
def prepare_ctc_scores(scores, targets):
    # convert from CTC targets (with blank = 0) to zero indexed
    targets = torch.clamp(targets - 1, 0)

    T, N, C = scores.shape
    scores = scores.to(torch.float32)
    n = targets.size(1) - (state_len - 1)
    stay_indices = sum(
        targets[:, i:n + i] * n_base ** (state_len - i - 1)
        for i in range(state_len)
    ) * len(alphabet)
    move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
    stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
    move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
    return stay_scores, move_scores

class SmoothCTCLoss(nn.Module):
    def __init__(self, num_classes = 5, blank = 0, weight=0.01):
        super().__init__()
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1) # switch to batch first
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)

        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss

class ReverseLSTM(nn.Module):

    """
    Customized LSTM module to handle direction
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False,
                 reverse = False):
        super(ReverseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            bias,
                            batch_first,
                            dropout,
                            bidirectional)
        self.reverse = reverse


    def forward(self, x):
        if self.reverse: x = x.flip(0)
        x, h = self.lstm(x)
        if self.reverse: x = x.flip(0)
        return x

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias=True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, qkv):
        if torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (
                torch.is_autocast_enabled() or qkv.dtype == torch.half):
            attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
        else:
            q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
            mask = sliding_window_mask(qkv.shape[1], self.attn_window, q.device)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.permute(0, 1, 3, 2, 4)
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)

        qkv = self.rotary_emb(qkv)

        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)

        out = self.out_proj(attn_output)

        return out


class MultiHeadAttention_2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        # 投影并分头 [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(*query.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(*key.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(*value.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)

        # 调用高效注意力计算
        with torch.backends.cuda.sdp_kernel(enable_flash=True):  # 强制使用Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        # 合并多头输出 [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(*query.shape[:2], self.embed_dim)
        return self.out_proj(attn_output)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000, n = 4000.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(n) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta,
            "attn_window": attn_window
        }

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        # self.drop_out = nn.Dropout(p=0.1)
        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        # x = self.norm1(self.drop_out(self.self_attn(x)), self.deepnorm_alpha*x)
        # x = self.norm2(self.drop_out(self.ff(x)), self.deepnorm_alpha*x)
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha * x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha * x)
        return x

class CTC_encoder(nn.Module):
    def __init__(self, conv : list = [8,64], n_hid : int = 512, dropout : float = 0.2):
        super(CTC_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv[0]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[0], out_channels=conv[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv[1]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[1], out_channels=n_hid, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            Permute(2, 0, 1), # (N, C, T) -> (T, N, C)
            ReverseLSTM(n_hid, n_hid, batch_first=False, reverse=True),
            ReverseLSTM(n_hid, n_hid, batch_first=False, reverse=False),
            ReverseLSTM(n_hid, n_hid, batch_first=False, reverse=True),
            ReverseLSTM(n_hid,n_hid // 4, batch_first=False, reverse=False),
            ReverseLSTM(n_hid // 4, n_hid // 16, batch_first=False, reverse=True),
            nn.Linear(n_hid // 16, 5)
        )

    def forward(self, x):
        return self.encoder(x).log_softmax(dim=2)

class Transformer_CTC_encoder(nn.Module):
    def __init__(self, conv: list = [16, 16], n_hid: int = 512, num_tf : int = 2):
        super(Transformer_CTC_encoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv[0]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[0], out_channels=conv[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv[1]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[1], out_channels=n_hid, kernel_size=19, stride=5, padding=9),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            Permute(0, 2, 1),  # (N, C, T) -> (N, T, C)
        )
        self.tf_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model = n_hid,
                    nhead = 8,
                    dim_feedforward = n_hid * 4,
                    deepnorm_alpha = 2.0,
                    deepnorm_beta = 2.0,
                    attn_window=[127, 128]
                )
            for _ in range(num_tf)],
            Permute(1, 0, 2), # (N, T, C) -> (T, N, C)
        )

        self.linear = nn.Linear(n_hid, 5)
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.tf_encoder(x)
        return self.linear(x).log_softmax(dim=2)


class Mamba_CTC_encoder(nn.Module):
    def __init__(self, conv: list = [16, 16], n_hid: int = 512, num_layers: int = 2):
        super(Mamba_CTC_encoder, self).__init__()

        # CNN encoder remains the same
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv[0]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[0], out_channels=conv[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(conv[1]),
            nn.SiLU(),
            nn.Conv1d(in_channels=conv[1], out_channels=n_hid, kernel_size=19, stride=5, padding=9),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_hid, out_channels=n_hid, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(n_hid),
            nn.SiLU(),
            Permute(0, 2, 1),  # (N, C, T) -> (N, T, C)
        )

        # Replace Transformer with Mamba
        self.mamba_encoder = nn.ModuleList([
            Mamba(
                d_model=n_hid,  # Model dimension
                d_state=64,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2  # Expansion factor in block
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(n_hid)
        self.linear = nn.Linear(n_hid, 5)

    def forward(self, x):
        x = self.cnn_encoder(x)

        # Apply Mamba layers
        for mamba_layer in self.mamba_encoder:
            x = mamba_layer(x)

        x = self.norm(x)
        x = x.transpose(0, 1)  # (N, T, C) -> (T, N, C) to match original output format

        return self.linear(x).log_softmax(dim=2)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, hid_dim)  # [batch_size, seq_length, input_dim]
        self.key = nn.Linear(input_dim, hid_dim)  # [batch_size, seq_length, input_dim]
        self.value = nn.Linear(input_dim, hid_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):  # x.shape (batch_size, seq_length, input_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # scaled self attention
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention model, modified from MultiRM (https://github.com/Tsedao/MultiRM)

    Args:
        hidden_states (tensor): The hiden state from LSTM.
        values (tensor): The output from LSTM.

    Returns:
        tensor: context_vector, attention_weights.
    """

    def __init__(self, in_features = 512, hidden_units = 10, num_task = 1):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(nn.Tanh()(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values, 1, 2)
        # transpose to make it suitable for matrix multiplication

        context_vector = torch.matmul(values, attention_weights)
        context_vector = torch.transpose(context_vector, 1, 2)
        return context_vector, attention_weights

class BiLSTM_attn(nn.Module):
    def __init__(self,
                 kmer : int = 21,
                 hidden_size : int = 256,
                 embed_size : list = [16, 4],
                 dropout_rate : float = 0.5,
                 num_layer1 : int = 2,
                 num_layer2 : int = 3,
                 num_classes : int = 2,
                 ):
        super(BiLSTM_attn, self).__init__()
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(embed_size[0], embed_size[1])
        self.lstm_seq = nn.LSTM(8,
                                hidden_size // 2,
                                num_layers=num_layer1,
                                batch_first=True,
                                dropout=dropout_rate,
                                bidirectional=True,
                                )

        # self.linear_seq = nn.Linear(hidden_size, hidden_size // 2)
        # self.attn_seq = SelfAttention(hidden_size)

        self.lstm_signal = nn.LSTM(15,
                                   (hidden_size // 2),
                                   num_layers=num_layer1,
                                   batch_first=True,
                                   dropout=dropout_rate,
                                   bidirectional=True,
                                   )
        # self.linear_signal = nn.Linear(hidden_size, hidden_size // 2)
        # self.attn_sig = SelfAttention(hidden_size)

        self.attn_cat = SelfAttention(hidden_size * 2, hidden_size)

        self.lstm_comb = nn.LSTM(hidden_size,
                                 hidden_size,
                                 num_layers=num_layer2,
                                 batch_first=True,
                                 dropout=dropout_rate,
                                 bidirectional=True)

        self.drop_out = nn.Dropout(p=dropout_rate)

        self.linear_out_1 = nn.Linear(hidden_size * 2, 2)
        self.linear_out_2 = nn.Linear(2 * kmer, num_classes)

        self.soft_max = nn.Softmax(1)

    def forward(self, kmer, signals):
        kmer_embed = self.embed(kmer.long())
        # signals = signals.reshape(signals.shape[0], signals.shape[2], signals.shape[3])

        out_seq = torch.cat((kmer_embed, signals[:, :, :4]), 2)

        out_signal = signals[:, :, 4:]
        out_seq, _ = self.lstm_seq(out_seq)  # (N, L, nhid_seq * 2)
        # out_seq = self.linear_seq(out_seq)  # (N, L, nhid_seq)
        # out_seq = self.attn_seq(out_seq)
        out_seq = self.relu(out_seq)

        out_signal, _ = self.lstm_signal(out_signal)
        # out_signal = self.linear_signal(out_signal)  # (N, L, nhid_signal)
        # out_signal = self.attn_sig(out_signal)
        out_signal = self.relu(out_signal)

        # combined ================================================
        out = torch.cat((out_seq, out_signal), 2)  # (N, L, hidden_size)
        out = self.attn_cat(out)
        out, _ = self.lstm_comb(out, )  # (N, L, hidden_size * 2)

        out = self.drop_out(out)
        out = self.linear_out_1(out).flatten(1)
        out = self.drop_out(out)
        out = self.linear_out_2(out)

        return out, self.soft_max(out)