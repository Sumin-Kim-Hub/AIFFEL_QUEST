"""
autointmlp_v3.py
================
AutoInt+ (AutoInt with MLP branch) — TensorFlow 2.x / Keras Subclassing API
V3: Narrative_Keyword 피처 추가 버전

피처 구성 (필드 순서)
---------------------
0: user_id
1: movie_id
2: genre (multi-hot → 하나의 장르 인덱스로 label-encoded)
3: narrative_keyword  ← V3 신규

Author note
-----------
- 레이어 이름 충돌 방지 → 모든 add_weight / Embedding에 고유 name 부여
- dtype int64 통일 (longlong 제거)
- 가중치 저장: model.save_weights("autoIntMLP_V3_weights.weights.h5")
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer, Dense, Flatten, Dropout,
    BatchNormalization, Activation, Embedding,
)
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model


# ──────────────────────────────────────────────────────────────────────────────
# 1. FeaturesEmbedding
#    기존 필드(user/movie/genre) + narrative_keyword 를 단일 offset-shifted
#    embedding table로 처리.
# ──────────────────────────────────────────────────────────────────────────────
class FeaturesEmbedding(Layer):
    """
    Parameters
    ----------
    field_dims : list[int]
        각 필드의 고유값 수. 예: [user_cnt, movie_cnt, genre_cnt, narr_cnt]
    embed_dim : int
        임베딩 차원 (모든 필드 공유)
    """

    def __init__(self, field_dims: list[int], embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.total_dim = int(sum(field_dims))
        # offset: 각 필드 인덱스가 embedding table 내 어느 위치에서 시작하는지
        self.offsets = np.array(
            [0, *np.cumsum(field_dims)[:-1]], dtype=np.int64
        )
        self.embedding = Embedding(
            input_dim=self.total_dim,
            output_dim=self.embed_dim,
            embeddings_initializer="glorot_uniform",
            name=f"{self.name}_embed_table",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        x : (batch, num_fields)  dtype=int64
        returns : (batch, num_fields, embed_dim)
        """
        x = tf.cast(x, dtype=tf.int64)
        x = x + tf.constant(self.offsets, dtype=tf.int64)
        return self.embedding(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"field_dims": self.field_dims, "embed_dim": self.embed_dim})
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 2. MultiHeadSelfAttention (AutoInt 논문 구현 유지)
#    - W_Query / W_key / W_Value / W_Res 모두 고유 name 부여
#    - att_layer_num 만큼 반복 → 각 레이어 인스턴스에 layer_id 주입
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(Layer):
    """
    AutoInt 논문의 interacting layer.

    Parameters
    ----------
    att_embedding_size : int
        각 head의 출력 차원 (d_head)
    head_num : int
        attention head 수
    use_res : bool
        residual connection 사용 여부
    scaling : bool
        scaled dot-product attention 여부
    layer_id : int
        레이어 이름 충돌 방지용 고유 ID
    """

    def __init__(
        self,
        att_embedding_size: int = 8,
        head_num: int = 2,
        use_res: bool = True,
        scaling: bool = True,
        seed: int = 1024,
        layer_id: int = 0,
        **kwargs,
    ):
        if head_num <= 0:
            raise ValueError("head_num must be > 0")
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed
        self.layer_id = layer_id
        # name 충돌 방지: 각 인스턴스에 layer_id 포함
        kwargs.setdefault("name", f"mhsa_{layer_id}")
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input, got {len(input_shape)}D"
            )
        embed_dim = int(input_shape[-1])
        proj_dim = self.att_embedding_size * self.head_num
        lid = self.layer_id  # 이름 중복 방지

        self.W_Query = self.add_weight(
            name=f"W_Query_{lid}",
            shape=[embed_dim, proj_dim],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed),
        )
        self.W_key = self.add_weight(
            name=f"W_key_{lid}",
            shape=[embed_dim, proj_dim],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed + 1),
        )
        self.W_Value = self.add_weight(
            name=f"W_Value_{lid}",
            shape=[embed_dim, proj_dim],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed + 2),
        )
        if self.use_res:
            self.W_Res = self.add_weight(
                name=f"W_Res_{lid}",
                shape=[embed_dim, proj_dim],
                dtype=tf.float32,
                initializer=TruncatedNormal(seed=self.seed),
            )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        inputs : (batch, num_fields, embed_dim)
        returns : (batch, num_fields, head_num * att_embedding_size)
        """
        # Linear projections
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))  # (B, F, H*d)
        keys   = tf.tensordot(inputs, self.W_key,   axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value,  axes=(-1, 0))

        # Split heads → (head_num, B, F, d_head)
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys   = tf.stack(tf.split(keys,   self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        # Scaled dot-product
        scores = tf.matmul(querys, keys, transpose_b=True)          # (H, B, F, F)
        if self.scaling:
            scores = scores / (self.att_embedding_size ** 0.5)
        attn = tf.nn.softmax(scores)

        # Weighted sum + merge heads
        out = tf.matmul(attn, values)                               # (H, B, F, d_head)
        out = tf.concat(tf.split(out, self.head_num), axis=-1)      # (H, B, F, H*d_head) — wrong
        # ↑ 실제로는 tf.split이 이미 H개를 분리했으므로 concat으로 다시 합침
        out = tf.squeeze(out, axis=0)                               # (B, F, H*d_head)

        # Residual
        if self.use_res:
            out = out + tf.tensordot(inputs, self.W_Res, axes=(-1, 0))

        return tf.nn.relu(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],
                self.att_embedding_size * self.head_num)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "att_embedding_size": self.att_embedding_size,
            "head_num": self.head_num,
            "use_res": self.use_res,
            "scaling": self.scaling,
            "seed": self.seed,
            "layer_id": self.layer_id,
        })
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 3. AutoIntMLP_V3  (Core Layer)
#    필드: [user, movie, genre, narrative_keyword]  ← V3 추가
# ──────────────────────────────────────────────────────────────────────────────
class AutoIntMLP_V3(Layer):
    """
    AutoInt+ V3 Core Layer

    Parameters
    ----------
    field_dims : list[int]
        [user_cnt, movie_cnt, genre_cnt, narrative_keyword_cnt]
        반드시 길이 4 이상.
    embedding_size : int
        공유 임베딩 차원
    att_layer_num : int
        Interacting layer (MHSA) 반복 수
    att_head_num : int
        Multi-head attention head 수
    att_res : bool
        MHSA residual 사용 여부
    dnn_hidden_units : tuple[int]
        DNN hidden layer 크기
    dnn_activation : str
        DNN 활성화 함수
    l2_reg_dnn : float
        DNN L2 정규화
    l2_reg_embedding : float
        임베딩 L2 정규화 (현재 embedding에 직접 적용하지 않고 로그용)
    dnn_use_bn : bool
        DNN BatchNorm 사용 여부
    dnn_dropout : float
        DNN Dropout 비율
    init_std : float
        가중치 초기화 표준편차
    """

    def __init__(
        self,
        field_dims: list[int],
        embedding_size: int,
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: tuple = (256, 128, 64),
        dnn_activation: str = "relu",
        l2_reg_dnn: float = 0.0,
        l2_reg_embedding: float = 1e-5,
        dnn_use_bn: bool = False,
        dnn_dropout: float = 0.4,
        init_std: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.field_dims = list(field_dims)
        self.embedding_size = embedding_size
        self.att_layer_num = att_layer_num
        self.att_head_num = att_head_num
        self.att_res = att_res
        self.dnn_hidden_units = list(dnn_hidden_units)
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_use_bn = dnn_use_bn
        self.dnn_dropout = dnn_dropout
        self.init_std = init_std
        self.num_fields = len(field_dims)

        # ── Embedding (user + movie + genre + narrative_keyword 통합) ──
        self.embedding_layer = FeaturesEmbedding(
            field_dims, embedding_size, name="feat_embedding"
        )

        # ── Interacting Layers (att_layer_num 개) ──
        # 각 레이어에 고유 layer_id → 가중치 이름 중복 방지
        self.int_layers = [
            MultiHeadSelfAttention(
                att_embedding_size=embedding_size,
                head_num=att_head_num,
                use_res=att_res,
                scaling=True,
                layer_id=i,
                name=f"mhsa_layer_{i}",
            )
            for i in range(att_layer_num)
        ]

        # ── Attention branch output projection ──
        self.att_final = Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=init_std),
            name="att_final_dense",
        )

        # ── DNN branch ──
        # 입력 차원 = num_fields * embedding_size
        dnn_input_dim = self.num_fields * embedding_size
        self.dnn_layers: list[Layer] = []
        for idx, units in enumerate(dnn_hidden_units):
            self.dnn_layers.append(
                Dense(
                    units,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
                    name=f"dnn_dense_{idx}",
                )
            )
            if dnn_use_bn:
                self.dnn_layers.append(
                    BatchNormalization(name=f"dnn_bn_{idx}")
                )
            self.dnn_layers.append(
                Activation(dnn_activation, name=f"dnn_act_{idx}")
            )
            if dnn_dropout > 0:
                self.dnn_layers.append(
                    Dropout(dnn_dropout, name=f"dnn_drop_{idx}")
                )
        self.dnn_output_dense = Dense(
            1,
            kernel_initializer=tf.random_normal_initializer(stddev=init_std),
            name="dnn_output_dense",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        inputs : (batch, num_fields)  dtype=int64
                 필드 순서: [user_idx, movie_idx, genre_idx, narrative_keyword_idx]
        returns : (batch, 1)  sigmoid score
        """
        # 1. 임베딩 → (batch, num_fields, embed_dim)
        embed_x = self.embedding_layer(inputs)

        # 2. DNN branch: flatten embeddings
        dnn_input = tf.reshape(embed_x, (-1, self.num_fields * self.embedding_size))
        x_dnn = dnn_input
        for layer in self.dnn_layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x_dnn = layer(x_dnn, training=training)
            else:
                x_dnn = layer(x_dnn)
        dnn_out = self.dnn_output_dense(x_dnn)          # (batch, 1)

        # 3. Attention branch: interacting layers
        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)                 # (batch, num_fields, embed_dim)
        att_flat = Flatten()(att_input)                  # (batch, num_fields * embed_dim)
        att_out = self.att_final(att_flat)               # (batch, 1)

        # 4. Combine + sigmoid
        y_pred = tf.sigmoid(att_out + dnn_out)
        return y_pred

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "field_dims": self.field_dims,
            "embedding_size": self.embedding_size,
            "att_layer_num": self.att_layer_num,
            "att_head_num": self.att_head_num,
            "att_res": self.att_res,
            "dnn_hidden_units": self.dnn_hidden_units,
            "dnn_activation": self.dnn_activation,
            "l2_reg_dnn": self.l2_reg_dnn,
            "dnn_use_bn": self.dnn_use_bn,
            "dnn_dropout": self.dnn_dropout,
            "init_std": self.init_std,
        })
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 4. AutoIntMLPModel_V3  (Keras Model)
# ──────────────────────────────────────────────────────────────────────────────
class AutoIntMLPModel_V3(Model):
    """
    사용 예시
    ---------
    >>> model = AutoIntMLPModel_V3(
    ...     field_dims=[6041, 3953, 18, 6],   # user/movie/genre/narrative
    ...     embedding_size=16,
    ...     att_layer_num=3,
    ...     att_head_num=2,
    ... )
    >>> model.compile(
    ...     optimizer=tf.keras.optimizers.Adam(1e-3),
    ...     loss="binary_crossentropy",
    ...     metrics=["AUC", "binary_accuracy"],
    ... )
    >>> # 가중치 저장
    >>> model.save_weights("autoIntMLP_V3_weights.weights.h5")
    """

    def __init__(
        self,
        field_dims: list[int],
        embedding_size: int = 16,
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: tuple = (256, 128, 64),
        dnn_activation: str = "relu",
        l2_reg_dnn: float = 0.0,
        l2_reg_embedding: float = 1e-5,
        dnn_use_bn: bool = False,
        dnn_dropout: float = 0.4,
        init_std: float = 0.0001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core = AutoIntMLP_V3(
            field_dims=field_dims,
            embedding_size=embedding_size,
            att_layer_num=att_layer_num,
            att_head_num=att_head_num,
            att_res=att_res,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            l2_reg_dnn=l2_reg_dnn,
            l2_reg_embedding=l2_reg_embedding,
            dnn_use_bn=dnn_use_bn,
            dnn_dropout=dnn_dropout,
            init_std=init_std,
            name="autoint_mlp_v3_core",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.core(inputs, training=training)

    def get_config(self):
        cfg = super().get_config()
        core_cfg = self.core.get_config()
        # core_cfg에서 필요한 항목 병합
        for key in [
            "field_dims", "embedding_size", "att_layer_num", "att_head_num",
            "att_res", "dnn_hidden_units", "dnn_activation", "l2_reg_dnn",
            "dnn_use_bn", "dnn_dropout", "init_std",
        ]:
            cfg[key] = core_cfg.get(key)
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 5. predict_model  (추론 유틸)
# ──────────────────────────────────────────────────────────────────────────────
def predict_model(model: AutoIntMLPModel_V3, pred_df, top: int = 10) -> list[tuple]:
    """
    Parameters
    ----------
    model : AutoIntMLPModel_V3
    pred_df : pd.DataFrame
        컬럼 순서: [user_idx, movie_idx, genre_idx, narrative_keyword_idx]
        (label 컬럼 없음, 피처만)
    top : int
        반환할 상위 추천 수

    Returns
    -------
    list of (movie_idx, score) tuples, 점수 내림차순
    """
    batch_size = 2048
    results: list[tuple] = []
    total_rows = len(pred_df)

    for start in range(0, total_rows, batch_size):
        features = pred_df.iloc[start: start + batch_size].values.astype(np.int64)
        y_pred = model.predict(features, verbose=0)  # (batch, 1)

        for feat, p in zip(features, y_pred):
            movie_idx = int(feat[1])
            score = float(p[0] if hasattr(p, "__len__") else p)
            results.append((movie_idx, score))

    return sorted(results, key=lambda s: s[1], reverse=True)[:top]
