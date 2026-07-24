"""How an attention block produces its queries, keys and values.

Two schemes. The ordinary one keeps three full projection matrices. The shared one
keeps a single matrix and recovers the three roles with a small per-role operator.
Every parameter the experiment saves is saved here, and nowhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor, nn

from ana.nn.roles import RoleTransform

RoleFactory = Callable[[int], RoleTransform]


class QKVProjection(nn.Module, ABC):
    """Turns the block's inputs into a query, a key and a value stream.

    The three roles can also be asked for separately. Generation needs that: a step of the
    decoder wants a query for the token it has just produced, while the keys and values of
    the encoder output were computed once and have not changed since.
    """

    @abstractmethod
    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor: ...

    @abstractmethod
    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def forward(
        self,
        query_input: Tensor,
        kv_input: Tensor,
        query_mask: Tensor,
        kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class SeparateQKV(QKVProjection):
    """Three independent projections. The usual arrangement."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        return self.query(query_input)

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        return self.key(kv_input), self.value(kv_input)

    def forward(
        self,
        query_input: Tensor,
        kv_input: Tensor,
        query_mask: Tensor,
        kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.query(query_input), self.key(kv_input), self.value(kv_input)


class SharedQKV(QKVProjection):
    """One projection, then one small operator per role.

    In self-attention the query stream and the key-value stream are the same tensor, so
    the shared projection runs once. In cross-attention they differ, so it runs twice —
    once over the decoder side for the queries, once over the encoder output for the keys
    and values — but it is still a single matrix, shared by both.
    """

    def __init__(self, d_model: int, role_factory: RoleFactory) -> None:
        super().__init__()
        self.shared = nn.Linear(d_model, d_model)
        self.query_role = role_factory(d_model)
        self.key_role = role_factory(d_model)
        self.value_role = role_factory(d_model)

    def query_only(self, query_input: Tensor, query_mask: Tensor) -> Tensor:
        return self.query_role(self.shared(query_input), query_mask)

    def key_value(self, kv_input: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
        z = self.shared(kv_input)
        return self.key_role(z, kv_mask), self.value_role(z, kv_mask)

    def forward(
        self,
        query_input: Tensor,
        kv_input: Tensor,
        query_mask: Tensor,
        kv_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        z_query = self.shared(query_input)
        z_kv = z_query if kv_input is query_input else self.shared(kv_input)

        query = self.query_role(z_query, query_mask)
        key = self.key_role(z_kv, kv_mask)
        value = self.value_role(z_kv, kv_mask)
        return query, key, value
