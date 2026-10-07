# type: ignore
import warnings
from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import gather, scatter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import Optimizer, ParamsT

from aro.processors import (
    AdamProcessor,
    ConstantNormProcessor,
    GradientProcessor,
    RotationProcessor,
)


def params_to_groups(params: ParamsT) -> list[dict[str, Tensor]]:
    """Helper function to create a group of parameters.

    Args:
        params (ParamsT): The input params.

    Returns:
        list[dict[str, Tensor]]: A list of dict that contains the parameter tensors.
    """
    if all(isinstance(p, dict) for p in params):
        return params
    if all(isinstance(p, torch.nn.Parameter) for p in params):
        return [dict(params=params)]
    if all(isinstance(p, list) for p in params):
        return [dict(params=p) for p in params]
    raise ValueError(f"Invalid paramst_to_groups input: {params}")


def _mesh_of(p: DTensor | Tensor) -> DeviceMesh:
    if isinstance(p, DTensor):
        return p.device_mesh
    elif torch.distributed.is_initialized():
        ws = dist.get_world_size()
        return DeviceMesh(p.device.type, list(range(ws)))
    raise ValueError("Cannot determine mesh for non-DTensor param without initialized dist")


def _world_size(mesh: DeviceMesh) -> int:
    return mesh.size(-1)


def _global_rank(mesh: DeviceMesh) -> int:
    return mesh.get_rank()


# helpers for uneven splits and padding.
def _rowsplit_sizes(dim: int, ws: int) -> list[int]:
    """Return per-rank row counts for sharding dim across ws ranks."""
    q, r = divmod(dim, ws)
    return [q + (1 if i < r else 0) for i in range(ws)]  # first r ranks get +1


def _pad_rows(x: Tensor, target_rows: int) -> Tensor:
    """Pad x on dim 0 with zeros up to target_rows (no-op if already equal)."""
    rows = x.shape[0]
    if rows == target_rows:
        return x
    out = x.new_zeros((target_rows, *x.shape[1:]))
    if rows > 0:
        out[:rows].copy_(x)
    return out


def _split_by_sizes(x: Tensor, sizes: list[int]) -> list[Tensor]:
    """Split x along dim 0 using exact sizes."""
    out, start = [], 0
    for s in sizes:
        out.append(x[start : start + s])
        start += s
    return out


# identity builder now supports uneven splits
def _build_u_identity(p: Tensor, owner: bool) -> Tensor | None:
    """Build U = I(rs) as a DTensor sharded on rows (dim 0), supporting uneven row splits.

    Global shape: (rs, rs), placement: Shard(0) on a 1D mesh.
    """
    assert p.ndim == 2
    m, n = p.shape
    rs = min(m, n)

    if isinstance(p, DTensor):
        mesh = _mesh_of(p)
        rloc = _global_rank(mesh)
        ws = _world_size(mesh)
        sizes = _rowsplit_sizes(rs, ws)
        rows_local = sizes[rloc]
        start = sum(sizes[:rloc])

        # local (rows_local, rs) identity slice in fp32 (for numerical stability), cast later
        u_local = torch.zeros((rows_local, rs), device=p.device, dtype=torch.float32)
        if rows_local > 0:
            u_local[:, start : start + rows_local] = torch.eye(
                rows_local, dtype=u_local.dtype, device=p.device
            )
        return DTensor.from_local(
            u_local.to(dtype=p.dtype),
            device_mesh=mesh,
            placements=(Shard(0),),
            shape=(rs, rs),
            stride=(rs, 1),
            run_check=True,
        )
    elif owner:
        return torch.eye(rs, dtype=p.dtype, device=p.device)
    return None


# main optimizer
@dataclass
class _ChunkSlot:
    # Events that gate the pipeline for one parameter within a chunk
    gathered: torch.cuda.Event | None = None
    rotated: torch.cuda.Event | None = None
    scattered: torch.cuda.Event | None = None
    # Owner-side working tensors (live on GPU; None on non-owner)
    g_full: Tensor | None = None
    u_full: Tensor | None = None
    b_full: dict[str, Tensor] = field(default_factory=dict)  # base processor states

    # DDP async broadcast handles + optional step buffer, this is not used at the moment. Will fix later.
    bcast_works: list[Any] = field(default_factory=list)
    step_buf: torch.Tensor | None = None


#  main optimizer
class AroBase(Optimizer):
    """Aro Optimizer with overlapped gather/rotate/scatter.

    This efficient implementation can be 10x faster than Aro DDP with large number of GPUs (e.g. 64) and large models (e.g. 1b).
    """

    def __init__(
        self,
        params: ParamsT,
        base_processor: GradientProcessor,
        ema_processor: AdamProcessor,
        rot_processor: RotationProcessor,
        norm_processor: ConstantNormProcessor | None = None,
        *,
        lr: float = 1e-3,
        alpha: float = 0.2,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        momentum: float = 0.95,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            update_freq=update_freq,
            momentum=momentum,
        )
        super().__init__(params_to_groups(list(params)), defaults)

        self._base_processor = base_processor
        self._ema_processor = ema_processor
        self._norm_processor = norm_processor
        self._rot_processor = rot_processor

        first_param = next(chain.from_iterable(pg["params"] for pg in self.param_groups))
        use_fsdp_cuda = isinstance(first_param, DTensor) and first_param.device.type == "cuda"
        self._comm_stream = torch.cuda.Stream() if use_fsdp_cuda else None
        self._comp_stream = torch.cuda.Stream() if use_fsdp_cuda else None

        # Per-parameter persistent state (same keys you already use)
        for pg in self.param_groups:
            for p in pg["params"]:
                assert p.ndim == 2, f"AROFSDP2 supports 2D only; got {p.shape}"
                self.state[p] = dict(step=0, u=None, ema=None, base=None)

    def ensure_state_initialized(self) -> None:
        """Materialize optimizer states for checkpoint loading paths.

        DCP load builds optimizer state skeletons before the first backward/step.
        ARO normally initializes these lazily in `step`, so we need an explicit
        initializer to create state tensors up front.
        """
        p = next(chain.from_iterable(pg["params"] for pg in self.param_groups))
        mesh = _mesh_of(p) if dist.is_initialized() else None
        r_global = _global_rank(mesh) if dist.is_initialized() else 0
        ws = _world_size(mesh) if dist.is_initialized() else 1
        assert mesh is None or len(mesh.shape) == 1, f"Expect 1D mesh; got {mesh}"

        for pg in self.param_groups:
            for idx, p in enumerate(pg["params"]):
                st = self.state[p]
                ref = p.grad if p.grad is not None else p
                if st["ema"] is None:
                    st["ema"] = self._ema_processor.init_state_dict(ref)
                if st["base"] is None:
                    st["base"] = self._base_processor.init_state_dict(ref)
                if st["u"] is None:
                    st["u"] = _build_u_identity(p, owner=idx % ws == r_global)

    # state helpers
    def _ensure_ema(self, p: Tensor) -> AdamProcessor:
        st = self.state[p]
        if st["ema"] is None:
            st["ema"] = self._ema_processor.init_state_dict(p.grad)
        return st["ema"]

    def _ensure_base(self, p: Tensor) -> GradientProcessor:
        st = self.state[p]
        if st["base"] is None:
            st["base"] = self._base_processor.init_state_dict(p.grad)
        return st["base"]

    def _ensure_u(self, p: Tensor, owner: bool) -> DTensor:
        st = self.state[p]
        if st["u"] is None:
            U = _build_u_identity(p, owner=owner)
            self.state[p]["u"] = U
        return self.state[p]["u"]

    def _local_step(self) -> None:
        for pg in self.param_groups:
            lr: float = pg["lr"]
            alpha: float = pg["alpha"]
            wd: float = pg["weight_decay"]
            update_freq: int = pg["update_freq"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plist: list[Tensor] = [p for p in pg["params"] if p.grad is not None]

            for p in plist:
                st = self.state[p]
                update_sub = (st["step"] % update_freq) == 0
                ema_st = self._ensure_ema(p)
                b_st = self._ensure_base(p)
                u = self._ensure_u(p, owner=True)

                grad = self._ema_processor.process_grad(p.grad, state_dict=ema_st)
                grad, u = self._rot_processor.process_grad(grad, u, update_sub, state_dict=b_st)
                grad = grad.view_as(p).type_as(p)
                if self._norm_processor is not None:
                    grad = self._norm_processor.process_grad(grad)

                st["step"] += 1
                st["u"] = u.view_as(st["u"]).type_as(st["u"])
                p.add_(grad, alpha=-(lr * alpha))
                if wd != 0.0:
                    p.add_(p, alpha=-(lr * wd))

    # main step
    @torch.no_grad()
    def step(self, closure: None = None, *, prefetch_factor: int = 8):
        if self._comm_stream is None or self._comp_stream is None:
            self._local_step()
            return

        p = next(chain.from_iterable(pg["params"] for pg in self.param_groups))
        mesh = _mesh_of(p) if dist.is_initialized() else None
        r_global = _global_rank(mesh) if dist.is_initialized() else 0
        ws = _world_size(mesh) if dist.is_initialized() else 1
        assert mesh is None or len(mesh.shape) == 1, f"Expect 1D mesh; got {mesh}"

        comm_stream = self._comm_stream
        comp_stream = self._comp_stream
        cur_stream = torch.cuda.current_stream()

        # gather/schedule strictly in chunks of size world_size to keep collective
        # order identical across ranks
        def _chunk_bounds(n: int, k: int) -> tuple[int, int]:
            start = k * ws
            end = min(n, start + ws)
            return start, end

        for pg in self.param_groups:
            lr: float = pg["lr"]
            alpha: float = pg["alpha"]
            wd: float = pg["weight_decay"]
            update_freq: int = pg["update_freq"]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plist: list[Tensor] = [p for p in pg["params"] if p.grad is not None]
            if not plist:
                continue

            n_params = len(plist)
            deferred: deque[tuple[Tensor, torch.cuda.Event, float, float, float]] = deque()
            num_chunks = (n_params + ws - 1) // ws

            # (1) EMA on local shard
            for j, p in enumerate(plist):
                # this should not increase the memory, since we process the shard p.
                ema_st = self._ensure_ema(p)  # ema's state dict
                self._ensure_base(p)  # initialize base processor's state dict
                self._ensure_u(p, owner=j % ws == r_global)  # initialize U if needed
                # inplace on p.grad local shard
                p.grad = self._ema_processor.process_grad(p.grad, state_dict=ema_st)

            # Make sure momentum computation is finished before gathering
            comm_stream.wait_stream(cur_stream)

            # process by chunks
            for k in range(num_chunks):
                s, e = _chunk_bounds(n_params, k)
                # Per-param slot for gathered states
                slots: list[_ChunkSlot] = [_ChunkSlot() for _ in range(e - s)]

                # (2) enqueue gather for the chunk on the comm stream
                with torch.cuda.stream(comm_stream):
                    for j, p in enumerate(plist[s:e]):
                        dest = (s + j) % ws
                        st = self.state[p]
                        g_dt: Tensor = p.grad
                        u_dt: Tensor | None = st["u"]
                        b_st: dict[str, Any] = st["base"]

                        # precompute uneven split meta (m/rs + max rows)
                        if isinstance(g_dt, DTensor) and isinstance(u_dt, DTensor):
                            m, n = p.shape
                            rs = min(m, n)
                            rowsP_all = _rowsplit_sizes(m, ws)
                            rowsU_all = _rowsplit_sizes(rs, ws)
                            max_rowsP = max(rowsP_all)
                            max_rowsU = max(rowsU_all)

                        # owner vs non-owner
                        if (
                            dest == r_global
                            and isinstance(g_dt, DTensor)
                            and isinstance(u_dt, DTensor)
                        ):
                            # pad-to-max gather for G, U, and DTensor base states
                            g_local = g_dt.to_local()
                            u_local = u_dt.to_local()
                            g_pad = _pad_rows(g_local, max_rowsP)
                            u_pad = _pad_rows(u_local, max_rowsU)

                            g_recv = [torch.empty_like(g_pad) for _ in range(ws)]
                            u_recv = [torch.empty_like(u_pad) for _ in range(ws)]

                            b_recv: dict[str, list[Tensor]] = {}
                            b_local_pad: dict[str, Tensor] = {}

                            for bk, bv in b_st.items():
                                if isinstance(bv, DTensor):
                                    # DTensor base states are sharded on dim0; split by their own global rows
                                    b_rows_all = _rowsplit_sizes(bv.shape[0], ws)
                                    b_max_rows = max(b_rows_all)
                                    v_local = bv.to_local()
                                    v_pad_local = _pad_rows(v_local, b_max_rows)
                                    b_local_pad[bk] = v_pad_local
                                    b_recv[bk] = [torch.empty_like(v_pad_local) for _ in range(ws)]

                            gather(g_pad, g_recv, dst=dest, async_op=False)
                            gather(u_pad, u_recv, dst=dest, async_op=False)
                            for bk, bv in b_st.items():
                                if isinstance(bv, DTensor):
                                    gather(b_local_pad[bk], b_recv[bk], dst=dest, async_op=False)

                            # slice back to true sizes and concatenate
                            g_parts = [t[: rowsP_all[i]] for i, t in enumerate(g_recv)]
                            u_parts = [t[: rowsU_all[i]] for i, t in enumerate(u_recv)]
                            slots[j].g_full = torch.cat(g_parts, dim=0)
                            slots[j].u_full = torch.cat(u_parts, dim=0)

                            slots[j].b_full = {}
                            for bk, bv in b_st.items():
                                if isinstance(bv, DTensor):
                                    b_rows_all = _rowsplit_sizes(bv.shape[0], ws)
                                    b_parts = [
                                        t[: b_rows_all[i]] for i, t in enumerate(b_recv[bk])
                                    ]
                                    slots[j].b_full[bk] = torch.cat(b_parts, dim=0)
                                else:
                                    slots[j].b_full[bk] = bv  # non-DTensor states unchanged

                            # set flag for event
                            slots[j].gathered = torch.cuda.Event()
                            slots[j].gathered.record(comm_stream)

                        elif (
                            dest != r_global
                            and isinstance(g_dt, DTensor)
                            and isinstance(u_dt, DTensor)
                        ):
                            # non-owner: send padded locals (equal sizes) to owner
                            g_local = g_dt.to_local()
                            u_local = u_dt.to_local()
                            g_pad = _pad_rows(g_local, max_rowsP)
                            u_pad = _pad_rows(u_local, max_rowsU)

                            gather(g_pad, None, dst=dest, async_op=False)
                            gather(u_pad, None, dst=dest, async_op=False)
                            for bv in b_st.values():
                                if isinstance(bv, DTensor):
                                    b_rows_all = _rowsplit_sizes(bv.shape[0], ws)
                                    b_max_rows = max(b_rows_all)
                                    v_pad_local = _pad_rows(bv.to_local(), b_max_rows)
                                    gather(v_pad_local, None, dst=dest, async_op=False)

                            slots[j].gathered = None  # not used on non-owner

                        elif dest == r_global:
                            # DDP path (non-DTensor): owner runs with full tensors already local
                            slots[j].g_full = g_dt
                            slots[j].u_full = u_dt
                            slots[j].b_full = b_st
                            slots[j].gathered = torch.cuda.Event()
                            slots[j].gathered.record(comm_stream)

                # (3) owner computation
                with torch.cuda.stream(comp_stream):
                    for j, p in enumerate(plist[s:e]):
                        dest = (s + j) % ws
                        st = self.state[p]
                        if dest != r_global:
                            continue  # only owner computes

                        # wait gather stream
                        comp_stream.wait_event(slots[j].gathered)

                        # retrieve from per-param slot
                        g_full = slots[j].g_full
                        u_full = slots[j].u_full
                        b_full = slots[j].b_full
                        assert g_full is not None and u_full is not None

                        update_sub = (st["step"] % update_freq) == 0
                        gt, ut = self._rot_processor.process_grad(
                            g_full, u_full, update_sub, state_dict=b_full
                        )

                        # same casting logic you had
                        g_full = g_full.copy_(gt).view_as(p).type_as(p)
                        u_full = u_full.copy_(ut).type_as(st["u"])

                        if self._norm_processor is not None:
                            g_full = self._norm_processor.process_grad(g_full)

                        slots[j].g_full = g_full
                        slots[j].u_full = u_full

                        # persist the updated U and base on the owner for DDP,
                        # non-owners will receive them via broadcast below
                        if not isinstance(p.grad, DTensor):  # DDP
                            st["u"] = u_full
                            st["base"] = b_full

                        slots[j].rotated = torch.cuda.Event()
                        slots[j].rotated.record(comp_stream)

                # (4) scatter back the updates
                with torch.cuda.stream(comm_stream):
                    for j, p in enumerate(plist[s:e]):
                        dest = (s + j) % ws
                        st = self.state[p]
                        g_dt: Tensor = p.grad if p.grad is not None else torch.zeros_like(p)
                        u_dt: Tensor | None = st["u"]
                        b_st: dict[str, Any] = st["base"]

                        if isinstance(g_dt, DTensor) and isinstance(u_dt, DTensor):
                            # compute sizes for uneven split
                            m, n = p.shape
                            rs = min(m, n)
                            rowsP_all = _rowsplit_sizes(m, ws)
                            rowsU_all = _rowsplit_sizes(rs, ws)
                            max_rowsP = max(rowsP_all)
                            max_rowsU = max(rowsU_all)
                            rowsP_local = g_dt.to_local().shape[0]
                            rowsU_local = u_dt.to_local().shape[0]

                        if dest == r_global and isinstance(g_dt, DTensor):  # fsdp and owner
                            # wait for finished computation
                            comm_stream.wait_event(slots[j].rotated)

                            g_full = slots[j].g_full
                            u_full = slots[j].u_full
                            b_full = slots[j].b_full
                            assert g_full is not None and u_full is not None

                            # split by true sizes, then pad each piece to max_rows for NCCL scatter
                            g_chunks = _split_by_sizes(g_full, rowsP_all)
                            u_chunks = _split_by_sizes(u_full, rowsU_all)
                            g_chunks_pad = [_pad_rows(t, max_rowsP) for t in g_chunks]
                            u_chunks_pad = [_pad_rows(t, max_rowsU) for t in u_chunks]

                            # prepare base state chunks similarly
                            b_chunks_pad: dict[str, list[Tensor] | None] = {}
                            for k2, v in b_full.items():
                                if isinstance(v, Tensor):
                                    b_rows_all = _rowsplit_sizes(v.shape[0], ws)
                                    b_max_rows = max(b_rows_all)
                                    parts = _split_by_sizes(v, b_rows_all)
                                    b_chunks_pad[k2] = [_pad_rows(t, b_max_rows) for t in parts]
                                else:
                                    b_chunks_pad[k2] = None  # non-DTensor

                            # receive into temp padded buffers, then copy into local DTensor
                            g_recv_tmp = torch.empty(
                                (max_rowsP, g_dt.to_local().shape[1]),
                                device=g_dt.to_local().device,
                                dtype=g_dt.to_local().dtype,
                            )
                            scatter(g_recv_tmp, g_chunks_pad, src=dest, async_op=False)
                            if rowsP_local > 0:
                                g_dt.to_local().copy_(g_recv_tmp[:rowsP_local])

                            u_recv_tmp = torch.empty(
                                (max_rowsU, u_dt.to_local().shape[1]),
                                device=u_dt.to_local().device,
                                dtype=u_dt.to_local().dtype,
                            )
                            scatter(u_recv_tmp, u_chunks_pad, src=dest, async_op=False)
                            if rowsU_local > 0:
                                u_dt.to_local().copy_(u_recv_tmp[:rowsU_local])

                            for k2, v in b_st.items():
                                if isinstance(v, DTensor):
                                    b_rows_all = _rowsplit_sizes(v.shape[0], ws)
                                    b_max_rows = max(b_rows_all)
                                    recv_tmp = torch.empty(
                                        (b_max_rows, *v.to_local().shape[1:]),
                                        device=v.to_local().device,
                                        dtype=v.to_local().dtype,
                                    )
                                    scatter(recv_tmp, b_chunks_pad[k2], src=dest, async_op=False)
                                    local_rows = v.to_local().shape[0]
                                    if local_rows > 0:
                                        v.to_local().copy_(recv_tmp[:local_rows])
                                else:
                                    b_st[k2] = b_full[k2]  # non-DTensor states are unchanged

                        elif dest != r_global and isinstance(g_dt, DTensor):  # fsdp and not owner
                            # receive padded chunk, then copy only true rows to local
                            g_recv_tmp = torch.empty(
                                (max_rowsP, g_dt.to_local().shape[1]),
                                device=g_dt.to_local().device,
                                dtype=g_dt.to_local().dtype,
                            )
                            scatter(g_recv_tmp, None, src=dest, async_op=False)
                            rowsP_local = g_dt.to_local().shape[0]
                            if rowsP_local > 0:
                                g_dt.to_local().copy_(g_recv_tmp[:rowsP_local])

                            u_recv_tmp = torch.empty(
                                (max_rowsU, u_dt.to_local().shape[1]),
                                device=u_dt.to_local().device,
                                dtype=u_dt.to_local().dtype,
                            )
                            scatter(u_recv_tmp, None, src=dest, async_op=False)
                            rowsU_local = u_dt.to_local().shape[0]
                            if rowsU_local > 0:
                                u_dt.to_local().copy_(u_recv_tmp[:rowsU_local])

                            for v in b_st.values():
                                if isinstance(v, DTensor):
                                    b_rows_all = _rowsplit_sizes(v.shape[0], ws)
                                    b_max_rows = max(b_rows_all)
                                    recv_tmp = torch.empty(
                                        (b_max_rows, *v.to_local().shape[1:]),
                                        device=v.to_local().device,
                                        dtype=v.to_local().dtype,
                                    )
                                    scatter(recv_tmp, None, src=dest, async_op=False)
                                    local_rows = v.to_local().shape[0]
                                    if local_rows > 0:
                                        v.to_local().copy_(recv_tmp[:local_rows])

                        elif dest == r_global and dist.is_initialized():  # ddp and owner
                            comm_stream.wait_event(slots[j].rotated)
                            g_full = slots[j].g_full
                            assert g_full is not None
                            g_chunks = [g_full for _ in range(ws)]
                            scatter(g_dt, g_chunks, src=dest, async_op=False)
                        elif dest != r_global and dist.is_initialized():  # ddp and not owner
                            scatter(g_dt, None, src=dest, async_op=False)
                        elif not dist.is_initialized():  # single-process / owner
                            # no scatter, but keep dependency comp_stream -> comm_stream
                            comm_stream.wait_event(slots[j].rotated)
                            p.grad.copy_(slots[j].g_full)

                        # set flag for scatter event
                        slots[j].scattered = torch.cuda.Event()
                        slots[j].scattered.record(comm_stream)

                        # continue to next param
                        continue

                # (5) set deferred queue
                for j, p in enumerate(plist[s:e]):
                    st = self.state[p]
                    st["step"] += 1
                    deferred.append((p, slots[j].scattered, slots[j], lr, alpha, wd))

                # prefetch factor
                while len(deferred) > prefetch_factor:
                    p0, ev0, slot0, _lr, _alpha, _wd = deferred.popleft()
                    if ev0 is not None:
                        torch.cuda.current_stream().wait_event(ev0)

                    for wk in getattr(slot0, "bcast_works", []):
                        wk.wait()  # wait until broadcast is done

                    # align base['step'] if it exists
                    if (
                        getattr(slot0, "step_buf", None) is not None
                        and isinstance(self.state[p0]["base"], dict)
                        and "step" in self.state[p0]["base"]
                    ):
                        self.state[p0]["base"]["step"] = int(slot0.step_buf.item())
                    # now the update is safe & identical across ranks
                    p0.add_(
                        p0.grad if p0.grad is not None else torch.zeros_like(p0),
                        alpha=-(_lr * _alpha),
                    )
                    if _wd != 0.0:
                        p0.add_(p0, alpha=-(_lr * _wd))

            # remaining updates
            while deferred:
                p0, ev0, slot0, _lr, _alpha, _wd = deferred.popleft()
                if ev0 is not None:
                    torch.cuda.current_stream().wait_event(ev0)

                for wk in getattr(slot0, "bcast_works", []):
                    wk.wait()  # wait until broadcast is done

                # align base['step'] if it exists
                if (
                    getattr(slot0, "step_buf", None) is not None
                    and isinstance(self.state[p0]["base"], dict)
                    and "step" in self.state[p0]["base"]
                ):
                    self.state[p0]["base"]["step"] = int(slot0.step_buf.item())
                # now the update is safe & identical across ranks
                p0.add_(
                    p0.grad if p0.grad is not None else torch.zeros_like(p0), alpha=-(_lr * _alpha)
                )
                if _wd != 0.0:
                    p0.add_(p0, alpha=-(_lr * _wd))
