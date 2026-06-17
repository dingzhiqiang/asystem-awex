"""Unit tests for awex.delta module (CPU-only, no GPU required)."""

import importlib.util
import os
import sys

import pytest
import torch

# Import delta submodules directly to avoid awex/__init__.py (which pulls megatron)
_delta_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "awex", "delta"
)


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_delta_dir, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_patch = _load("awex.delta.patch", "patch.py")
# codec must precede detector (detector imports bitwise_changed_mask from it).
_mod_codec = _load("awex.delta.codec", "codec.py")
_mod_detector = _load("awex.delta.detector", "detector.py")
_mod_remap = _load("awex.delta.remap", "remap.py")

SparseWeightPatch = _mod_patch.SparseWeightPatch
DeltaResult = _mod_patch.DeltaResult
patches_to_dict = _mod_patch.patches_to_dict
dict_to_patches = _mod_patch.dict_to_patches
DeltaWeightDetector = _mod_detector.DeltaWeightDetector
remap_delta_indices = _mod_remap.remap_delta_indices
remap_mask_for_op = _mod_remap.remap_mask_for_op
_unravel_index = _mod_remap._unravel_index
_ravel_multi_index = _mod_remap._ravel_multi_index

DeltaTracker = _mod_codec.DeltaTracker
DeltaHeader = _mod_codec.DeltaHeader
DecodedDelta = _mod_codec.DecodedDelta
EncodedDelta = _mod_codec.EncodedDelta
decode_delta_payload = _mod_codec.decode_delta_payload
apply_sparse_patch_ = _mod_codec.apply_sparse_patch_
bitwise_changed_mask = _mod_codec.bitwise_changed_mask
int_view = _mod_codec.int_view
is_delta_payload = _mod_codec.is_delta_payload
reconstruct_against_base = _mod_codec.reconstruct_against_base
DELTA_HEADER_NAME = _mod_codec.DELTA_HEADER_NAME
DELTA_IDX_SUFFIX = _mod_codec.DELTA_IDX_SUFFIX
DELTA_VAL_SUFFIX = _mod_codec.DELTA_VAL_SUFFIX


class TestSparseWeightPatch:
    def test_basic(self):
        p = SparseWeightPatch(
            name="layer.0.weight",
            indices=torch.tensor([1, 5, 10], dtype=torch.int32),
            values=torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16),
        )
        assert p.num_updates == 3
        assert p.size_bytes == 3 * 4 + 3 * 2  # int32 + bf16

    def test_empty_patch(self):
        p = SparseWeightPatch(
            name="empty",
            indices=torch.empty(0, dtype=torch.int32),
            values=torch.empty(0, dtype=torch.bfloat16),
        )
        assert p.num_updates == 0
        assert p.size_bytes == 0


class TestDeltaResult:
    def test_sparsity(self):
        r = DeltaResult(total_elements=1000, changed_elements=20, step=1)
        assert r.sparsity == pytest.approx(0.98)

    def test_sparsity_zero_elements(self):
        r = DeltaResult(total_elements=0, changed_elements=0, step=1)
        assert r.sparsity == 1.0

    def test_should_use_delta(self):
        patches = [
            SparseWeightPatch(
                name="w",
                indices=torch.tensor([0, 1], dtype=torch.int32),
                values=torch.tensor([0.1, 0.2], dtype=torch.bfloat16),
            )
        ]
        # delta = 2*4 + 2*2 = 12 bytes, full = 1000*2 = 2000 bytes
        r = DeltaResult(
            patches=patches, total_elements=1000, changed_elements=2, step=1
        )
        assert r.should_use_delta(threshold=0.5) is True

    def test_should_not_use_delta_when_too_large(self):
        indices = torch.arange(800, dtype=torch.int32)
        values = torch.randn(800, dtype=torch.bfloat16)
        patches = [SparseWeightPatch(name="w", indices=indices, values=values)]
        # delta = 800*4 + 800*2 = 4800, full = 1000*2 = 2000
        r = DeltaResult(
            patches=patches, total_elements=1000, changed_elements=800, step=1
        )
        assert r.should_use_delta(threshold=0.5) is False

    def test_summary(self):
        r = DeltaResult(total_elements=10000, changed_elements=100, step=5)
        s = r.summary()
        assert "step=5" in s
        assert "sparsity=0.99" in s


class TestPatchesSerialization:
    def test_round_trip(self):
        patches = [
            SparseWeightPatch(
                "a",
                torch.tensor([0, 2], dtype=torch.int32),
                torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            ),
            SparseWeightPatch(
                "b",
                torch.tensor([5], dtype=torch.int32),
                torch.tensor([3.0], dtype=torch.bfloat16),
            ),
        ]
        d = patches_to_dict(patches)
        restored = dict_to_patches(d)
        assert len(restored) == 2
        assert restored[0].name == "a"
        assert torch.equal(restored[0].indices, patches[0].indices)

    def test_merge_duplicates(self):
        """patches_to_dict should merge patches with same name."""
        patches = [
            SparseWeightPatch(
                "w",
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            ),
            SparseWeightPatch(
                "w",
                torch.tensor([5, 6], dtype=torch.int32),
                torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            ),
        ]
        d = patches_to_dict(patches)
        assert len(d) == 1
        indices, values = d["w"]
        assert indices.numel() == 4
        assert values.numel() == 4
        assert torch.equal(indices, torch.tensor([0, 1, 5, 6], dtype=torch.int32))


class TestDeltaWeightDetector:
    def _make_model_params(self, shapes):
        """Create a list of (name, tensor) pairs."""
        params = []
        for i, shape in enumerate(shapes):
            t = torch.randn(shape, dtype=torch.bfloat16)
            params.append((f"layer.{i}.weight", t))
        return params

    def test_init_snapshot(self):
        detector = DeltaWeightDetector()
        params = self._make_model_params([(4, 4), (8,)])
        detector.init_snapshot(iter(params))
        assert detector.initialized
        assert detector.num_params == 2

    def test_no_change(self):
        detector = DeltaWeightDetector()
        params = self._make_model_params([(4, 4)])
        detector.init_snapshot(iter(params))
        # Same params → no delta
        result = detector.compute_delta(iter(params))
        assert result.changed_elements == 0
        assert result.sparsity == 1.0
        assert len(result.patches) == 0

    def test_all_changed(self):
        detector = DeltaWeightDetector()
        params = self._make_model_params([(4, 4)])
        detector.init_snapshot(iter(params))
        # Change all weights
        new_params = [(name, torch.randn_like(t)) for name, t in params]
        result = detector.compute_delta(iter(new_params))
        assert result.changed_elements > 0
        assert result.sparsity < 1.0

    def test_partial_change(self):
        detector = DeltaWeightDetector()
        t = torch.zeros(100, dtype=torch.bfloat16)
        detector.init_snapshot(iter([("w", t)]))
        # Change only 3 elements
        t2 = t.clone()
        t2[10] = 1.0
        t2[50] = 2.0
        t2[99] = 3.0
        result = detector.compute_delta(iter([("w", t2)]))
        assert result.changed_elements == 3
        assert result.total_elements == 100
        assert result.sparsity == pytest.approx(0.97)
        patch = result.patches[0]
        assert torch.equal(patch.indices, torch.tensor([10, 50, 99], dtype=torch.int32))

    def test_snapshot_always_updated(self):
        """Snapshot should update even for unchanged params (prevent drift)."""
        detector = DeltaWeightDetector()
        t = torch.zeros(10, dtype=torch.bfloat16)
        detector.init_snapshot(iter([("w", t)]))
        # First call: no change
        result1 = detector.compute_delta(iter([("w", t)]))
        assert result1.changed_elements == 0
        # Second call: still no change (snapshot was updated)
        result2 = detector.compute_delta(iter([("w", t)]))
        assert result2.changed_elements == 0

    def test_consecutive_deltas(self):
        detector = DeltaWeightDetector()
        t = torch.zeros(10, dtype=torch.bfloat16)
        detector.init_snapshot(iter([("w", t)]))
        # Step 1: change element 0
        t1 = t.clone()
        t1[0] = 1.0
        r1 = detector.compute_delta(iter([("w", t1)]))
        assert r1.changed_elements == 1
        # Step 2: change element 5 (element 0 no longer changing)
        t2 = t1.clone()
        t2[5] = 2.0
        r2 = detector.compute_delta(iter([("w", t2)]))
        assert r2.changed_elements == 1
        assert r2.patches[0].indices.item() == 5


class TestIndexArithmetic:
    def test_unravel_ravel_roundtrip_2d(self):
        shape = (3, 4)
        flat = torch.arange(12)
        multi = _unravel_index(flat, shape)
        recovered = _ravel_multi_index(list(multi), shape)
        assert torch.equal(recovered, flat)

    def test_unravel_ravel_roundtrip_3d(self):
        shape = (2, 3, 4)
        flat = torch.arange(24)
        multi = _unravel_index(flat, shape)
        recovered = _ravel_multi_index(list(multi), shape)
        assert torch.equal(recovered, flat)

    def test_unravel_ravel_roundtrip_1d(self):
        shape = (10,)
        flat = torch.arange(10)
        multi = _unravel_index(flat, shape)
        recovered = _ravel_multi_index(list(multi), shape)
        assert torch.equal(recovered, flat)


class TestRemapDeltaIndices:
    def test_identity_remap(self):
        """Same slices → indices unchanged."""
        patch = SparseWeightPatch(
            name="w",
            indices=torch.tensor([0, 3, 7], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        )
        result = remap_delta_indices(
            patch,
            train_shape=(4, 4),
            train_slices=(slice(None), slice(None)),
            inf_slices=(slice(None), slice(None)),
            infer_shape=(4, 4),
        )
        assert result is not None
        assert torch.equal(result.indices, patch.indices)
        assert torch.equal(result.values, patch.values)

    def test_filter_out_of_range(self):
        """Indices outside train_slices should be filtered."""
        # 2D tensor [4, 4], flat index 0-15
        # index 0 = (0,0), index 3 = (0,3), index 8 = (2,0), index 12 = (3,0)
        patch = SparseWeightPatch(
            name="w",
            indices=torch.tensor([0, 3, 8, 12], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
        )
        # Only keep rows 0-1 (indices 0-7)
        result = remap_delta_indices(
            patch,
            train_shape=(4, 4),
            train_slices=(slice(0, 2), slice(None)),
            inf_slices=(slice(0, 2), slice(None)),
            infer_shape=(2, 4),
        )
        assert result is not None
        assert result.num_updates == 2  # only index 0 and 3
        assert torch.equal(result.indices, torch.tensor([0, 3], dtype=torch.int32))

    def test_remap_offset(self):
        """Indices should be remapped when train and infer slices differ."""
        # Train shard [8, 4], take rows 4-7 (second half)
        # Map to infer shard [4, 4], rows 0-3
        # flat index 16 = (4,0) in train → (0,0) in infer = flat 0
        # flat index 20 = (5,0) in train → (1,0) in infer = flat 4
        patch = SparseWeightPatch(
            name="w",
            indices=torch.tensor([16, 20], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        )
        result = remap_delta_indices(
            patch,
            train_shape=(8, 4),
            train_slices=(slice(4, 8), slice(None)),
            inf_slices=(slice(0, 4), slice(None)),
            infer_shape=(4, 4),
        )
        assert result is not None
        assert torch.equal(result.indices, torch.tensor([0, 4], dtype=torch.int32))

    def test_empty_overlap(self):
        """No indices in overlap → return None."""
        patch = SparseWeightPatch(
            name="w",
            indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        )
        # All indices are in rows 0, but we only want rows 2-3
        result = remap_delta_indices(
            patch,
            train_shape=(4, 4),
            train_slices=(slice(2, 4), slice(None)),
            inf_slices=(slice(0, 2), slice(None)),
            infer_shape=(2, 4),
        )
        assert result is None

    def test_1d_tensor(self):
        """1D tensor remap."""
        patch = SparseWeightPatch(
            name="bias",
            indices=torch.tensor([2, 5, 8], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        )
        # Take elements 4-7 of a [10] tensor → map to [4] tensor
        result = remap_delta_indices(
            patch,
            train_shape=(10,),
            train_slices=(slice(4, 8),),
            inf_slices=(slice(0, 4),),
            infer_shape=(4,),
        )
        assert result is not None
        assert result.num_updates == 1  # only index 5 is in [4,8)
        assert result.indices.item() == 1  # 5 - 4 + 0 = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# remap_mask_for_op: per-op entry point the transport send path calls
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _make_op(train_slices, inf_slices, infer_shape, name="w"):
    """Mock CommunicationOperation: only the fields remap_mask_for_op reads."""
    return SimpleNamespace(
        send_shard_meta=SimpleNamespace(name=name),
        recv_shard_meta=SimpleNamespace(name=name, shape=tuple(infer_shape)),
        train_slices=tuple(train_slices),
        inf_slices=tuple(inf_slices),
    )


class TestRemapMaskForOp:
    def test_mask_to_patch_2d_tp_split(self):
        # train shard [8,4]; this op sends rows 4-7 to an infer shard [4,4].
        train = torch.zeros(8, 4, dtype=torch.bfloat16)
        train[4, 0] = 1.0  # flat 16 -> infer (0,0)=0
        train[5, 1] = 2.0  # flat 21 -> infer (1,1)=5
        train[0, 0] = 9.0  # row 0, NOT in this op's overlap -> dropped
        mask = train != 0
        op = _make_op((slice(4, 8), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        patch = remap_mask_for_op("w", mask, train, (8, 4), op)
        assert patch is not None
        assert torch.equal(patch.indices.sort().values,
                           torch.tensor([0, 5], dtype=torch.int32))
        # values gathered from train at the changed (in-overlap) positions
        assert set(patch.values.float().tolist()) == {1.0, 2.0}

    def test_mask_no_overlap_returns_none(self):
        train = torch.zeros(4, 4, dtype=torch.bfloat16)
        train[0, 0] = 1.0  # row 0, op wants rows 2-3
        mask = train != 0
        op = _make_op((slice(2, 4), slice(None)), (slice(0, 2), slice(None)), (2, 4))
        assert remap_mask_for_op("w", mask, train, (4, 4), op) is None

    def test_mask_empty_returns_none(self):
        train = torch.zeros(4, 4, dtype=torch.bfloat16)
        mask = train != 0  # all False
        op = _make_op((slice(None), slice(None)), (slice(None), slice(None)), (4, 4))
        assert remap_mask_for_op("w", mask, train, (4, 4), op) is None

    def test_mask_1d_identity(self):
        train = torch.zeros(10, dtype=torch.bfloat16)
        train[3] = 5.0
        mask = train != 0
        op = _make_op((slice(None),), (slice(None),), (10,))
        patch = remap_mask_for_op("b", mask, train, (10,), op)
        assert patch is not None
        assert patch.indices.item() == 3
        assert patch.values.item() == 5.0

    def test_reconstruct_equals_dense_after_remap(self):
        # End-to-end: apply remapped patch onto an infer shard == direct slice copy.
        torch.manual_seed(0)
        train = torch.randn(8, 4, dtype=torch.bfloat16)
        prev = train.clone()
        train[4, 2] += 1.0
        train[6, 0] += 1.0
        train[2, 1] += 1.0  # row 2 not in overlap (rows 4-7)
        mask = bitwise_changed_mask(train, prev)
        op = _make_op((slice(4, 8), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        patch = remap_mask_for_op("w", mask, train, (8, 4), op)
        # infer shard starts as the old train rows 4-7
        infer = prev[4:8].clone()
        apply_sparse_patch_(infer, patch.indices, patch.values)
        assert torch.equal(infer, train[4:8])  # matches new train rows 4-7


class TestRemapGuards:
    def test_strided_slice_raises(self):
        patch = SparseWeightPatch(
            name="w",
            indices=torch.tensor([0], dtype=torch.int32),
            values=torch.tensor([1.0], dtype=torch.bfloat16),
        )
        with pytest.raises(NotImplementedError, match="strided"):
            remap_delta_indices(
                patch, train_shape=(10,),
                train_slices=(slice(0, 8, 2),), inf_slices=(slice(0, 4),),
                infer_shape=(4,),
            )

    def test_int32_overflow_raises(self):
        # infer shard >= 2**31 elements -> int32 flat index unsafe.
        patch = SparseWeightPatch(
            name="huge",
            indices=torch.tensor([0], dtype=torch.int32),
            values=torch.tensor([1.0], dtype=torch.bfloat16),
        )
        with pytest.raises(ValueError, match="2\\*\\*31|overflow"):
            remap_delta_indices(
                patch, train_shape=(2**31 + 8,),
                train_slices=(slice(None),), inf_slices=(slice(None),),
                infer_shape=(2**31 + 8,),
            )


# ---------------------------------------------------------------------------
# delta_p2p: variable-length payload protocol (control plane + data plane)
# ---------------------------------------------------------------------------

_mod_p2p = _load("awex.delta.delta_p2p", "delta_p2p.py")
build_send_patches = _mod_p2p.build_send_patches
nnz_vector = _mod_p2p.nnz_vector
allocate_recv_buffers = _mod_p2p.allocate_recv_buffers
scatter_recv_into = _mod_p2p.scatter_recv_into
OpDeltaPayload = _mod_p2p.OpDeltaPayload


def _slice_fn(tensor, op):
    """Mock slice_tensor for the inference (recv) side."""
    return tensor[op.inf_slices]


def _transmit(payloads, recv_buffers):
    """Simulate the data-plane copy: sender payload -> receiver pre-alloc buffers.

    Asserts the control plane (nnz) pre-sized the buffers correctly (this is
    exactly what NCCL would require to stay symmetric)."""
    for p, (idx_buf, val_buf) in zip(payloads, recv_buffers, strict=True):
        assert idx_buf.numel() == p.nnz
        idx_buf.copy_(p.indices)
        val_buf.copy_(p.values)


class TestDeltaP2PProtocol:
    def test_roundtrip_two_ops_equals_dense(self):
        # train shard [8,4]; two ops to two infer shards [4,4] (rows 0-3, 4-7).
        torch.manual_seed(1)
        train = torch.randn(8, 4, dtype=torch.bfloat16)
        prev = train.clone()
        train[1, 2] += 1.0  # row 1 -> op0
        train[5, 0] += 1.0  # row 5 -> op1
        mask = bitwise_changed_mask(train, prev)
        op0 = _make_op((slice(0, 4), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        op1 = _make_op((slice(4, 8), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        ops = [op0, op1]

        # sender
        payloads = build_send_patches(ops, {"w": mask}, {"w": train})
        nnz = nnz_vector(payloads)
        assert nnz.tolist() == [1, 1]

        # receiver: each infer shard starts as old train rows
        infer = {"w_op0": prev[0:4].clone(), "w_op1": prev[4:8].clone()}
        # (in real transport recv_params is keyed by name; here both ops target
        #  the same recv name "w" on different ranks — simulate per-op targets)
        recv_buffers = allocate_recv_buffers(nnz.tolist(), train.dtype)
        _transmit(payloads, recv_buffers)

        # apply op0 onto rows-0-3 shard, op1 onto rows-4-7 shard
        scatter_recv_into({"w": infer["w_op0"]}, [op0], [recv_buffers[0]], _slice_fn)
        scatter_recv_into({"w": infer["w_op1"]}, [op1], [recv_buffers[1]], _slice_fn)
        assert torch.equal(infer["w_op0"], train[0:4])
        assert torch.equal(infer["w_op1"], train[4:8])

    def test_zero_nnz_op_keeps_slot(self):
        # op whose overlap had no change must still produce an (empty) payload,
        # so sender/receiver iterate the same op count (deadlock safety).
        train = torch.zeros(8, 4, dtype=torch.bfloat16)
        train[1, 0] = 1.0  # only row 1 -> op0 changes; op1 (rows 4-7) unchanged
        prev = torch.zeros(8, 4, dtype=torch.bfloat16)
        mask = bitwise_changed_mask(train, prev)
        op0 = _make_op((slice(0, 4), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        op1 = _make_op((slice(4, 8), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        payloads = build_send_patches([op0, op1], {"w": mask}, {"w": train})
        assert len(payloads) == 2          # both ops kept
        assert payloads[0].nnz == 1
        assert payloads[1].nnz == 0        # zero-nnz slot preserved
        nnz = nnz_vector(payloads)
        assert nnz.tolist() == [1, 0]
        # receiver pre-allocates an empty buffer for op1 and applies nothing
        recv_buffers = allocate_recv_buffers(nnz.tolist(), train.dtype)
        _transmit(payloads, recv_buffers)
        infer1 = prev[4:8].clone()
        n = scatter_recv_into({"w": infer1}, [op1], [recv_buffers[1]], _slice_fn)
        assert n == 0
        assert torch.equal(infer1, prev[4:8])  # untouched

    def test_all_zero_nnz_version(self):
        # a version where nothing changed: every op zero-nnz, still symmetric.
        train = torch.ones(8, 4, dtype=torch.bfloat16)
        mask = bitwise_changed_mask(train, train.clone())  # all False
        op0 = _make_op((slice(0, 4), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        op1 = _make_op((slice(4, 8), slice(None)), (slice(0, 4), slice(None)), (4, 4))
        payloads = build_send_patches([op0, op1], {"w": mask}, {"w": train})
        assert [p.nnz for p in payloads] == [0, 0]
        assert nnz_vector(payloads).tolist() == [0, 0]

    def test_missing_param_yields_zero_nnz_slot(self):
        # op references a param with no mask/source -> still a zero-nnz slot.
        op0 = _make_op((slice(None),), (slice(None),), (4,), name="ghost")
        payloads = build_send_patches([op0], {}, {})
        assert len(payloads) == 1
        assert payloads[0].nnz == 0


# ---------------------------------------------------------------------------
# Codec: bitwise comparison
# ---------------------------------------------------------------------------


class TestBitwiseCompare:
    def test_int_view_roundtrip_bf16(self):
        t = torch.randn(100, dtype=torch.bfloat16)
        v = int_view(t)
        assert v.dtype == torch.int16
        assert v.numel() == t.numel()

    def test_nan_bits_equal_are_unchanged(self):
        # Same NaN bit pattern must NOT be flagged as changed (float != would).
        a = torch.tensor([float("nan")], dtype=torch.bfloat16)
        b = a.clone()
        assert bool((a != b).any())  # float semantics: NaN != NaN
        assert not bool(bitwise_changed_mask(a, b).any())  # bitwise: identical

    def test_signed_zero_is_changed(self):
        pos = torch.tensor([0.0], dtype=torch.bfloat16)
        neg = torch.tensor([-0.0], dtype=torch.bfloat16)
        assert not bool((pos != neg).any())  # float: +0 == -0
        assert bool(bitwise_changed_mask(pos, neg).any())  # bits differ

    def test_real_change_detected(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = torch.tensor([1.0, 2.5, 3.0], dtype=torch.bfloat16)
        mask = bitwise_changed_mask(b, a)
        assert mask.tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Codec: DeltaHeader
# ---------------------------------------------------------------------------


class TestDeltaHeader:
    def test_roundtrip(self):
        h = DeltaHeader(payload_version=7, base_version=5, num_sparse=3, num_dense=1)
        t = h.to_tensor()
        assert t.dtype == torch.int64 and t.numel() == 6
        h2 = DeltaHeader.from_tensor(t)
        assert h2.payload_version == 7
        assert h2.base_version == 5
        assert h2.num_sparse == 3
        assert h2.num_dense == 1

    def test_bad_magic(self):
        bad = torch.zeros(6, dtype=torch.int64)
        with pytest.raises(ValueError, match="magic"):
            DeltaHeader.from_tensor(bad)

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            DeltaHeader.from_tensor(torch.zeros(3, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Codec: DeltaTracker encode/decode round-trip
# ---------------------------------------------------------------------------


def _named(d):
    return list(d.items())


def _apply_decoded_to_base(decoded, base):
    """Apply a DecodedDelta onto a copy of ``base``, return the result dict."""
    out = {name: t.clone() for name, t in base.items()}
    for name, dense in decoded.dense.items():
        out[name] = dense.clone()
    for name, indices, values in decoded.iter_sparse():
        apply_sparse_patch_(out[name], indices, values)
    return out


class TestDeltaTrackerRoundTrip:
    def test_not_seeded_raises(self):
        tracker = DeltaTracker()
        assert not tracker.seeded
        assert tracker.full_sync_reason(1) == "not_seeded"
        with pytest.raises(RuntimeError):
            tracker.encode(_named({"w": torch.randn(4, dtype=torch.bfloat16)}), 1)

    def test_seed_then_encode_reconstructs(self):
        torch.manual_seed(0)
        base = {
            "embed": torch.randn(8, 4, dtype=torch.bfloat16),
            "mlp": torch.randn(16, dtype=torch.bfloat16),
        }
        tracker = DeltaTracker()
        tracker.seed(_named(base), version=0)
        assert tracker.seeded and tracker.base_version == 0

        # Mutate a few elements.
        new = {k: v.clone() for k, v in base.items()}
        new["embed"][0, 0] = base["embed"][0, 0] + 1.0
        new["embed"][3, 2] = base["embed"][3, 2] - 2.0
        new["mlp"][7] = base["mlp"][7] + 0.5

        encoded = tracker.encode(_named(new), version=1)
        assert encoded.header.payload_version == 1
        assert encoded.header.base_version == 0
        assert encoded.changed_elements == 3
        assert is_delta_payload(encoded.names)

        named_tensors = dict(zip(encoded.names, encoded.tensors))
        decoded = decode_delta_payload(named_tensors)
        reconstructed = _apply_decoded_to_base(decoded, base)
        for k in base:
            assert torch.equal(reconstructed[k], new[k]), k

    def test_unchanged_param_omitted(self):
        base = {
            "a": torch.ones(10, dtype=torch.bfloat16),
            "b": torch.ones(10, dtype=torch.bfloat16),
        }
        tracker = DeltaTracker()
        tracker.seed(_named(base), 0)
        new = {k: v.clone() for k, v in base.items()}
        new["a"][0] = 5.0  # only "a" changes
        encoded = tracker.encode(_named(new), 1)
        assert encoded.num_sparse == 1
        assert encoded.num_unchanged == 1
        named_tensors = dict(zip(encoded.names, encoded.tensors))
        # "b" must not appear in the payload at all.
        assert not any(n.startswith("b") for n in named_tensors)
        decoded = decode_delta_payload(named_tensors)
        recon = _apply_decoded_to_base(decoded, base)
        assert torch.equal(recon["b"], base["b"])
        assert torch.equal(recon["a"], new["a"])

    def test_dense_fallback_when_too_dense(self):
        # All elements change -> sparse would be 3x bigger -> dense fallback.
        base = {"w": torch.zeros(100, dtype=torch.bfloat16)}
        tracker = DeltaTracker()
        tracker.seed(_named(base), 0)
        new = {"w": torch.arange(100, dtype=torch.bfloat16) + 1.0}
        encoded = tracker.encode(_named(new), 1)
        assert encoded.num_dense_fallback == 1
        assert encoded.num_sparse == 0
        named_tensors = dict(zip(encoded.names, encoded.tensors))
        assert "w" in named_tensors  # dense entry under plain name
        decoded = decode_delta_payload(named_tensors)
        recon = _apply_decoded_to_base(decoded, base)
        assert torch.equal(recon["w"], new["w"])

    def test_snapshot_refresh_chains_versions(self):
        base = {"w": torch.zeros(50, dtype=torch.bfloat16)}
        tracker = DeltaTracker()
        tracker.seed(_named(base), 0)
        cur = base["w"].clone()

        # v1: change index 0
        cur[0] = 1.0
        e1 = tracker.encode([("w", cur)], 1)
        assert e1.changed_elements == 1
        assert tracker.base_version == 1

        # v2: change index 1 only (index 0 already in snapshot -> not re-sent)
        cur[1] = 2.0
        e2 = tracker.encode([("w", cur)], 2)
        assert e2.changed_elements == 1
        d2 = decode_delta_payload(dict(zip(e2.names, e2.tensors)))
        # Applying v2 delta onto v1 state reproduces cur.
        v1_state = base["w"].clone()
        v1_state[0] = 1.0
        recon = _apply_decoded_to_base(d2, {"w": v1_state})
        assert torch.equal(recon["w"], cur)

    def test_anchor_interval_forces_full(self):
        base = {"w": torch.zeros(10, dtype=torch.bfloat16)}
        tracker = DeltaTracker(anchor_interval=2)
        tracker.seed(_named(base), 0)
        cur = base["w"].clone()
        cur[0] = 1.0
        tracker.encode([("w", cur)], 1)  # delta 1
        assert tracker.full_sync_reason(2) is None
        cur[1] = 1.0
        tracker.encode([("w", cur)], 2)  # delta 2 -> reaches interval
        reason = tracker.full_sync_reason(3)
        assert reason is not None and "anchor_interval" in reason

    def test_request_full_sync(self):
        base = {"w": torch.zeros(10, dtype=torch.bfloat16)}
        tracker = DeltaTracker()
        tracker.seed(_named(base), 0)
        tracker.request_full_sync("reader_mismatch")
        reason = tracker.full_sync_reason(1)
        assert reason is not None and "reader_mismatch" in reason
        # After re-seed the flag clears.
        tracker.seed(_named(base), 1)
        assert tracker.full_sync_reason(2) is None

    def test_tied_params_share_delta(self):
        shared = torch.zeros(20, dtype=torch.bfloat16)
        base = {"embed_tokens": shared, "lm_head": shared}  # same storage
        tracker = DeltaTracker()
        tracker.seed(_named(base), 0)
        shared[3] = 9.0
        encoded = tracker.encode([("embed_tokens", shared), ("lm_head", shared)], 1)
        # Both names emitted, each as a sparse patch of the single change.
        assert encoded.num_sparse == 2
        named_tensors = dict(zip(encoded.names, encoded.tensors))
        decoded = decode_delta_payload(named_tensors)
        assert "embed_tokens" in decoded.sparse
        assert "lm_head" in decoded.sparse


# ---------------------------------------------------------------------------
# Codec: scatter apply equivalence
# ---------------------------------------------------------------------------


class TestApplySparsePatch:
    def test_scatter_equals_dense_copy(self):
        base = torch.zeros(64, dtype=torch.bfloat16)
        full_new = base.clone()
        idx = torch.tensor([1, 7, 30, 63], dtype=torch.int32)
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
        full_new[idx.long()] = vals

        scattered = base.clone()
        apply_sparse_patch_(scattered, idx, vals)
        assert torch.equal(scattered, full_new)

    def test_apply_into_2d_view(self):
        target = torch.zeros(4, 4, dtype=torch.bfloat16)
        idx = torch.tensor([0, 5, 15], dtype=torch.int32)  # flat indices
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        apply_sparse_patch_(target, idx, vals)
        assert target[0, 0] == 1.0
        assert target[1, 1] == 2.0
        assert target[3, 3] == 3.0

    def test_apply_into_noncontiguous(self):
        full = torch.zeros(4, 8, dtype=torch.bfloat16)
        view = full[:, ::2]  # non-contiguous, shape (4, 4)
        idx = torch.tensor([0, 5], dtype=torch.int32)
        vals = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        apply_sparse_patch_(view, idx, vals)
        assert view[0, 0] == 1.0
        assert view[1, 1] == 2.0

    def test_empty_patch_noop(self):
        target = torch.ones(10, dtype=torch.bfloat16)
        apply_sparse_patch_(
            target,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.bfloat16),
        )
        assert torch.equal(target, torch.ones(10, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# Codec: payload validation
# ---------------------------------------------------------------------------


class TestDecodeValidation:
    def test_missing_header_raises(self):
        with pytest.raises(ValueError, match="header"):
            decode_delta_payload({"w": torch.zeros(3)})

    def test_unpaired_sparse_raises(self):
        h = DeltaHeader(payload_version=1, base_version=0, num_sparse=1, num_dense=0)
        payload = {
            DELTA_HEADER_NAME: h.to_tensor(),
            "w" + DELTA_IDX_SUFFIX: torch.tensor([0], dtype=torch.int32),
            # missing matching @delta_val
        }
        with pytest.raises(ValueError, match="unpaired"):
            decode_delta_payload(payload)

    def test_count_mismatch_raises(self):
        # header claims 2 sparse but only 1 present.
        h = DeltaHeader(payload_version=1, base_version=0, num_sparse=2, num_dense=0)
        payload = {
            DELTA_HEADER_NAME: h.to_tensor(),
            "w" + DELTA_IDX_SUFFIX: torch.tensor([0], dtype=torch.int32),
            "w" + DELTA_VAL_SUFFIX: torch.tensor([1.0], dtype=torch.bfloat16),
        }
        with pytest.raises(ValueError, match="count mismatch"):
            decode_delta_payload(payload)


# ---------------------------------------------------------------------------
# End-to-end: writer (DeltaTracker) -> payload -> reader base reconstruction.
#
# The reader half drives the SAME production function the real reader uses
# (codec.reconstruct_against_base), so these are true end-to-end checks of the
# writer/reader contract rather than a parallel reimplementation.
# ---------------------------------------------------------------------------


class _ReaderBaseSim:
    """Reader stand-in: version-chain bookkeeping around the production
    ``reconstruct_against_base`` (device='cpu')."""

    def __init__(self):
        self.base = {}
        self.base_version = None

    def receive(self, names, tensors, step_id):
        named = dict(zip(names, tensors))
        if not is_delta_payload(named):
            # Dense full sync seeds the base.
            self.base = {k: v.clone() for k, v in named.items()}
            self.base_version = step_id
            return {k: v.clone() for k, v in named.items()}

        decoded = decode_delta_payload(named)
        assert decoded.header.base_version == self.base_version, (
            f"version chain broken: payload base {decoded.header.base_version} "
            f"!= reader base {self.base_version}"
        )
        result, _counts = reconstruct_against_base(self.base, decoded, "cpu")
        self.base_version = decoded.header.payload_version
        return result


class TestWriterReaderLoop:
    def test_multi_version_loop_with_fallback(self):
        torch.manual_seed(1)
        # True weights the writer holds, evolving each step.
        weights = {
            "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.bfloat16),
            "model.layers.0.mlp.weight": torch.randn(64, dtype=torch.bfloat16),
            "model.layers.0.attn.weight": torch.randn(16, 4, dtype=torch.bfloat16),
        }
        tracker = DeltaTracker(sparse_bytes_ratio=0.9)
        reader = _ReaderBaseSim()

        # Version 0: full sync (writer not seeded -> dense path + seed).
        assert tracker.full_sync_reason(0) == "not_seeded"
        names0 = list(weights.keys())
        tensors0 = list(weights.values())
        tracker.seed(zip(names0, tensors0), 0)
        recon0 = reader.receive(names0, tensors0, 0)
        for k in weights:
            assert torch.equal(recon0[k], weights[k])

        # Versions 1..4: sparse deltas of a few elements each.
        for step in range(1, 5):
            # Mutate a handful of elements in two of the three tensors.
            weights["model.embed_tokens.weight"][step, 0] += 1.0
            weights["model.layers.0.attn.weight"][0, step % 4] -= 0.5
            assert tracker.full_sync_reason(step) is None
            encoded = tracker.encode(_named(weights), step)
            assert encoded.num_sparse >= 1
            recon = reader.receive(encoded.names, encoded.tensors, step)
            for k in weights:
                assert torch.equal(recon[k], weights[k]), (step, k)

        # Version 5: a fully-changed tensor triggers dense fallback inside the
        # same delta payload; reader must still reconstruct exactly.
        weights["model.layers.0.mlp.weight"] = (
            torch.arange(64, dtype=torch.bfloat16) + 100.0
        )
        encoded = tracker.encode(_named(weights), 5)
        assert encoded.num_dense_fallback >= 1
        recon = reader.receive(encoded.names, encoded.tensors, 5)
        for k in weights:
            assert torch.equal(recon[k], weights[k]), k

    def test_no_change_version_is_noop(self):
        weights = {"w": torch.randn(50, dtype=torch.bfloat16)}
        tracker = DeltaTracker()
        reader = _ReaderBaseSim()
        names = list(weights.keys())
        tracker.seed(zip(names, weights.values()), 0)
        reader.receive(names, list(weights.values()), 0)
        # No mutation: encode emits an empty (header-only) delta.
        encoded = tracker.encode(_named(weights), 1)
        assert encoded.changed_elements == 0
        assert encoded.num_unchanged == 1
        recon = reader.receive(encoded.names, encoded.tensors, 1)
        assert torch.equal(recon["w"], weights["w"])

    def test_anchor_full_sync_reseeds_both_sides(self):
        weights = {"w": torch.zeros(20, dtype=torch.bfloat16)}
        tracker = DeltaTracker(anchor_interval=1)
        reader = _ReaderBaseSim()
        names = list(weights.keys())
        tracker.seed(zip(names, weights.values()), 0)
        reader.receive(names, list(weights.values()), 0)

        # v1: one delta, reaches anchor_interval=1.
        weights["w"][0] = 1.0
        encoded = tracker.encode(_named(weights), 1)
        reader.receive(encoded.names, encoded.tensors, 1)

        # v2: tracker forces full sync -> writer ships dense + reseeds.
        assert tracker.full_sync_reason(2) is not None
        weights["w"][1] = 2.0
        tracker.seed(_named(weights), 2)  # writer full-sync path
        recon = reader.receive(list(weights.keys()), list(weights.values()), 2)
        assert torch.equal(recon["w"], weights["w"])
        assert reader.base_version == 2
        # Subsequent delta chains from the new anchor.
        weights["w"][2] = 3.0
        encoded = tracker.encode(_named(weights), 3)
        recon = reader.receive(encoded.names, encoded.tensors, 3)
        assert torch.equal(recon["w"], weights["w"])


# ---------------------------------------------------------------------------
# reconstruct_against_base: the production reader-reconstruction function.
# ---------------------------------------------------------------------------


def _decoded_from(names, tensors):
    return decode_delta_payload(dict(zip(names, tensors)))


class TestReconstructAgainstBase:
    def _seed_tracker_base(self, weights):
        tracker = DeltaTracker()
        tracker.seed(_named(weights), 0)
        base = {k: v.clone() for k, v in weights.items()}
        return tracker, base

    def test_unchanged_dense_sparse_mix(self):
        weights = {
            "a": torch.zeros(10, dtype=torch.bfloat16),
            "b": torch.zeros(64, dtype=torch.bfloat16),  # will go dense fallback
            "c": torch.zeros(10, dtype=torch.bfloat16),  # stays unchanged
        }
        tracker, base = self._seed_tracker_base(weights)
        weights["a"][3] = 1.0  # sparse
        weights["b"] = torch.arange(64, dtype=torch.bfloat16) + 1.0  # dense fb
        encoded = tracker.encode(_named(weights), 1)
        decoded = _decoded_from(encoded.names, encoded.tensors)
        result, counts = reconstruct_against_base(base, decoded, "cpu")
        assert counts == {"sparse": 1, "dense": 1, "unchanged": 1}
        for k in weights:
            assert torch.equal(result[k], weights[k]), k
        # base refreshed in place to the new version's values.
        for k in weights:
            assert torch.equal(base[k], weights[k]), k

    def test_sparse_absent_from_base_raises(self):
        # A sparse patch for a name the base never had: full shape unknown,
        # must raise (never dead-reckon from indices.max()).
        base = {"known": torch.zeros(10, dtype=torch.bfloat16)}
        decoded = DecodedDelta(
            header=DeltaHeader(
                payload_version=1, base_version=0, num_sparse=1, num_dense=0
            ),
            sparse={
                "ghost": (
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([1.0], dtype=torch.bfloat16),
                )
            },
        )
        with pytest.raises(ValueError, match="absent from base"):
            reconstruct_against_base(base, decoded, "cpu")

    def test_dense_first_seen_is_adopted(self):
        base = {"a": torch.zeros(4, dtype=torch.bfloat16)}
        new = torch.arange(6, dtype=torch.bfloat16)
        decoded = DecodedDelta(
            header=DeltaHeader(
                payload_version=1, base_version=0, num_sparse=0, num_dense=1
            ),
            dense={"b_new": new.clone()},
        )
        result, counts = reconstruct_against_base(base, decoded, "cpu")
        assert torch.equal(result["b_new"], new)
        assert "b_new" in base and torch.equal(base["b_new"], new)
        # "a" unchanged this version is still restored.
        assert torch.equal(result["a"], torch.zeros(4, dtype=torch.bfloat16))

    def test_result_does_not_alias_base(self):
        # Mutating the result must not corrupt the base (independent storage).
        base = {"w": torch.zeros(8, dtype=torch.bfloat16)}
        decoded = DecodedDelta(
            header=DeltaHeader(
                payload_version=1, base_version=0, num_sparse=0, num_dense=0
            ),
        )
        result, _ = reconstruct_against_base(base, decoded, "cpu")
        result["w"][0] = 99.0
        assert base["w"][0] == 0.0  # base untouched
