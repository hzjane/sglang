"""Single-XPU paging of experts quantized from BF16 at load time."""

import sys
from dataclasses import replace

import pytest
import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="extra-test-1-gpu-xpu")

if is_xpu():
    from sgl_kernel.utils import is_xe2_arch

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        CudaGraphConfig,
        PhaseConfig,
    )
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs

pytestmark = pytest.mark.skipif(
    not is_xpu() or not is_xe2_arch(), reason="Native FP8 MoE requires Xe2 XPU"
)


def _load_layer(weights, *, paged, graph=False):
    args = ServerArgs(
        model_path="dummy",
        device="xpu",
        enable_paged_experts=paged,
        paged_experts_num_resident=8,
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED),
            decode=PhaseConfig(backend=Backend.FULL if graph else Backend.DISABLED),
        ),
    )
    publish(args, role="test", ranks=SpawnRanks(world_rank=0))
    with torch.device("xpu"):
        layer = FusedMoE(
            num_experts=16,
            hidden_size=256,
            intermediate_size=128,
            layer_id=0,
            top_k=4,
            params_dtype=torch.bfloat16,
            quant_config=Fp8Config(is_checkpoint_fp8_serialized=False),
            uses_expert_location=False,
        )
    for name, shard, rows in weights:
        param = getattr(layer, name)
        for expert, row in enumerate(rows):
            param.weight_loader(param, row, name, shard, expert_id=expert)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _checkpoint_weights(rng):
    return [
        (
            name,
            shard,
            torch.randn((16, *shape), generator=rng, dtype=torch.bfloat16) * 0.05,
        )
        for name, shard, shape in (
            ("w13_weight", "w1", (128, 256)),
            ("w13_weight", "w3", (128, 256)),
            ("w2_weight", "w2", (256, 128)),
        )
    ]


def _dispatch(ids, rng):
    x = torch.randn((len(ids), 256), generator=rng, dtype=torch.bfloat16).to("xpu")
    topk = StandardTopKOutput(
        topk_ids=torch.tensor(ids, dtype=torch.int32, device="xpu"),
        topk_weights=torch.tensor(
            [0.4, 0.3, 0.2, 0.1], dtype=torch.float32, device="xpu"
        ).repeat(len(ids), 1),
        router_logits=None,
    )
    return StandardDispatchOutput(x, None, topk)


@torch.inference_mode()
def test_online_fp8_multiple_waves():
    rng = torch.Generator().manual_seed(8)
    weights = _checkpoint_weights(rng)
    reset_context()
    try:
        reference = _load_layer(weights, paged=False)
        paged = _load_layer(weights, paged=True)
        assert paged.w13_weight.shape[0] == paged.w2_weight.shape[0] == 8
        assert paged.w13_weight.dtype == torch.float8_e4m3fn
        assert paged.w2_weight.dtype == torch.float8_e4m3fn
        host = paged.quant_method.store.host
        assert set(host) == {
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        }
        for name, rows in host.items():
            assert rows.is_pinned()
            torch.testing.assert_close(
                rows.view(torch.uint8),
                getattr(reference, name).detach().cpu().view(torch.uint8),
                rtol=0,
                atol=0,
            )

        # One wave with a miss, several waves with more than eight distinct experts,
        # then revisit the first route after eviction.
        first = [[0, 3, 6, 10], [1, 3, -1, 10]]
        for ids in (first, [[2, 9, 12, 15], [0, 3, 11, 14], [6, 4, -1, 13]], first):
            dispatch = _dispatch(ids, rng)
            expected = reference.quant_method.apply(reference, dispatch).hidden_states
            actual = paged.quant_method.apply(paged, dispatch).hidden_states
            # Top-1 outputs round to BF16 before the framework's FP32 reduction.
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0.01)
    finally:
        reset_context()


@torch.inference_mode()
def test_online_fp8_decode_graph_replay():
    rng = torch.Generator().manual_seed(16)
    weights = _checkpoint_weights(rng)
    reset_context()
    graph = torch.xpu.XPUGraph()
    try:
        reference = _load_layer(weights, paged=False)
        reference.quant_method.moe_runner_config = replace(
            reference.quant_method.moe_runner_config,
            no_combine=True,
            no_combine_keep_router_weight=True,
        )
        paged = _load_layer(weights, paged=True, graph=True)
        method = paged.quant_method
        residency = method.device_executor.residency

        def expected_output(dispatch):
            entries = reference.quant_method.apply(reference, dispatch).hidden_states
            return entries.sum(dim=1, dtype=torch.float32).to(torch.bfloat16)

        static = _dispatch([[0, 1, 2, 3], [4, 5, 6, 7]], rng)
        stream = torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(stream):
            method.apply(paged, static)
        stream.synchronize()
        with torch.xpu.graph(xpu_graph=graph, stream=stream):
            captured = method.apply(paged, static).hidden_states

        misses = _dispatch([[9, 2, 9, 11], [12, 15, -1, 9]], rng)
        static.hidden_states.copy_(misses.hidden_states)
        static.topk_output.topk_ids.copy_(misses.topk_output.topk_ids)
        for expected_copies in (4, 0):
            graph.replay()
            torch.xpu.synchronize()
            assert residency.count.item() == expected_copies
            expected = expected_output(misses)
            torch.testing.assert_close(captured, expected, atol=5e-4, rtol=0.01)

        # A larger step uses host-planned waves, grows the eager workspace, and
        # hands the new residency back to the next replay.
        prefill = _dispatch([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]] * 5, rng)
        actual = method.apply(paged, prefill).hidden_states
        expected = expected_output(prefill)
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0.01)

        for ids in ([[0, 15, 5, 14], [4, 12, -1, 3]], [[-1] * 4] * 2):
            dispatch = _dispatch(ids, rng)
            static.hidden_states.copy_(dispatch.hidden_states)
            static.topk_output.topk_ids.copy_(dispatch.topk_output.topk_ids)
            graph.replay()
            torch.xpu.synchronize()
            expected = expected_output(dispatch)
            torch.testing.assert_close(captured, expected, atol=5e-4, rtol=0.01)
            slots = residency.slot_expert.cpu().long()
            for name, host in method.store.host.items():
                table = getattr(paged, name).detach().cpu()
                actual_bytes = table.reshape(table.shape[0], -1).view(torch.uint8)
                host_bytes = host.reshape(host.shape[0], -1).view(torch.uint8)
                assert torch.equal(actual_bytes, host_bytes[slots])
    finally:
        torch.xpu.synchronize()
        graph.reset()
        reset_context()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
