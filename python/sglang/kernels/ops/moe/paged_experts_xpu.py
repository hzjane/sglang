"""Device-planned expert paging from XPU pinned host memory."""

import triton
import triton.language as tl


@triton.jit
def _decide(
    TopK,
    Weights,
    Step,
    SlotExpert,
    ExpertSlot,
    LastUse,
    Src,
    Dst,
    Count,
    RunIds,
    RunWeights,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    entry = tl.arange(0, BLOCK_T)
    slot = tl.arange(0, BLOCK_K)
    routed = tl.load(TopK + entry, entry < T, other=-1)
    valid = (routed >= 0) & (routed < E)
    resident = tl.load(ExpertSlot + routed, valid, other=-1)
    hit = resident >= 0
    keep = tl.sum((slot[:, None] == resident[None, :]).to(tl.int32), 1) > 0
    first = tl.min(
        tl.where(routed[:, None] == routed[None, :], entry[None, :], BLOCK_T),
        1,
    )
    miss = valid & ~hit & (entry == first)
    count = tl.sum(miss.to(tl.int32), 0)
    stamp = tl.load(Step) + 1
    mapped = resident

    if count > 0:
        # Sort unused slots by (last use, slot id). Assign one victim to each
        # distinct missing expert. T <= K guarantees enough victims.
        age = tl.load(LastUse + slot, slot < K, other=0)
        priority = age.to(tl.uint64) * BLOCK_K + slot.to(tl.uint64)
        priority = tl.where((slot < K) & ~keep, priority, 0xFFFFFFFFFFFFFFFF)
        victims = (tl.sort(priority, descending=False) % BLOCK_K).to(tl.int32)
        page = tl.cumsum(miss.to(tl.int32), 0) - 1
        new_slot = tl.gather(victims, tl.maximum(page, 0), axis=0)
        evicted = tl.load(SlotExpert + new_slot, miss, other=-1)
        # Old and new experts are disjoint because a miss was absent.
        tl.store(ExpertSlot + evicted, -1, miss & (evicted >= 0))
        tl.store(ExpertSlot + routed, new_slot, miss)
        tl.store(SlotExpert + new_slot, routed, miss)
        tl.store(LastUse + new_slot, stamp, miss)
        tl.store(Src + page, routed, miss)
        tl.store(Dst + page, new_slot, miss)
        missing_slot = tl.gather(new_slot, first, axis=0)
        mapped = tl.where(hit, resident, missing_slot)

    tl.store(LastUse + slot, stamp, (slot < K) & keep)
    tl.store(Step, stamp)
    tl.store(Count, count)
    weights = tl.load(Weights + entry, entry < T, other=0)
    active = valid & (mapped >= 0)
    tl.store(RunIds + entry, tl.where(active, mapped, 0), entry < T)
    tl.store(RunWeights + entry, tl.where(active, weights, 0), entry < T)


def paged_experts_decide(
    topk,
    weights,
    step,
    slot_expert,
    expert_slot,
    last_use,
    src,
    dst,
    count,
    ids,
    scores,
):
    entries, experts, slots = topk.numel(), expert_slot.numel(), slot_expert.numel()
    _decide[(1,)](
        topk,
        weights,
        step,
        slot_expert,
        expert_slot,
        last_use,
        src,
        dst,
        count,
        ids,
        scores,
        entries,
        experts,
        slots,
        triton.next_power_of_2(max(entries, 1)),
        triton.next_power_of_2(slots),
        num_warps=4,
    )


@triton.jit
def _gather(
    Hosts,
    Tables,
    Src,
    Dst,
    Count,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    count = tl.load(Count)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    stride = tl.num_programs(0) * BLOCK
    for field in tl.static_range(len(ROW_WORDS)):
        words = ROW_WORDS[field]
        total = count * words
        host = tl.load(Hosts + field).to(tl.pointer_type(tl.uint32))
        table = tl.load(Tables + field).to(tl.pointer_type(tl.uint32))
        for start in range(0, total, stride):
            position = start + offset
            page = position // words
            word = position % words
            active = position < total
            expert = tl.load(Src + page, active, other=0)
            slot = tl.load(Dst + page, active, other=0)
            value = tl.load(host + expert.to(tl.int64) * words + word, active, other=0)
            tl.store(table + slot.to(tl.int64) * words + word, value, active)


def paged_experts_gather(hosts, tables, row_words, src, dst, count, blocks):
    _gather[(blocks,)](
        hosts, tables, src, dst, count, row_words, BLOCK=512, num_warps=4
    )
