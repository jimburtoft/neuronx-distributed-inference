"""
Pre-computed window schedule for RollingForcing on Neuron.

This module computes the exact schedule for all 22 model calls (11 windows × 2 calls each)
in a 21-frame T2V RollingForcing generation. Every value is a deterministic function of window
index, eliminating all .item() calls and dynamic tensor operations.

Configuration (from rolling_forcing_dmd.yaml + default_config.yaml):
    num_frame_per_block = 3
    denoising_step_list = [1000, 800, 600, 400, 200]  (5 steps)
    num_frames = 21 (latent frames)
    frame_seq_length = 1560  (tokens per latent frame)
    kv_cache_capacity = 1560 * 24 = 37,440 tokens
    max_attention_size = 21 * 1560 = 32,760 tokens
    block_length = 3 * 1560 = 4,680 tokens (1 block = 3 frames)
    sink_tokens = 1 * block_length = 4,680 (anchor block = first block)

Derived:
    num_blocks = 21 // 3 = 7
    rolling_window_length_blocks = len(denoising_step_list) = 5
    window_num = num_blocks + rolling_window_length_blocks - 1 = 11
"""

from dataclasses import dataclass
from typing import List, Optional


# ─── Constants ──────────────────────────────────────────────────────────────

FRAME_SEQ_LENGTH = 1560  # tokens per latent frame
NUM_FRAME_PER_BLOCK = 3  # frames per block
BLOCK_LENGTH = NUM_FRAME_PER_BLOCK * FRAME_SEQ_LENGTH  # 4680
MAX_ATTENTION_SIZE = 21 * FRAME_SEQ_LENGTH  # 32760
KV_CACHE_CAPACITY = 24 * FRAME_SEQ_LENGTH  # 37440
SINK_TOKENS = 1 * BLOCK_LENGTH  # 4680
NUM_HEADS = 12
HEAD_DIM = 128

NUM_FRAMES = 21
NUM_BLOCKS = NUM_FRAMES // NUM_FRAME_PER_BLOCK  # 7
DENOISING_STEPS = 5  # len([1000, 800, 600, 400, 200])
ROLLING_WINDOW_LENGTH_BLOCKS = DENOISING_STEPS  # 5
WINDOW_NUM = NUM_BLOCKS + ROLLING_WINDOW_LENGTH_BLOCKS - 1  # 11
NUM_TRANSFORMER_LAYERS = 30


# ─── Data Structures ───────────────────────────────────────────────────────


@dataclass
class MainCallSchedule:
    """Schedule for the main DiT call within a window (denoising)."""

    window_index: int
    start_block: int
    end_block: int
    current_start_frame: int  # in latent frame space
    current_start_token: int  # current_start_frame * FRAME_SEQ_LENGTH
    num_input_frames: int  # (end_block + 1 - start_block) * NUM_FRAME_PER_BLOCK
    input_seq_len: int  # num_input_frames * FRAME_SEQ_LENGTH
    attention_path: str  # "A" (self-attn, no cache), "B" (cache-augmented)

    # Cache write parameters
    cache_start_token: (
        int  # = current_start_token (cache_start defaults to current_start)
    )
    cache_end_token: int  # cache_start_token + BLOCK_LENGTH
    local_start_index: int  # where in cache buffer the block is written
    local_end_index: int  # local_start_index + BLOCK_LENGTH
    global_end_index_before: int  # kv_cache["global_end_index"] before this call
    global_end_index_after: int  # kv_cache["global_end_index"] after this call
    local_end_index_after: int  # kv_cache["local_end_index"] after this call
    is_first_block: bool  # True if local_start_index == 0 (anchor stored un-roped)
    num_new_tokens: int  # cache_end - global_end_index_before

    # Attention dimensions (for Path B only, None for Path A)
    query_length: Optional[int]  # = input_seq_len
    kv_length_unpadded: Optional[int]  # anchor + working + current
    kv_length_padded: (
        int  # MAX_ATTENTION_SIZE (32760) for Path B, or input_seq_len for Path A
    )

    # Working cache slice (for Path B only)
    extract_cache_start: Optional[int]  # start of working cache extract
    extract_cache_end: Optional[
        int
    ]  # end of working cache extract (= local_start_index)
    working_cache_length: Optional[int]  # extract_cache_end - extract_cache_start

    # RoPE parameters
    current_start_frame_for_rope: (
        int  # current_start_token // (H * W) where H*W is spatial size per frame
    )
    rope_start_frame_anchor: Optional[
        int
    ]  # Re-RoPE start_frame for anchor block (Path B only)

    # Grid sizes (for patch embedding output)
    grid_f: int  # num_input_frames (after patch, since patch_size_t=1)
    grid_h: int  # H // patch_h = 60 // 2 = 30 (at 480 resolution) or 16 (at 256)
    grid_w: int  # W // patch_w = 104 // 2 = 52 (at 480 resolution) or 16 (at 256)


@dataclass
class UpdateCallSchedule:
    """Schedule for the cache update call within a window (re-run with t=0 to cache clean frame)."""

    window_index: int
    current_start_frame: int  # same as main call's current_start_frame
    current_start_token: int
    num_input_frames: (
        int  # NUM_FRAME_PER_BLOCK = 3 (only first block of denoised output)
    )
    input_seq_len: int  # BLOCK_LENGTH = 4680
    attention_path: str  # "A" (windows 0-4) or "C" (windows 5-10, cache-only attention)
    updating_cache: bool  # True

    # Cache write parameters
    cache_start_token: int
    cache_end_token: int
    local_start_index: int
    local_end_index: int
    global_end_index_before: int
    global_end_index_after: (
        int  # same as before (num_new_tokens=0 for update calls after window 0)
    )
    local_end_index_after: int
    is_first_block: bool
    num_new_tokens: int  # 0 for most update calls (cache_end == global_end_index)

    # Attention dimensions (for Path C)
    query_length: int  # BLOCK_LENGTH = 4680
    kv_length_unpadded: Optional[int]  # local_end_index for Path C
    kv_length_padded: int  # MAX_ATTENTION_SIZE for Path C, BLOCK_LENGTH for Path A

    # Working cache slice (for Path C: extract_cache_start..extract_cache_end)
    extract_cache_start: Optional[int]
    extract_cache_end: Optional[int]
    working_cache_length: Optional[int]

    # RoPE
    current_start_frame_for_rope: int
    rope_start_frame_anchor: Optional[
        int
    ]  # only for Path C when extract_cache_start==0

    # Grid sizes
    grid_f: int  # NUM_FRAME_PER_BLOCK = 3
    grid_h: int
    grid_w: int


@dataclass
class WindowSchedule:
    """Complete schedule for one rolling window."""

    window_index: int
    main_call: MainCallSchedule
    update_call: UpdateCallSchedule


def compute_schedule() -> List[WindowSchedule]:
    """
    Compute the complete pre-determined schedule for all 11 windows × 2 calls.

    Returns:
        List of 11 WindowSchedule objects, one per rolling window.
    """
    schedule = []

    # Track cache state across windows
    global_end_index = 0  # kv_cache["global_end_index"]
    local_end_index = 0  # kv_cache["local_end_index"]

    for window_index in range(WINDOW_NUM):
        # Window boundaries (from rolling_forcing_inference.py lines 176-180)
        start_block = max(0, window_index - ROLLING_WINDOW_LENGTH_BLOCKS + 1)
        end_block = min(NUM_BLOCKS - 1, window_index)

        current_start_frame = start_block * NUM_FRAME_PER_BLOCK
        current_end_frame = (end_block + 1) * NUM_FRAME_PER_BLOCK
        current_num_frames = current_end_frame - current_start_frame
        current_start_token = current_start_frame * FRAME_SEQ_LENGTH
        input_seq_len = current_num_frames * FRAME_SEQ_LENGTH

        # ─── MAIN CALL ─────────────────────────────────────────────────
        # cache_start defaults to current_start (line 108-109 of causal_model.py)
        cache_start_token = current_start_token
        cache_end_token = cache_start_token + BLOCK_LENGTH
        num_new_tokens_main = cache_end_token - global_end_index

        # Compute local indices (no eviction path for 21-frame generation)
        main_local_end_index = local_end_index + cache_end_token - global_end_index
        main_local_start_index = main_local_end_index - BLOCK_LENGTH
        is_first_block_main = main_local_start_index == 0

        # Determine attention path
        if main_local_start_index == 0:
            # Path A: pure self-attention, no cache interaction for attention
            attention_path_main = "A"
            query_length_main = input_seq_len
            kv_length_unpadded_main = input_seq_len  # self-attention: Q == KV
            kv_length_padded_main = input_seq_len
            extract_cache_start_main = None
            extract_cache_end_main = None
            working_cache_length_main = None
            rope_start_frame_anchor_main = None
        else:
            # Path B: cache-augmented attention
            attention_path_main = "B"
            query_length_main = input_seq_len

            # Working cache extraction (lines 267-273 of causal_model.py)
            working_cache_max_length = (
                MAX_ATTENTION_SIZE - query_length_main - BLOCK_LENGTH
            )
            extract_cache_end_m = main_local_start_index
            extract_cache_start_m = max(
                BLOCK_LENGTH, main_local_start_index - working_cache_max_length
            )
            working_cache_length_m = extract_cache_end_m - extract_cache_start_m

            # Anchor re-RoPE (lines 276-280)
            working_cache_frame_length = working_cache_length_m // FRAME_SEQ_LENGTH
            current_start_frame_for_rope = (
                current_start_frame  # current_start_token // FRAME_SEQ_LENGTH
            )
            rope_start_frame_anchor_m = (
                current_start_frame_for_rope - working_cache_frame_length - 3
            )

            # Total KV = anchor(4680) + working + current(input_seq_len)
            kv_length_unpadded_m = BLOCK_LENGTH + working_cache_length_m + input_seq_len
            kv_length_padded_main = MAX_ATTENTION_SIZE  # pad to max

            extract_cache_start_main = extract_cache_start_m
            extract_cache_end_main = extract_cache_end_m
            working_cache_length_main = working_cache_length_m
            rope_start_frame_anchor_main = rope_start_frame_anchor_m
            kv_length_unpadded_main = kv_length_unpadded_m

        # Update cache state after main call
        if num_new_tokens_main > 0:
            global_end_index_after_main = cache_end_token
            local_end_index_after_main = main_local_end_index
        else:
            global_end_index_after_main = global_end_index
            local_end_index_after_main = local_end_index

        main_call = MainCallSchedule(
            window_index=window_index,
            start_block=start_block,
            end_block=end_block,
            current_start_frame=current_start_frame,
            current_start_token=current_start_token,
            num_input_frames=current_num_frames,
            input_seq_len=input_seq_len,
            attention_path=attention_path_main,
            cache_start_token=cache_start_token,
            cache_end_token=cache_end_token,
            local_start_index=main_local_start_index,
            local_end_index=main_local_end_index,
            global_end_index_before=global_end_index,
            global_end_index_after=global_end_index_after_main,
            local_end_index_after=local_end_index_after_main,
            is_first_block=is_first_block_main,
            num_new_tokens=num_new_tokens_main,
            query_length=query_length_main,
            kv_length_unpadded=kv_length_unpadded_main,
            kv_length_padded=kv_length_padded_main,
            extract_cache_start=extract_cache_start_main,
            extract_cache_end=extract_cache_end_main,
            working_cache_length=working_cache_length_main,
            current_start_frame_for_rope=current_start_frame,
            rope_start_frame_anchor=rope_start_frame_anchor_main,
            grid_f=current_num_frames,
            grid_h=0,  # filled at runtime based on resolution
            grid_w=0,  # filled at runtime based on resolution
        )

        # Apply cache state update from main call
        global_end_index = global_end_index_after_main
        local_end_index = local_end_index_after_main

        # ─── UPDATE CALL ───────────────────────────────────────────────
        # The update call re-runs with only the first block (3 frames) of denoised output
        # at timestep=0 (context_noise=0), with updating_cache=True.
        # current_start is same as main call's current_start.
        update_input_seq_len = BLOCK_LENGTH  # only first block
        update_cache_start_token = current_start_token
        update_cache_end_token = update_cache_start_token + BLOCK_LENGTH
        update_num_new_tokens = update_cache_end_token - global_end_index
        # num_new_tokens should be 0 for update calls (cache_end == global_end_index after main)
        # because main call already updated global_end_index to cache_end_token

        # Compute local indices for update call
        update_local_end_index = (
            local_end_index + update_cache_end_token - global_end_index
        )
        update_local_start_index = update_local_end_index - BLOCK_LENGTH
        is_first_block_update = update_local_start_index == 0

        if update_local_start_index == 0:
            # Path A: pure self-attention (windows 0-4 update calls)
            attention_path_update = "A"
            kv_length_unpadded_update = update_input_seq_len
            kv_length_padded_update = update_input_seq_len
            extract_cache_start_update = None
            extract_cache_end_update = None
            working_cache_length_update = None
            rope_start_frame_anchor_update = None
        else:
            # Path C: cache-only attention (windows 5-10 update calls)
            # updating_cache=True path (lines 248-262 of causal_model.py)
            attention_path_update = "C"
            extract_cache_end_u = update_local_end_index
            extract_cache_start_u = max(0, update_local_end_index - MAX_ATTENTION_SIZE)
            working_cache_length_u = extract_cache_end_u - extract_cache_start_u

            kv_length_unpadded_update = working_cache_length_u
            kv_length_padded_update = MAX_ATTENTION_SIZE  # pad to max

            # Re-RoPE: if extract_cache_start == 0, anchor is re-roped with start_frame=0
            rope_start_frame_anchor_update = 0 if extract_cache_start_u == 0 else None

            extract_cache_start_update = extract_cache_start_u
            extract_cache_end_update = extract_cache_end_u
            working_cache_length_update = working_cache_length_u

        # Update cache state after update call
        if update_num_new_tokens > 0:
            global_end_index_after_update = update_cache_end_token
            local_end_index_after_update = update_local_end_index
        else:
            global_end_index_after_update = global_end_index
            local_end_index_after_update = local_end_index

        update_call = UpdateCallSchedule(
            window_index=window_index,
            current_start_frame=current_start_frame,
            current_start_token=current_start_token,
            num_input_frames=NUM_FRAME_PER_BLOCK,
            input_seq_len=update_input_seq_len,
            attention_path=attention_path_update,
            updating_cache=True,
            cache_start_token=update_cache_start_token,
            cache_end_token=update_cache_end_token,
            local_start_index=update_local_start_index,
            local_end_index=update_local_end_index,
            global_end_index_before=global_end_index,
            global_end_index_after=global_end_index_after_update,
            local_end_index_after=local_end_index_after_update,
            is_first_block=is_first_block_update,
            num_new_tokens=update_num_new_tokens,
            query_length=update_input_seq_len,
            kv_length_unpadded=kv_length_unpadded_update,
            kv_length_padded=kv_length_padded_update,
            extract_cache_start=extract_cache_start_update,
            extract_cache_end=extract_cache_end_update,
            working_cache_length=working_cache_length_update,
            current_start_frame_for_rope=current_start_frame,
            rope_start_frame_anchor=rope_start_frame_anchor_update,
            grid_f=NUM_FRAME_PER_BLOCK,
            grid_h=0,  # filled at runtime
            grid_w=0,  # filled at runtime
        )

        # Apply cache state update from update call
        global_end_index = global_end_index_after_update
        local_end_index = local_end_index_after_update

        schedule.append(
            WindowSchedule(
                window_index=window_index,
                main_call=main_call,
                update_call=update_call,
            )
        )

    return schedule


def get_bucket_configs(schedule: List[WindowSchedule]) -> dict:
    """
    Extract the distinct bucket configurations needed for Neuron compilation.

    Returns a dict mapping input_seq_len -> list of (window_index, call_type) tuples.
    """
    buckets = {}
    for ws in schedule:
        # Main call
        seq_len = ws.main_call.input_seq_len
        if seq_len not in buckets:
            buckets[seq_len] = []
        buckets[seq_len].append((ws.window_index, "main"))

        # Update call
        seq_len = ws.update_call.input_seq_len
        if seq_len not in buckets:
            buckets[seq_len] = []
        buckets[seq_len].append((ws.window_index, "update"))

    return buckets


def print_schedule(schedule: List[WindowSchedule]):
    """Pretty-print the full schedule for debugging."""
    print("=" * 100)
    print(
        "ROLLING FORCING WINDOW SCHEDULE — 21-frame T2V (11 windows × 2 calls = 22 total)"
    )
    print("=" * 100)

    for ws in schedule:
        mc = ws.main_call
        uc = ws.update_call

        print(f"\n{'─' * 80}")
        print(
            f"Window {ws.window_index}: blocks [{mc.start_block}..{mc.end_block}], "
            f"frames [{mc.current_start_frame}..{mc.current_start_frame + mc.num_input_frames})"
        )
        print(f"{'─' * 80}")

        # Main call
        print(f"  MAIN CALL (denoising):")
        print(
            f"    Path: {mc.attention_path} | input_frames: {mc.num_input_frames} | "
            f"input_seq_len: {mc.input_seq_len}"
        )
        print(
            f"    cache_write: [{mc.local_start_index}..{mc.local_end_index}) | "
            f"first_block: {mc.is_first_block} | new_tokens: {mc.num_new_tokens}"
        )
        print(
            f"    global_end: {mc.global_end_index_before} → {mc.global_end_index_after} | "
            f"local_end: {mc.local_end_index_after}"
        )
        if mc.attention_path == "B":
            print(
                f"    working_cache: [{mc.extract_cache_start}..{mc.extract_cache_end}) "
                f"len={mc.working_cache_length}"
            )
            print(f"    anchor re-RoPE start_frame: {mc.rope_start_frame_anchor}")
            print(
                f"    Q={mc.query_length}, KV_unpadded={mc.kv_length_unpadded}, "
                f"KV_padded={mc.kv_length_padded}"
            )
        else:
            print(f"    Q=KV={mc.query_length} (self-attention)")

        # Update call
        print(f"  UPDATE CALL (cache clean frame):")
        print(
            f"    Path: {uc.attention_path} | input_frames: {uc.num_input_frames} | "
            f"input_seq_len: {uc.input_seq_len}"
        )
        print(
            f"    cache_write: [{uc.local_start_index}..{uc.local_end_index}) | "
            f"first_block: {uc.is_first_block} | new_tokens: {uc.num_new_tokens}"
        )
        if uc.attention_path == "C":
            print(
                f"    working_cache: [{uc.extract_cache_start}..{uc.extract_cache_end}) "
                f"len={uc.working_cache_length}"
            )
            if uc.rope_start_frame_anchor is not None:
                print(f"    anchor re-RoPE start_frame: {uc.rope_start_frame_anchor}")
            print(
                f"    Q={uc.query_length}, KV_unpadded={uc.kv_length_unpadded}, "
                f"KV_padded={uc.kv_length_padded}"
            )
        else:
            print(f"    Q=KV={uc.query_length} (self-attention)")

    # Summary
    print(f"\n{'=' * 100}")
    print("BUCKET SUMMARY")
    print(f"{'=' * 100}")
    buckets = get_bucket_configs(schedule)
    for seq_len in sorted(buckets.keys()):
        frames = seq_len // FRAME_SEQ_LENGTH
        calls = buckets[seq_len]
        print(
            f"  seq_len={seq_len:6d} ({frames:2d} frames): {len(calls)} calls — "
            f"{[f'w{w}_{t}' for w, t in calls]}"
        )

    # Distinct attention configurations
    print(f"\nDISTINCT ATTENTION CONFIGURATIONS:")
    configs = set()
    for ws in schedule:
        mc = ws.main_call
        configs.add((mc.attention_path, mc.query_length, mc.kv_length_padded))
        uc = ws.update_call
        configs.add((uc.attention_path, uc.query_length, uc.kv_length_padded))

    for path, q, kv in sorted(configs):
        print(f"  Path {path}: Q={q}, KV_padded={kv}")


if __name__ == "__main__":
    schedule = compute_schedule()
    print_schedule(schedule)
