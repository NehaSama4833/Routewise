"""Per-backend prefix cache over block ids, with LRU eviction.

A prompt is a list of block ids. Two prompts share a prefix when their leading block ids are equal
(ids are constructed so that equal id at position i implies equal content up to i).

Eviction detail: blocks are inserted in REVERSE order, so the first block of a prompt is the most
recently used and the tail is evicted first. Evicting the head first would destroy a prefix's
reuse value while its tail still sits in memory (real systems evict leaves first for this reason).
"""

from collections import OrderedDict


class PrefixCache:
    def __init__(self, capacity_blocks: int):
        if capacity_blocks < 1:
            raise ValueError("capacity_blocks must be >= 1")
        self.capacity = capacity_blocks
        self._blocks: OrderedDict = OrderedDict()  # oldest first

    def __len__(self) -> int:
        return len(self._blocks)

    def match(self, blocks: list) -> int:
        """Number of leading blocks present. Does not change recency."""
        n = 0
        for b in blocks:
            if b not in self._blocks:
                break
            n += 1
        return n

    def insert(self, blocks: list) -> None:
        for b in reversed(blocks):
            if b in self._blocks:
                self._blocks.move_to_end(b)
            else:
                self._blocks[b] = None
        while len(self._blocks) > self.capacity:
            self._blocks.popitem(last=False)
