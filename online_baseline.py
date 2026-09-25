"""Causal, sample-by-sample version of helpers_detection.rolling_baseline, for live detection.

pandas' rolling().median() needs the whole signal up front; this computes exactly the same numbers one sample at a
time. For every new sample x:
    med   = median of the last `win` samples (fewer at the start, like min_periods=1)
    sigma = 1.4826 * median of the last `win` values of |x - med|
Each rolling median is two heaps (lower half / upper half). Samples that fall out of the window are removed lazily,
by sample index, so an update costs O(log win) instead of O(win) (win = 300 000 samples for 10 min at 500 Hz).
"""
import heapq


class RollingMedian:
    """Median of the last `win` values pushed. Gives the same result as pandas' rolling(win, min_periods=1).median()."""

    def __init__(self, win: int):
        self.win = max(1, int(win))
        self.reset()

    def reset(self):
        self._lo = []      # max-heap of the lower half, stored as (-value, index)
        self._hi = []      # min-heap of the upper half, stored as (value, index)
        self._side = {}    # index -> 0 if the sample is in _lo, 1 if in _hi (only samples still in the window)
        self._n_lo = 0     # number of in-window samples in each heap (the heaps also hold expired ones)
        self._n_hi = 0
        self._n = 0        # number of samples pushed so far
        self._cutoff = -1  # samples with index <= _cutoff have left the window

    def _prune(self):
        """Pop expired samples sitting at the top of either heap."""
        lo, hi, cutoff = self._lo, self._hi, self._cutoff
        while lo and lo[0][1] <= cutoff:
            heapq.heappop(lo)
        while hi and hi[0][1] <= cutoff:
            heapq.heappop(hi)

    def _compact(self):
        """Drop expired samples buried inside the heaps, so memory stays bounded over hours of recording."""
        cutoff = self._cutoff
        self._lo = [e for e in self._lo if e[1] > cutoff]
        self._hi = [e for e in self._hi if e[1] > cutoff]
        heapq.heapify(self._lo)
        heapq.heapify(self._hi)

    def push(self, v: float) -> float:
        """Add one sample and return the median of the current window."""
        i = self._n
        self._n += 1

        # 1. The sample that was `win` samples ago leaves the window
        self._cutoff = i - self.win
        if self._cutoff >= 0:
            if self._side.pop(self._cutoff):
                self._n_hi -= 1
            else:
                self._n_lo -= 1
        self._prune()

        # 2. Insert the new sample in the right half
        if not self._n_lo or v <= -self._lo[0][0]:
            heapq.heappush(self._lo, (-v, i))
            self._side[i] = 0
            self._n_lo += 1
        else:
            heapq.heappush(self._hi, (v, i))
            self._side[i] = 1
            self._n_hi += 1

        # 3. Rebalance so that _lo holds as many samples as _hi, or one more
        while self._n_lo > self._n_hi + 1:
            neg_v, j = heapq.heappop(self._lo)
            heapq.heappush(self._hi, (-neg_v, j))
            self._side[j] = 1
            self._n_lo -= 1
            self._n_hi += 1
            self._prune()
        while self._n_lo < self._n_hi:
            v_, j = heapq.heappop(self._hi)
            heapq.heappush(self._lo, (-v_, j))
            self._side[j] = 0
            self._n_hi -= 1
            self._n_lo += 1
            self._prune()

        if len(self._lo) + len(self._hi) > 2 * self.win + 1000:
            self._compact()

        # 4. Median (for an even count: mean of the two middle values, as pandas does)
        if self._n_lo > self._n_hi:
            return -self._lo[0][0]
        return (-self._lo[0][0] + self._hi[0][0]) / 2


class RollingBaseline:
    """Live equivalent of helpers_detection.rolling_baseline for one channel. Call reset() at every session start,
    like the offline version does at each entry of `session_starts`."""

    def __init__(self, fs: float, window_s: float):
        win = max(1, int(window_s * fs))  # same rounding as rolling_baseline
        self._med = RollingMedian(win)
        self._mad = RollingMedian(win)

    def reset(self):
        self._med.reset()
        self._mad.reset()

    def update(self, x: float):
        """Add one sample. Returns (med, sigma) for that sample."""
        med = self._med.push(x)
        mad = self._mad.push(abs(x - med))
        return med, 1.4826 * mad
