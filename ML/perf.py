"""Lightweight perf-logger for measuring per-event timings.

Designed for the realtime engine on the Raspberry Pi: cheap to call inside
the per-tick loop, doesn't allocate during the measurement, and gives you
back the statistics that actually matter for realtime budgeting (p50, p95,
p99, max — not just the mean, which hides the tail spikes that cause tick
drops on the Pi).

Typical usage from Piano_display:

    from perf import PerfLogger
    perf = PerfLogger("realtime")

    # Per tick:
    with perf.timed(engine.last_action_will_be_or_default):
        rollout = engine.commit(played_token)
    # Or, when you don't know the bucket up-front:
    t = perf.start()
    rollout = engine.commit(played_token)
    perf.stop(t, engine.last_action)

    # Every N ticks:
    perf.print_summary(recent_n=200)

    # On exit:
    perf.save_csv("commit_timings.csv")

Bucketing by `engine.last_action` is the important bit — it separates the
cheap match/silence commits (one LSTM forward) from the *expensive*
deviation commits (`rollout_ticks` LSTM forwards in `_build_deviation_rollout`).
That's where the realtime hits come from, and where you'd notice if Pi
performance is tightening up.
"""

import csv
import time


class PerfLogger:
    """Buckets of duration samples (milliseconds), keyed by event label.

    Storage is a dict-of-lists; appending is O(1). No fancy ring buffer —
    just don't run the program for hours. If memory is a concern, call
    `clear()` periodically.
    """

    def __init__(self, name="perf"):
        self.name = name
        # bucket_name -> list of durations in ms
        self.buckets = {}

    # ----- recording ----------------------------------------------------
    def start(self):
        """Return a starting timestamp. Pass it to stop() with a bucket."""
        return time.perf_counter()

    def stop(self, t0, bucket):
        """End a measurement started by start(t0) and record into `bucket`."""
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.record(bucket, dt_ms)
        return dt_ms

    def record(self, bucket, dt_ms):
        """Append a raw duration sample (in ms) into the named bucket."""
        b = self.buckets.get(bucket)
        if b is None:
            b = []
            self.buckets[bucket] = b
        b.append(dt_ms)

    def timed(self, bucket):
        """Context-manager wrapper around start/stop.

            with perf.timed("commit"):
                engine.commit(...)
        """
        return _Timed(self, bucket)

    def clear(self):
        """Drop all recorded samples."""
        self.buckets.clear()

    # ----- statistics ---------------------------------------------------
    @staticmethod
    def _stats(samples):
        if not samples:
            return None
        s = sorted(samples)
        n = len(s)
        return {
            "n":    n,
            "mean": sum(s) / n,
            "p50":  s[n // 2],
            "p95":  s[min(n - 1, int(n * 0.95))],
            "p99":  s[min(n - 1, int(n * 0.99))],
            "max":  s[-1],
        }

    def summary(self, recent_n=None):
        """Return {bucket: stats_dict}. If recent_n is given, only consider
        the last `recent_n` samples per bucket — useful for "what does the
        last few seconds look like" without losing the historical record."""
        out = {}
        for name, samples in self.buckets.items():
            if not samples:
                continue
            window = samples[-recent_n:] if recent_n else samples
            out[name] = self._stats(window)
        return out

    def print_summary(self, recent_n=None, header=True):
        """Pretty-print a summary table to stdout."""
        stats = self.summary(recent_n)
        if not stats:
            return
        suffix = f" (last {recent_n})" if recent_n else ""
        if header:
            print(f"\n── {self.name} timings (ms){suffix} ──")
            print(f"  {'bucket':<14s} {'n':>6s} {'mean':>7s} "
                  f"{'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
        # Sort buckets so deviation always stands out at the top.
        order = ["deviation", "locked", "match", "silence", "done", "init"]
        keys = sorted(stats.keys(),
                      key=lambda k: (order.index(k) if k in order else 99, k))
        for name in keys:
            s = stats[name]
            print(f"  {name:<14s} {s['n']:>6d} {s['mean']:>7.2f} "
                  f"{s['p50']:>7.2f} {s['p95']:>7.2f} "
                  f"{s['p99']:>7.2f} {s['max']:>7.2f}")

    # ----- export -------------------------------------------------------
    def save_csv(self, path):
        """Dump every raw sample to CSV (bucket, sample_idx, duration_ms).
        Lets you analyse the distribution properly in a spreadsheet or
        a plotting tool."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bucket", "sample_idx", "duration_ms"])
            for name, samples in self.buckets.items():
                for i, dt in enumerate(samples):
                    w.writerow([name, i, f"{dt:.4f}"])

    def save_summary_csv(self, path, recent_n=None):
        """Dump the aggregated summary (one row per bucket)."""
        stats = self.summary(recent_n)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bucket", "n", "mean_ms", "p50_ms",
                        "p95_ms", "p99_ms", "max_ms"])
            for name, s in stats.items():
                w.writerow([name, s["n"],
                            f"{s['mean']:.3f}", f"{s['p50']:.3f}",
                            f"{s['p95']:.3f}", f"{s['p99']:.3f}",
                            f"{s['max']:.3f}"])


class _Timed:
    """Context-manager returned by PerfLogger.timed()."""
    __slots__ = ("perf", "bucket", "t0")

    def __init__(self, perf, bucket):
        self.perf = perf
        self.bucket = bucket
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self.t0) * 1000.0
        self.perf.record(self.bucket, dt_ms)
        return False
