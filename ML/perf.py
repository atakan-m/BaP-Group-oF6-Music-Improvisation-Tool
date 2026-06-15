"""Lightweight performance logger for measuring per-event timings.

Designed for the realtime engine on the Raspberry Pi: cheap to call inside
the per-tick loop, allocates nothing during the measurement, and reports
the statistics that actually matter for realtime budgeting (p50, p95, p99,
max — not just the mean, which hides the tail spikes that cause dropped
ticks on the Pi).

Typical usage from Piano_display.py:

    from perf import PerfLogger
    perf = PerfLogger("realtime")

    # Per tick (bucket known up-front):
    with perf.timed("commit"):
        rollout = engine.commit(played_token)

    # Per tick (bucket determined by the call itself, e.g. last_action):
    t = perf.start()
    rollout = engine.commit(played_token)
    perf.stop(t, engine.last_action)

    # Every N ticks:
    perf.print_summary(recent_n=200)

    # On exit:
    perf.save_csv("commit_timings.csv")

Bucketing by engine.last_action is the important bit — it separates the
cheap match / silence commits (~2 LSTM forwards) from the expensive
deviation commits, where the rollout is partially rebuilt and the per-tick
cost spikes. The deviation bucket is the one to watch.
"""

import csv
import time


class PerfLogger:
    """Collects duration samples (in milliseconds) into named buckets.

    Storage is a dict-of-lists; appending each sample is O(1). There's no
    ring buffer — for very long runs, call clear() periodically if memory
    becomes an issue.
    """

    def __init__(self, name="perf"):
        self.name = name
        # bucket_name -> list of durations in milliseconds.
        self.buckets = {}

    # ----- recording ---------------------------------------------------
    def start(self):
        """Return a starting timestamp. Pair with stop(t0, bucket) to record."""
        return time.perf_counter()

    def stop(self, t0, bucket):
        """End a measurement started by start() and record into `bucket`.

        Returns the elapsed time in milliseconds so the caller can log it
        inline if desired.
        """
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.record(bucket, dt_ms)
        return dt_ms

    def record(self, bucket, dt_ms):
        """Append a duration sample (in milliseconds) to the named bucket."""
        b = self.buckets.get(bucket)
        if b is None:
            b = []
            self.buckets[bucket] = b
        b.append(dt_ms)

    def timed(self, bucket):
        """Return a context-manager that times its body.

            with perf.timed("commit"):
                engine.commit(...)
        """
        return _Timed(self, bucket)

    def clear(self):
        """Drop all recorded samples in every bucket."""
        self.buckets.clear()

    # ----- statistics --------------------------------------------------
    @staticmethod
    def _stats(samples):
        """Compute {n, mean, p50, p95, p99, max} for a list of samples.

        Returns None for empty input. Percentiles are picked with a simple
        sort-and-index — accurate enough for the sample counts we collect.
        """
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
        """Return per-bucket aggregate stats as a dict.

        If recent_n is given, only the last `recent_n` samples per bucket
        are considered — useful for "how is performance right now" without
        discarding the historical record.
        """
        out = {}
        for name, samples in self.buckets.items():
            if not samples:
                continue
            window = samples[-recent_n:] if recent_n else samples
            out[name] = self._stats(window)
        return out

    def print_summary(self, recent_n=None, header=True):
        """Pretty-print a summary table to stdout.

        Buckets are sorted so that deviation is always at the top — that's
        the spike most worth watching during a realtime run.
        """
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

    # ----- export ------------------------------------------------------
    def save_csv(self, path):
        """Dump every raw sample to a CSV file.

        Columns: bucket, sample_idx, duration_ms. Suitable for loading into
        a spreadsheet or plotting tool to inspect the full distribution.
        """
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bucket", "sample_idx", "duration_ms"])
            for name, samples in self.buckets.items():
                for i, dt in enumerate(samples):
                    w.writerow([name, i, f"{dt:.4f}"])

    def save_summary_csv(self, path, recent_n=None):
        """Dump the aggregated summary (one row per bucket) to CSV.

        Columns: bucket, n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms.
        """
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
    """Internal context manager returned by PerfLogger.timed().

    Records the elapsed time on __exit__ regardless of whether the body
    raised; uses __slots__ to keep allocation cost down inside hot loops.
    """
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
