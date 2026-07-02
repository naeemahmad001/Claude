"""FXE frequency-counter analyzer.

A small toolkit + GUI to load K+K-style FXE frequency-counter files,
plot up to 8 channels with interactive zoom / time-interval selection,
join several files on a common time base, and compute the fractional
frequency stability (overlapping Allan deviation) at arbitrary averaging
times such as 1 ms, 10 ms, 100 ms, 200 ms, 1 s and 10 s.
"""

__version__ = "0.1.0"

__all__ = ["loader", "stability"]
