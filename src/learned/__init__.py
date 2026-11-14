"""Learned possession evidence, trained ONLY on other games (RULES.md: other games are fine).

Training data = Metrica Sports' public sample games (tracking + event data), never the
provided BAR-ATH match. The model sees only columns of the P1 feature table, so exactly the
same features exist at training time and at inference time on the challenge match.
"""
