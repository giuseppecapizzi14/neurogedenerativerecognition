#!/usr/bin/env python3
"""
Modulo per la definizione delle metriche di valutazione.
Contiene solo le definizioni di tipo necessarie per la configurazione.
"""

from typing import Literal

# Definizione dei tipi di metriche supportate
EvaluationMetric = Literal[
    "accuracy",
    "precision", 
    "recall",
    "f1",
    "loss",
    "sensitivity",
    "specificity"
]
