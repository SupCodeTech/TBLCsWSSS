from __future__ import annotations

from .classification_cam_affinity import (
    ClassificationHead,
    CAMGenerator,
    generate_initial_pseudo_label,
    AffinityCalculator,
    generate_affinity_labels,
    compute_affinity_loss,
)
