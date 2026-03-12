from basicsr.utils.registry import MODEL_REGISTRY

from .bpsr_refinement_model import BPSRRefinementModel


@MODEL_REGISTRY.register()
class PPV5Model(BPSRRefinementModel):
    """Backward-compatible alias for old BP-SR configs."""
