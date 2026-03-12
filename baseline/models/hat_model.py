from basicsr.utils.registry import MODEL_REGISTRY

from .bpsr_inference_model import BPSRInferenceModel


@MODEL_REGISTRY.register()
class HATModel(BPSRInferenceModel):
    """Backward-compatible alias for old BP-SR configs."""
