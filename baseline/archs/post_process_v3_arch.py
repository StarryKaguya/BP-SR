from basicsr.utils.registry import ARCH_REGISTRY

from .bpsr_dualstream_cross_attention_arch import BPSR_DualStreamCrossAttention


@ARCH_REGISTRY.register()
class PostProcess_V3(BPSR_DualStreamCrossAttention):
    """Backward-compatible alias for old BP-SR checkpoints and configs."""
