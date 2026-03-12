from basicsr.utils.registry import DATASET_REGISTRY

from .bpsr_aligned_triplet_dataset import BPSRAlignedTripletDataset


@DATASET_REGISTRY.register()
class NtireImageDataset(BPSRAlignedTripletDataset):
    """Backward-compatible alias for old BP-SR configs."""
