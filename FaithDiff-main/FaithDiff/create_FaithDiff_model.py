from .pipelines.pipeline_FaithDiff_tlc import DiffaVA_lr_pipeline
import torch
from diffusers import AutoencoderKL
from .models.unet_2d_condition_w_vae import UNet2DConditionModel as UNet2DConditionModel_vae
from diffusers import AutoencoderKL, DDPMScheduler



def FaithDiff_pipeline(sdxl_path, VAE_FP16_path, FaithDiff_path):
    vae = AutoencoderKL.from_pretrained(VAE_FP16_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet2DConditionModel_vae.from_pretrained(sdxl_path, subfolder="unet")
    unet.init_vae_encoder()
    unet.init_information_transformer_layes()
    unet.init_ControlNetConditioningEmbedding()
    unet.denoise_encoder.dtype=torch.float16
    unet.load_state_dict(torch.load(FaithDiff_path), strict=True)
    unet = unet.to(dtype=torch.float16)
    DDPM_scheduler = DDPMScheduler.from_pretrained(sdxl_path, subfolder="scheduler")
    pipe = DiffaVA_lr_pipeline.from_pretrained(
        sdxl_path,
        vae = vae,
        unet=unet,
        add_sample = True,
        denoise_encoder = unet.denoise_encoder,
        DDPM_scheduler = DDPM_scheduler,
        add_watermarker=False,
        torch_dtype=torch.float16,
    )


    return pipe