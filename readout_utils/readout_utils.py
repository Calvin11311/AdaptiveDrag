import bitsandbytes as bnb
import gc
from IPython.display import display
from IPython.display import clear_output
import numpy as np
from omegaconf import ListConfig
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from diffusers import DDIMScheduler
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_xl_adapter import _preprocess_adapter_image

from dhf.stable_diffusion.resnet import init_resnet_func, collect_feats, collect_channels
import rg_operators,rg_helpers
def diffusion_step(
        model,
        controller, 
        latents, 
        context,
        added_cond_kwargs,
        i, 
        t,
        guidance_scale,
        log=False,
        num_optimizer_steps=[], 
        lr=2e-2,
        low_memory=True,
        pbar=None
    ):
    
    def prepare_control(guess_mode=False):
        control_image = [edit["control_image"] for edit in controller.edits if "control_image" in edit][0]
        control_image_input = model.prepare_image(
            control_image, 
            control_image.size[0], 
            control_image.size[1], 
            latents.shape[0], 
            1, 
            latents.device, 
            latents.dtype, 
            True, 
            guess_mode
        )
        # Infer ControlNet only for the conditional batch.
        control_model_input = torch.cat([latents] * 2)
        controlnet_prompt_embeds = context
        down_block_res_samples, mid_block_res_sample = model.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=control_image_input,
            conditioning_scale=1.0,
            guess_mode=guess_mode,
            return_dict=False,
        )
        return {
            "down_block_additional_residuals": down_block_res_samples,
            "mid_block_additional_residual": mid_block_res_sample
        }
    
    def prepare_adapter():
        control_image = [edit["control_image"] for edit in controller.edits if "control_image" in edit][0]
        adapter_input = _preprocess_adapter_image(control_image, control_image.size[0], control_image.size[1])
        adapter_input = adapter_input.to(model.adapter.device).to(model.adapter.dtype)
        adapter_state = model.adapter(adapter_input)
        adapter_conditioning_scale = 1.0
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * adapter_conditioning_scale
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2 * latents.shape[0], dim=0)
        down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
        return {
            "down_block_additional_residuals": down_intrablock_additional_residuals
        }

    def _diffusion_step(latents):
        # unet_kwargs = {}
        # unet_context = context
        # if hasattr(model, "controlnet"):
        #     unet_kwargs = prepare_control()
        # elif hasattr(model, "adapter"):
        #     unet_kwargs = prepare_adapter()
        # latents_input = torch.cat([latents] * 2)
        # noise_pred = model.unet(
        #     latents_input,
        #     t,
        #     encoder_hidden_states=unet_context,
        #     added_cond_kwargs=added_cond_kwargs,
        #     **unet_kwargs
        # )["sample"]
        # noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        # if type(guidance_scale) is ListConfig or type(guidance_scale) is list:
        #     # guidance_scale_pt = torch.tensor(guidance_scale)[:, None, None, None]
        #     # guidance_scale_pt = guidance_scale_pt.to(noise_pred.device).to(noise_pred.dtype)
        # else:
        #     guidance_scale_pt = guidance_scale
        
        if controller.edits:
            feats = controller.collect_and_resize_feats()
        else:
            feats = None
        return feats
    
    def optimize_latents(latents):
        # Use gradient checkpointing which recomputes
        # intermediates on the backwards pass to save memory.
        latents.requires_grad_(True)
        torch.set_grad_enabled(True)
        # optimizer_cls = bnb.optim.AdamW8bit if low_memory else torch.optim.AdamW
        # optimizer = optimizer_cls([latents], lr=lr, weight_decay=0.0)
        losses = []
        for j in range(num_optimizer_steps[i]):
            loss = 0
            feats = checkpoint(_diffusion_step, latents)
            emb = None
            # b = feats.shape[0]
            # Compute the loss over both the uncond and cond branch
            for gt_idx in [0]:
                log_branch = (log and gt_idx != 0)
                batch_idx = gt_idx + 1
                latents_scale = (latents.detach().min(), latents.detach().max())
                rg_loss = rg_operators.loss_guidance(controller, feats, batch_idx, gt_idx, edits=controller.edits, log=log_branch, emb=emb, latents_scale=latents_scale, t=t, i=i)
                loss += rg_loss
            losses.append(loss.item())
            # if optimizer is not None:
            #     optimizer.zero_grad(set_to_none=True)
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
        return losses
    # Clear previous logged outputs
    
    losses = optimize_latents(latents)
    rg_loss = np.mean(losses)

    if pbar is not None:
        description = f"Readout Guidance Loss {rg_loss}"
        pbar.set_description(description)
    # torch.cuda.empty_cache()
    # gc.collect()
    return rg_loss