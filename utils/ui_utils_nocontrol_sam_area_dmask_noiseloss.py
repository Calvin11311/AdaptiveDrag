import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.embeddings import ImageProjection
from drag_pipeline_nocontrol_sam_area_dmask_noiseloss import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .drag_utils_nocontrol_sam_area_dmask_noiseloss import drag_diffusion_update, drag_diffusion_update_gen
from .lora_utils import train_lora
from .attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from .freeu_utils import register_free_upblock2d, register_free_crossattn_upblock2d
# from .utils_ootd import get_mask_location
# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose
from .mask_utils import get_seg_mask,move_mask
from skimage.measure import regionprops
import copy
# from .SAM.sam_utils import generate_sam_seg
from .sam2.sam2_utils import generate_sam2_seg
# -------------- general UI functionality --------------
def clear_all(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=True), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None

def clear_all_gen(length=480):
    return gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        gr.Image.update(value=None, height=length, width=length, interactive=False), \
        [], None, None, None

def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height,width,_ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length,int(length*height/width)), PIL.Image.BILINEAR)
    mask  = cv2.resize(mask, (length,int(length*height/width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], gr.Image.update(value=masked_img, interactive=True), mask

# once user upload an image, the original image is stored in `original_image`
# the same image is displayed in `input_image` for point clicking purpose
def store_img_gen(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = np.array(image)
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask

# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []
# ------------------------------------------------------

# ----------- dragging user-input image utils -----------
def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress()):
    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress)
    return "Training LoRA Done!"


from skimage.io import imread, imsave

def preprocess_image(image,
                     device,
                     dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image


# from tqdm import tqdm
def run_drag(source_image,
             image_with_clicks,
             mask,
             prompt,
             points,
             inversion_strength,
             lam,
             latent_lr,
             n_pix_step,
             model_path,
             vae_path,
             lora_path,
             start_step,
             start_layer,
             save_dir="./results"
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    # image = Image.fromarray(image_array)
    # image.save('./original_image')
    source_image2=source_image[..., ::-1]
    cv2.imwrite("./original_image.png",source_image2)
    
   
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)

    # model = DragPipeline.from_pretrained(model_path, scheduler=scheduler)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # off load model to cpu, which save some memory.
    model.enable_model_cpu_offload()

    # initialize parameters
    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    # generator=torch.manual_seed(seed)
    generator = torch.Generator("cuda").manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [3]
    #motion距离和点跟踪距离
    args.r_m = 5
    args.r_p = 15
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    print(args)

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)
    print("lora_path",lora_path)
    # sam_img,handle_seg = generate_sam_seg("./original_image.png",handle_points,target_points)
    sam_img,handle_seg = generate_sam2_seg("./original_image.png",handle_points,target_points)

    # save_handle_seg=(handle_seg * 255).astype(np.uint8)
    # cv2.imwrite("./handle_seg.png",save_handle_seg)

    #自动估计mask区域 超像素source image (682, 512, 3)
    num_seg=200
    mask_arr, segments=get_seg_mask(sam_img,num_seg,10,handle_points,target_points)

    #获得每个超像素的中心坐标点
    regions = regionprops(segments + 1)  # 区域标签从1开始

    # 存储每个超像素中心点的坐标
    centroids = np.array([props.centroid for props in regions])


    source_image = preprocess_image(source_image, device, dtype=torch.float16)
    # source_image = preprocess_image(source_image, device)

    height,width=source_image.shape[2:]
    
    image_with_clicks = preprocess_image(image_with_clicks, device)

    mask_arr = torch.from_numpy(mask_arr).float() / 255.
    mask_arr[mask_arr > 0.0] = 1.0
    handle_seg = torch.from_numpy(handle_seg).float() / 255.
    handle_seg[handle_seg > 0.0] = 1.0
    mask_arr+=handle_seg
    mask_arr[mask_arr > 0.0] = 1.0
    # imsave('./mask_image.png', mask_arr)
    # move_mask_img=move_mask(handle_seg, handle_points, target_points)
    move_mask_img = handle_seg
    imsave('./mask_image.png', move_mask_img)
    mask_arr=move_mask_img

    mask_arr = rearrange(mask_arr, "h w -> 1 1 h w").cuda()
    mask_arr = F.interpolate(mask_arr, (args.sup_res_h, args.sup_res_w), mode="nearest")


    # mask = torch.from_numpy(mask).float() / 255.
    # mask[mask > 0.0] = 1.0
    # mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    # mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")
    
    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # obtain text embeddings
    text_embeddings = model.get_text_embeddings(prompt)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               encoder_hidden_states=text_embeddings,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step,
                               genrator=generator)

    # empty cache to save memory
    torch.cuda.empty_cache()

    init_code = invert_code# input size // 8
    init_code_orig = copy.deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # convert dtype to float for optimization
    init_code = init_code.float()
    text_embeddings = text_embeddings.float()
    model.unet = model.unet.float()
    

    # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    updated_init_code = drag_diffusion_update(
        model,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        mask_arr,
        args,
        segments,
        centroids,
        lora_path,model_path, vae_path,
        num_seg
        )

    updated_init_code = updated_init_code.half()
    # for tt in range(len(updated_init_code_list)):
        # updated_init_code_list[tt]=updated_init_code_list[tt].half()
    # for tt in range(len(test_code)):
    #     test_code[tt]=test_code[tt].half()

    text_embeddings = text_embeddings.half()
    model.unet = model.unet.half()

    # empty cache to save memory
    torch.cuda.empty_cache()

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # latents=torch.cat([init_code_orig, updated_init_code], dim=0)
    # latents_leaf = latents.detach().requires_grad_(True)
    # latents_ori=torch.cat([init_code_orig, init_code_orig], dim=0).detach()
    latents = torch.tensor(init_code_orig.tolist() + updated_init_code.tolist(), requires_grad=True,dtype=torch.float16)


    latents_ori = torch.tensor(init_code_orig.tolist() + init_code_orig.tolist(), requires_grad=True,dtype=torch.float16)


    # inference the synthesized image
    # generator = torch.Generator("cuda").manual_seed(42)
    gen_image= model(
        prompt=args.prompt,
        encoder_hidden_states=torch.cat([text_embeddings]*2, dim=0),
        batch_size=2,
        latents=latents,
        latents_ori=latents_ori,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
        return_intermediates=False,
        handle_points=handle_points,
        target_points=target_points,

        #control
        height=height,
        width=width,
        #for control net
        # controlnet=controlnet.cuda(),
        # image=pose,#(512.704)
        # controlnet_conditioning_scale=1.0,
        # negative_prompt=negative_prompt,
        generator=generator
    )[1].unsqueeze(dim=0)

    

    # 最终结果
    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        image_with_clicks.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        gen_image[0:1].float()
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image

# -------------------------------------------------------

# ----------- dragging generated image utils -----------
# once the user generated an image
# it will be displayed on mask drawing-areas and point-clicking area
def gen_img(
    length, # length of the window displaying the image
    height, # height of the generated image
    width, # width of the generated image
    n_inference_step,
    scheduler_name,
    seed,
    guidance_scale,
    prompt,
    neg_prompt,
    model_path,
    vae_path,
    lora_path,
    b1,
    b2,
    s1,
    s2):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DragPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    # model = DragPipeline.from_pretrained(model_path).to(device)

    if scheduler_name == "DDIM":
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                        beta_schedule="scaled_linear", clip_sample=False,
                        set_alpha_to_one=False, steps_offset=1)
    elif scheduler_name == "DPM++2M":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config
        )
    elif scheduler_name == "DPM++2M_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config, use_karras_sigmas=True
        )
    else:
        raise NotImplementedError("scheduler name not correct")
    model.scheduler = scheduler
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)
    # set lora
    #if lora_path != "":
    #    print("applying lora for image generation: " + lora_path)
    #    model.unet.load_attn_procs(lora_path)
    if lora_path != "":
        print("applying lora: " + lora_path)
        model.load_lora_weights(lora_path, weight_name="lora.safetensors")

    # apply FreeU
    if b1 != 1.0 or b2!=1.0 or s1!=1.0 or s2!=1.0:
        print('applying FreeU')
        register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
    else:
        print('do not apply FreeU')

    # initialize init noise
    seed_everything(seed)

    init_noise = torch.randn([1, 4, height // 8, width // 8], device=device, dtype=model.vae.dtype)
    gen_image, intermediate_latents = model(prompt=prompt,
                                            neg_prompt=neg_prompt,
                                            num_inference_steps=n_inference_step,
                                            latents=init_noise,
                                            guidance_scale=guidance_scale,
                                            return_intermediates=True)
    gen_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    gen_image = (gen_image * 255).astype(np.uint8)

    if height < width:
        # need to do this due to Gradio's bug
        return gr.Image.update(value=gen_image, height=int(length*height/width), width=length, interactive=True), \
            gr.Image.update(height=int(length*height/width), width=length, interactive=True), \
            gr.Image.update(height=int(length*height/width), width=length), \
            None, \
            intermediate_latents
    else:
        return gr.Image.update(value=gen_image, height=length, width=length, interactive=True), \
            gr.Image.update(value=None, height=length, width=length, interactive=True), \
            gr.Image.update(value=None, height=length, width=length), \
            None, \
            intermediate_latents

def run_drag_gen(
    n_inference_step,
    scheduler_name,
    source_image,
    image_with_clicks,
    intermediate_latents_gen,
    guidance_scale,
    mask,
    prompt,
    neg_prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    vae_path,
    lora_path,
    start_step,
    start_layer,
    b1,
    b2,
    s1,
    s2,
    save_dir="./results"):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DragPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    # model = DragPipeline.from_pretrained(model_path)
    if scheduler_name == "DDIM":
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                        beta_schedule="scaled_linear", clip_sample=False,
                        set_alpha_to_one=False, steps_offset=1)
    elif scheduler_name == "DPM++2M":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config
        )
    elif scheduler_name == "DPM++2M_karras":
        scheduler = DPMSolverMultistepScheduler.from_config(
            model.scheduler.config, use_karras_sigmas=True
        )
    else:
        raise NotImplementedError("scheduler name not correct")
    model.scheduler = scheduler
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # off load model to cpu, which save some memory.
    model.enable_model_cpu_offload()

    # initialize parameters
    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.neg_prompt = neg_prompt
    args.points = points
    args.n_inference_step = n_inference_step
    args.n_actual_inference_step = round(n_inference_step * inversion_strength)
    args.guidance_scale = guidance_scale

    args.unet_feature_idx = [3]

    full_h, full_w = source_image.shape[:2]

    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr

    args.n_pix_step = n_pix_step
    print(args)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    if lora_path != "":
        print("applying lora: " + lora_path)
        model.load_lora_weights(lora_path, weight_name="lora.safetensors")

    # preparing editing meta data (handle, target, mask)
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    # apply FreeU
    if b1 != 1.0 or b2!=1.0 or s1!=1.0 or s2!=1.0:
        print('applying FreeU')
        register_free_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
        register_free_crossattn_upblock2d(model, b1=b1, b2=b2, s1=s1, s2=s2)
    else:
        print('do not apply FreeU')

    # obtain text embeddings
    text_embeddings = model.get_text_embeddings(prompt)

    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]
    init_code = deepcopy(intermediate_latents_gen[args.n_inference_step - args.n_actual_inference_step])
    init_code_orig = deepcopy(init_code)

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    torch.cuda.empty_cache()
    init_code = init_code.to(torch.float16)
    text_embeddings = text_embeddings.to(torch.float16)
    model.unet = model.unet.to(torch.float16)
    updated_init_code = drag_diffusion_update_gen(model, init_code,
        text_embeddings, t, handle_points, target_points, mask, args)
    updated_init_code = updated_init_code.to(torch.float16)
    text_embeddings = text_embeddings.to(torch.float16)
    model.unet = model.unet.to(torch.float16)
    torch.cuda.empty_cache()

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # inference the synthesized image
    gen_image = model(
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        batch_size=2, # batch size is 2 because we have reference init_code and updated init_code
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
        )[1].unsqueeze(dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        image_with_clicks * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        gen_image[0:1]
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image

# ------------------------------------------------------
