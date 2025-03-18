import copy
import torch
import torch.nn.functional as F
from .pos_super_seg import cal_super_seg_dis
from tqdm import tqdm
import json
import os 
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from drag_pipeline_nocontrol_sam_area_dmask_noiseloss import DragPipeline
from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
import math




import numpy as np
import cv2
import matplotlib.pyplot as plt



from scipy.ndimage import zoom
def downsample_mask(mask, threshold,shape):
    # 输入检查
    # if (mask.shape[0]%2 != 0) or (mask.shape[1]%2 != 0):
    #     raise ValueError("Input mask array must can be divide by 2.")
    
    # # 确保mask是二值化的（0或1）
    # if not np.isin(mask, [0, 1]).all():
    #     raise ValueError("Input mask array must be binary (0 or 1).")

    # 定义新形状并进行下采样
    new_shape = shape
    reshaped_mask = mask.reshape(new_shape[0], 2, new_shape[1], 2)
    
    # 计算每个2x2块中1的比例
    downsampled_mask = reshaped_mask.mean(axis=(1, 3)) > threshold
    
    # 转换为0和1的二值化mask
    downsampled_mask = downsampled_mask.astype(int)
    
    return downsampled_mask
def extract_super_mask(matrix, p, r,t,segments,n_segments):
    # print("half_circle",p,r)
    _, _, h, w = matrix.shape
    py, px = p.numpy()
    ty, tx = t.numpy()

    # Create a mesh grid for coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate the distance from the center
    dist_from_center = np.sqrt((x - px)**2 + (y - py)**2)

    # Calculate the angle from the center to each coordinate
    angle = np.arctan2(y - py, x - px)

    # Calculate the angle of the line from p to t
    angle_pt = np.arctan2(ty - py, tx - px)

    # Normalize the angle to get half circle region
    angle_diff = np.abs(np.angle(np.exp(1j * (angle - angle_pt))))
    angle_region = (angle_diff <= np.pi / 2)

    super_region=np.zeros(segments.shape, dtype=bool)
    xx=int(2*px)
    yy=int(2*py)
    start_segment_id = segments[yy, xx]
    super_region[segments == start_segment_id] = True

    # super_region_s=zoom(super_region, (0.5, 0.5))
    # super_region_s=downsample_mask(super_region,0.5,angle_region.shape)
    super_region = (super_region).astype(np.uint8)
    new_shape=(angle_region.shape[1],angle_region.shape[0])
    super_region_s=cv2.resize(super_region, new_shape, interpolation=cv2.INTER_LINEAR)


    super_region_s = (super_region_s * 255).astype(np.uint8)
    cv2.imwrite("./mask_region.png",super_region_s)

    mask_region = angle_region & super_region_s
    area_min=int(((h*w)/(n_segments*4)))
    num_pixels = np.sum(mask_region)
    if num_pixels < area_min:
        mask_region =  (dist_from_center <= r) & (angle_diff <= np.pi / 2)
    
    mask_region=torch.from_numpy(mask_region)
    expanded_mask_region = mask_region.unsqueeze(0).unsqueeze(0).expand_as(matrix)  

    # Extract the half-circle region
    # half_circle_matrix = np.zeros_like(matrix.cpu())
    mask_all=expanded_mask_region .cuda()
    # half_circle_matrix[expanded_half_circle_region] = matrix[expanded_half_circle_region].cpu()
    # half_circle_matrix = torch.from_numpy(half_circle_matrix).cuda()
    mask_save=mask_region.cpu().numpy()
    mask_np = (mask_save * 255).astype(np.uint8)
    cv2.imwrite("./mask_super_point.png",mask_np)

    return mask_all


def point_tracking(F0,
                   F1,
                   handle_points,
                   handle_points_init,
                   segments,
                   centroids,
                   target_points,
                   n_segments):
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            ti=target_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            # 依据超像素给出r_m值
            x=int(handle_points[i][1])
            y=int(handle_points[i][0])
            r_p=int((2 * cal_super_seg_dis(segments,centroids,x,y))//5)
            # r_p=3*r_m
            print("new_r_p:",r_p)

            r1, r2 = max(0,int(pi[0])-r_p), min(max_r,int(pi[0])+r_p+1)
            c1, c2 = max(0,int(pi[1])-r_p), min(max_c,int(pi[1])+r_p+1)
            # F1_neighbor = F1[:, :, r1:r2, c1:c2]
            # Example Usage
            
            mask_all = extract_super_mask(F1, pi, r_p,ti,segments,n_segments)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            mask_all = mask_all[:, :, r1:r2, c1:c2]
            # print(F1_neighbor)
            print(F1_neighbor.shape)


            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            
            all_dist = torch.where(mask_all, all_dist, torch.tensor(float('inf')))
            all_dist = all_dist.squeeze(dim=0)

            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            # handle_points[i][0] = pi[0] - args.r_p + row
            # handle_points[i][1] = pi[1] - args.r_p + col
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
        return handle_points

def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()

# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
# 得到以 (x, y) 为中心、半径为 r 的双线性内插特征斑块
def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()#包含输入input张量每个元素的floor，即取不大于元素的最大整数。
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def get_original_features(model,
                          init_code,
                          text_embeddings,
                          args):
        timesteps = model.scheduler.timesteps
        strat_time_step_idx = args.n_inference_step - args.n_actual_inference_step
        original_step_output = {}
        features = {}
        cur_latents = init_code.detach().clone()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps[strat_time_step_idx:],
                                       desc="Denosing for mask features")):
                # if i <= self.t2:
                model_inputs = cur_latents
                noise_pred, F0 = model.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings,layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
                cur_latents = model.scheduler.step(noise_pred, t, model_inputs, return_dict=False)[0]
                original_step_output[t.item()] = cur_latents.cpu()
                features[t.item()] = F0.cpu()

        del noise_pred, cur_latents, F0
        torch.cuda.empty_cache()
        return original_step_output, features

import datetime
from .attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from .freeu_utils import register_free_upblock2d, register_free_crossattn_upblock2d



import numpy as np
from PIL import Image
import cv2
def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst
def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask
def drag_diffusion_update(model,
                          init_code,
                          text_embeddings,
                          t,
                          handle_points,
                          target_points,
                          mask,
                          args,
                          segments,
                          centroids,
                        #   init_code_orig, 
                        #   height,
                        #   width,
                        #   controlnet,
                        #   pose,#(512.704)
                        #   negative_prompt,
                        #   full_h, full_w,start_step,start_layer,
                          lora_path,model_path,
                          vae_path,num_seg):
    # mask_path="./mask_images/"#'./mask_images/mask_image_'+str(i)+'.png'
    # source_img_path="./original_image.png"
    # model_init=copy.deepcopy(model)
    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # original_step_output,features_=get_original_features(model,init_code,text_embeddings,args)
    # original_features=features_[t.item()].cuda()
    #原有F0
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(init_code, t,
            encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    # init_code_list=[]
    handle_points_list=[]
    handle_points_list.append(handle_points_init)



    # test_code=[]
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
    #                       beta_schedule="scaled_linear", clip_sample=False,
    #                       set_alpha_to_one=False, steps_offset=1)
    # model_gen = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    # model_gen.modify_unet_forward()
    # if vae_path != "default":
    #     model_gen.vae = AutoencoderKL.from_pretrained(
    #         vae_path
    #     ).to(model_gen.vae.device, model_gen.vae.dtype)
    # model_gen.enable_model_cpu_offload()
    # if lora_path == "":
    #     print("applying default parameters")
    #     model_gen.unet.set_default_attn_processor()
    # else:
    #     print("applying lora: " + lora_path)
    #     model_gen.unet.load_attn_procs(lora_path)
    di_lam=1
    max_step = 500
    for step_idx in range(args.n_pix_step):
        print("step_idx: ", str(step_idx))
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # if step_idx != 0:
            #     pre_x_prev_updated=x_prev_updated
            unet_output, F1 = model.forward_unet_features(init_code, t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)# 返回变量 return_features = torch.cat(all_return_features, dim=1) return unet_output, return_features
            x_prev_updated,_ = model.step(unet_output, t, init_code)
            #利用点跟踪函数估计下一个handel位置
            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, segments,centroids,target_points,num_seg)
                print('new handle points', handle_points)
                cur_handle_point=copy.deepcopy(handle_points)
                handle_points_list.append(cur_handle_point)
                for i in range(len(handle_points)):
                    pi, ti = handle_points[i], target_points[i]
                    # skip if the distance between target and source is less than 1
                    if (ti - pi).norm() < 2.:
                        continue
                    oi=handle_points_init[i]
                    prev_handel_points=handle_points_list[-1][i]

                    #更新距离小于平均移动距离，增加采样步数
                    if (prev_handel_points-pi).norm()<((oi-pi).norm()/args.n_pix_step) and args.n_pix_step < max_step:
                        args.n_pix_step+=1


            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            _, _, max_r, max_c = F0.shape# input size //2
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 1.:
                    continue
                # 当前handel点到目标点的距离,归一化
                di = ((ti - pi) / (ti - pi).norm()) * di_lam
                # 依据超像素给出r_m值
                x=int(handle_points[i][1])
                y=int(handle_points[i][0])
                r_m=int(cal_super_seg_dis(segments,centroids,x,y)//7)
                print("new_r_m:",r_m)

                # motion supervision
                # with boundary protection


                #add the original preserve
                # original_features.requires_grad_(True)
                #
                # pi = handle_points_init[i]
                # r1, r2 = max(0,int(pi[0])-r_m), min(max_r,int(pi[0])+r_m+1)
                # c1, c2 = max(0,int(pi[1])-r_m), min(max_c,int(pi[1])+r_m+1)
                # # f0_patch_1 = F1[:,:,r1:r2, c1:c2].detach()
                # f0_patch_1 = original_features[:,:,r1:r2, c1:c2].detach()#torch.Size([1, 640, 15, 15])
                # ori_f_lam=0.05

                #原始update
                # pi = handle_points[i]
                #r1,r2:y范围，c1,c2：x范围；handel为中心点
                r1, r2 = max(0,int(pi[0])-r_m), min(max_r,int(pi[0])+r_m+1)
                c1, c2 = max(0,int(pi[1])-r_m), min(max_c,int(pi[1])+r_m+1)
                # print(r1,r2,c1,c2)
                f0_patch_2 = F1[:,:,r1:r2, c1:c2].detach()#.detach()方法用于创建一个新的tensor，这个tensor与原始的计算图分离，但共享内存空间。 这意味着当你对一个tensor调用.detach()方法时，返回的新tensor将不再与原始的计算图相连，因此不会计算梯度。
                # print(f0_patch_2.shape,f0_patch_2.shape)
                # f0_patch=ori_f_lam*f0_patch_1+(1-ori_f_lam)*f0_patch_2 ##torch.Size([1, 640, 15, 15])
                mask_m=extract_super_mask(F1, pi, r_m,ti,segments,num_seg)
                mask_m=mask_m[:, :, r1:r2, c1:c2]

                f0_patch=f0_patch_2
                f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])#计算 torch.Size([1, 640, 15, 15])
                f0_patch*=mask_m
                f1_patch*=mask_m

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                # 运动监督损失的第一阶段损失部分
                loss += ((2*r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged dress change,
            # 运动监督的第二阶段损失，针对mask进行计算
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
                # loss += args.lam * ((x_prev_updated-x_prev_0)*(interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code#, init_code_list, handle_points_list, test_code

def drag_diffusion_update_gen(model,
                              init_code,
                              text_embeddings,
                              t,
                              handle_points,
                              target_points,
                              mask,
                              args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # positive prompt embedding
    if args.guidance_scale > 1.0:
        unconditional_input = model.tokenizer(
            [args.neg_prompt],
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_emb = model.text_encoder(unconditional_input.input_ids.to(text_embeddings.device))[0].detach()
        text_embeddings = torch.cat([unconditional_emb, text_embeddings], dim=0)

    # the init output feature of unet
    with torch.no_grad():
        if args.guidance_scale > 1.:
            model_inputs_0 = copy.deepcopy(torch.cat([init_code] * 2))
        else:
            model_inputs_0 = copy.deepcopy(init_code)
        unet_output, F0 = model.forward_unet_features(model_inputs_0, t, encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
        if args.guidance_scale > 1.:
            # strategy 1: discard the unconditional branch feature maps
            # F0 = F0[1].unsqueeze(dim=0)
            # strategy 2: concat pos and neg branch feature maps for motion-sup and point tracking
            # F0 = torch.cat([F0[0], F0[1]], dim=0).unsqueeze(dim=0)
            # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
            coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
            F0 = torch.cat([(1-coef)*F0[0], coef*F0[1]], dim=0).unsqueeze(dim=0)

            unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
            unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.guidance_scale > 1.:
                model_inputs = init_code.repeat(2,1,1,1)
            else:
                model_inputs = init_code
            unet_output, F1 = model.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx, interp_res_h=args.sup_res_h, interp_res_w=args.sup_res_w)
            if args.guidance_scale > 1.:
                # strategy 1: discard the unconditional branch feature maps
                # F1 = F1[1].unsqueeze(dim=0)
                # strategy 2: concat positive and negative branch feature maps for motion-sup and point tracking
                # F1 = torch.cat([F1[0], F1[1]], dim=0).unsqueeze(dim=0)
                # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
                coef = args.guidance_scale / (2*args.guidance_scale - 1.0)
                F1 = torch.cat([(1-coef)*F1[0], coef*F1[1]], dim=0).unsqueeze(dim=0)

                unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
                unet_output = unet_output_uncon + args.guidance_scale * (unet_output_con - unet_output_uncon)
            x_prev_updated,_ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                #利用最邻近搜索，估计下一个handelpoints
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            _, _, max_r, max_c = F0.shape
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                # with boundary protection
                r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                f0_patch = F1[:,:,r1:r2, c1:c2].detach()
                f1_patch = interpolate_feature_patch(F1,r1+di[0],r2+di[0],c1+di[1],c2+di[1])

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)

                loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig - init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code

