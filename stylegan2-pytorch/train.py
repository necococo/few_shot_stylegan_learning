import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from dataset import MultiResolutionDataset, AncherDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
### My addition
import torchextractor as tx # intermediate_features_getter 
import torch.distributions as td
import shutil
#from statistics import mean
######


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

### I added some functions for g_dist_loss ###

def sim(feat1, feat2):
    return F.cosine_similarity(torch.flatten(feat1), torch.flatten(feat2), dim=0) 

def smax(list_x):
    s = nn.Softmax(dim=0)
    return s(torch.tensor(list_x))

def calc_soft_sim(feature_list): #[7*tensor(B,C,H,W)] 
    list_list1 = []
    for f in feature_list:
        b = f.shape[0]
        yf = [ smax([ sim(f[i], f[j]) for j in range(0, b)  if i != j  ]) for i in range(0, b)  ] 
        list_list1.append(yf)
    #print(f'list_list1:{list_list1}')
    return list_list1 

# def kld(p,q):
#     p = td.categorical.Categorical(probs=p)
#     q = td.categorical.Categorical(probs=q)
#     return td.kl.kl_divergence(p, q)
def kld(p,q):
    loss = torch.nn.KLDivLoss(reduction="sum")
    #print(f'kld_loss:{loss(q.log(), p)}')
    return loss(q.log(), p)

def g_dist_loss(p_list_list, q_list_list):
    vals = [kld(p, q) for p_list, q_list in zip(p_list_list, q_list_list) for p, q in zip(p_list, q_list)]
    #print(f'vals:{vals}')
    vals = torch.tensor(vals)
    #print(f'vals.mean():{vals.mean()}')
    return vals.mean()
#######


def d_logistic_loss(real_pred, fake_pred): #torch.Size([4, 1]) or torch.Size([4, 512, N, N])
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths



def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0) # unbind(dim) – dimension to remove

    return noises# list


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        
        return make_noise(batch, latent_dim, 2, device) # n_noise=2

    else:
        return [make_noise(batch, latent_dim, 1, device)]# n_noise=1


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None



def train(args, loader_R_img, loader_Anc_noise1,  loader_Anc_noise2, generator, discriminator, g_optim, d_optim, g_ema_t, g_ema_s, device):

    loader_R_img = sample_data(loader_R_img)
    loader_Anc_noise1 = sample_data(loader_Anc_noise1)
    loader_Anc_noise2 = sample_data(loader_Anc_noise2)
    # requires_grad(g_ema_s, False)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        #g_ema_t = g_ema_t.module
        #g_ema_ts= g_ema_s.module
    else:
        g_module = generator
        d_module = discriminator
        #g_ema_t = g_ema_t
        #g_ema_s= g_ema_s

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    
    ### Creating a list of layers for extracting features used in g_dist_loss. ###
    # requires_grad(g_ema_s, False)
    inter_layers_list = ['to_rgb1.conv.modulation']
    n_dist_layers = int((math.log(args.size, 2) - 2))# (6 at 256=2^8)
    for i in range(n_dist_layers):
        layer_name = 'to_rgbs.'+str(i)+'.conv.modulation' 
        inter_layers_list.append(layer_name) #['to_rgb1.conv.modulation', 'to_rgbs.0.conv.modulation', 'to_rgbs.1.conv.modulation', ..., 'to_rgbs.5.conv.modulation'] =7 layers
    #######
    
    for idx in pbar:
        
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break
    
        real_img = next(loader_R_img)
        real_img = real_img.to(device)

        noises_A1 = next(loader_Anc_noise1)
        noises_A2 = next(loader_Anc_noise2)
        noises_A1 = noises_A1 + 0.05 * torch.randn_like(noises_A1)  # tensor([torch.Size([4, 512],device='cuda:0')
        noises_A2 = noises_A2 + 0.05 * torch.randn_like(noises_A2)
        
        noises_A = ( noises_A1,  noises_A2)
        noises_NA = mixing_noise(args.batch, args.latent , args.mixing, device)  #tensor[[torch.Size([4, 512]),device='cuda:0')],device='cuda:0', [torch.Size([4, 512])],device='cuda:0')]]
        #print('---')
        #print(f'noises_A:{noises_A}')
        #print('---')
        #print(f'noises_NA:{noises_NA}')
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        fake_img_A, _ = generator(noises_A) 
        fake_img_NA, _ = generator(noises_NA)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img_A, _ = augment(fake_img_A, ada_aug_p)
            fake_img_NA, _ = augment(fake_img_NA, ada_aug_p)
        else:
            real_img_aug = real_img

        
        fake_pred_A, fake_pred_patch = discriminator(fake_img_A, fake_img_NA)
        real_pred_A, real_pred_patch = discriminator(real_img_aug, real_img_aug)

        #real_pred_NA = torch.mean(real_pred_patch, [1,2,3]).unsqueeze(1)
        #fake_pred_NA = torch.mean(fake_pred_patch, [1,2,3]).unsqueeze(1)

        '''
        d_logistic_loss
        torch.nn.Softplus(beta=1, threshold=20)
        Applies the element-wise function: Output: same shape as the input
        '''
        d_loss = d_logistic_loss(real_pred_A, fake_pred_A) + d_logistic_loss(real_pred_patch, fake_pred_patch)
        #d_loss = d_logistic_loss(real_pred_A, fake_pred_A)

        loss_dict["d"] = d_loss
        #loss_dict["real_score"] = real_pred.mean() 
        #loss_dict["fake_score"] = fake_pred.mean() 
        loss_dict["real_score"] = real_pred_A.mean() + real_pred_patch.mean()
        loss_dict["fake_score"] = fake_pred_A.mean() + fake_pred_patch.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred_A)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0 # true or false

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred_A, real_pred_patch = discriminator(real_img_aug, real_img_aug)
            #print('---------L.293------------')
            #print(f'iter:{i}')
            #print(f':real_pred_A.size(){real_pred_A.size()}') # [3,1]
            #print(f':real_pred_patch.size(){real_pred_patch.size()}') # [3,1]
            real_pred_NA = torch.mean(real_pred_patch, [1,2,3]).unsqueeze(1)
            #r1_loss = d_r1_loss(real_pred, real_img) 
            my_HyP = 10.0
            real_pred_A = (my_HyP * real_pred_A + real_pred_NA)
            #r1_loss = d_r1_loss(real_pred_A, real_img)  
            r1_loss = d_r1_loss(real_pred_A, real_img_aug)  

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred_A[0]).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        
        noises_NA = mixing_noise(args.batch, args.latent, args.mixing, device)
        #　It would be a better result to have G generate from a different noise each time.
        
        ### My addition ######
        fake_img_A,  _ = generator(noises_A)
        fake_img_NA,  _ = generator(noises_NA)
        src_fake_img_NA,  _ = g_ema_s(noises_NA)
        
        generatorTx = tx.Extractor(generator, inter_layers_list)
        g_ema_sTx = tx.Extractor(g_ema_s, inter_layers_list)

        _, g_features = generatorTx(noises_NA)
        _, g_ema_s_features = g_ema_sTx(noises_NA)

        tgt_features = [t for name, t in g_features.items()]
        tgt_features.append(fake_img_NA)
        src_features = [t for name, t in g_ema_s_features.items()]
        src_features.append(src_fake_img_NA)
        
        tgt_dists = calc_soft_sim(tgt_features)
        src_dists = calc_soft_sim(src_features)
        #print('-----------L330---------')
        #print(f'src_dists:{src_dists}')
        g_Dist_loss = g_dist_loss(tgt_dists, src_dists) ## In each layer, Difference in the distribution of products between a generator that has been trained in the source domain and a generator that is being trained in the target domain (made from the same noises_NA).
        
        #######

        if args.augment:
            fake_img_A, _ = augment(fake_img_A, ada_aug_p)
            fake_img_NA, _ = augment(fake_img_NA, ada_aug_p)

        fake_pred_A, fake_pred_patch = discriminator(fake_img_A, fake_img_NA)
        
        ### In the paper, "Empirically, we find that a high λ,from 10^3 to 10^4, to work well." ###
        lmd = 1000  
        g_loss_Dimg = g_nonsaturating_loss(fake_pred_A)
        g_loss_Dpatch = g_nonsaturating_loss(fake_pred_patch)
        print('-----------L346---------')
        print(f'g_loss_Dimg:{g_loss_Dimg}')
        print(f'g_loss_Dpatch:{g_loss_Dpatch}')
        print(f'lmd * g_Dist_loss:{lmd * g_Dist_loss}')
        #g_loss = g_nonsaturating_loss(fake_pred_A) + lmd * g_Dist_loss
        g_loss = g_nonsaturating_loss(fake_pred_A) + g_nonsaturating_loss(fake_pred_patch) + lmd * g_Dist_loss
        
        #######

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noises_NA = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img_NA, latents = generator(noises_NA, return_latents=True)
            
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_NA, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_NA[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        # This area seems to be for distributed learning, so I won't touch it.
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_t, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                
                with torch.no_grad():
                    g_ema_t.eval()
                    g_ema_s.eval()
                    
                    sample_z = torch.randn(args.n_sample, args.latent, device=device) # torch.Size([4, 512])

                    sample1, _ = g_ema_t([sample_z])
                    sample2, _ = g_ema_s([sample_z])
                    sample = torch.cat([sample2, sample1], dim=2)
                    
                    utils.save_image(
                        sample,
                        args.sample_dir + f"/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 1000 == 0 and i != 0:
                
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema_t.state_dict(), # don't save in g_ema_s.
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    args.ckpt_t.split('/')[0] + f"/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    
    ### You just to write these 3. #############################################
    parser.add_argument(
        "--iter", type=int, default=560001, help="total training iterations"
    )
    parser.add_argument(
        "--ckpt_t",
        type=str,
        default="checkpoint_arcim_2nd/550000.pt",
        help="path to the checkpoints except for g_ema_t(fixed for source image generation) to resume training",
    )
    parser.add_argument(
        "--sample_dir", 
        type=str,
        default="sample_arcim_2nd",
        help="output_images_dir",
    )
    ##################################################################
    parser.add_argument(
        "--ckpt_s",
        type=str,
        default="550000.pt",
        help="path to the checkpoints for g_ema_s(fixed for source image generation) to resume training",
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=9,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_t.split('/')[0], exist_ok=True)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema_t = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    g_ema_s = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device) # to just writeout source images.
    
    g_ema_t.eval()
    accumulate(g_ema_t, generator, 0)# They're using moving averages to learn.
    
    g_ema_s.eval()
    requires_grad(g_ema_s, flag=False)
    
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    #if args.ckpt is not None and args.ckpt_source is not None:

    if args.ckpt_t is not None and args.ckpt_s is not None:
        shutil.copy(args.ckpt_s, args.ckpt_t)
        print("load model:", args.ckpt_t)
        print("load_s model:", args.ckpt_s)

        ckpt_t = torch.load(args.ckpt_t, map_location=lambda storage, loc: storage)
        ckpt_s = torch.load(args.ckpt_s, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt_t)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt_t["g"], strict=False)
        discriminator.load_state_dict(ckpt_t["d"], strict=False)

        g_ema_t.load_state_dict(ckpt_t["g_ema"], strict=False)
        g_ema_s.load_state_dict(ckpt_s["g_ema"], strict=False)
    
        g_optim.load_state_dict(ckpt_t["g_optim"])
        d_optim.load_state_dict(ckpt_t["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset_R = MultiResolutionDataset(args.path, transform, args.size)#target Image
    #print(f'len_dataset_M:{len(dataset_M)}')
    dataset_A = AncherDataset(args, device, len(dataset_R))# fixed noise in ancher　region
    #print(f'len_dataset_A:{len(dataset_A)}')
    
    loader_R_img = data.DataLoader(
        dataset_R,
        batch_size=args.batch,
        sampler=data_sampler(dataset_R, shuffle=False, distributed=False),
        drop_last=False,
    )

    loader_Anc_noise1 = data.DataLoader(
        dataset_A,
        batch_size=args.batch,
        sampler=data_sampler(dataset_A, shuffle=False, distributed=False),
        drop_last=False,
    )

    loader_Anc_noise2 = data.DataLoader(
        dataset_A,
        batch_size=args.batch,
        sampler=data_sampler(dataset_A, shuffle=False, distributed=False),
        drop_last=False,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    #tx.list_module_names(generator)
    #for name, module in generator.named_modules():
        #print(name)
    '''
    print("g_ema_s's state_dict:")
    for param_tensor in g_ema.state_dict():
            print(param_tensor, "\t", g_ema_s.state_dict()[param_tensor].size())
    print('-----------------')
    print("generator's state_dict:")
    
    for param_tensor in generator.state_dict():
            print(param_tensor, "\t", generator.state_dict()[param_tensor].size())
    print('-----------------')  
    '''
    
    train(args, loader_R_img, loader_Anc_noise1, loader_Anc_noise2, generator, discriminator, g_optim, d_optim, g_ema_t, g_ema_s, device)
