<<<<<<< HEAD
generate.py

=======
>>>>>>> origin/main
import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import os

<<<<<<< HEAD
def generate(args, g_base,  g_trained, device, mean_latent1, mean_latent2, outdir):
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        g_base.eval()
        g_trained.eval()

        image_list = []
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample1, _ = g_base(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent1)
            sample2, _ = g_trained(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent2)

            sample = torch.cat([sample1, sample2], dim=2)
            image_list.append(sample)
        
        out_sample = torch.cat(image_list, dim=3)
        utils.save_image(
            out_sample,
            outdir+f"/{str(i).zfill(6)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
=======
def generate(args, g_ema, device, mean_latent, outdir):
    os.makedirs(outdir, exist_ok=True)
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            utils.save_image(
                sample,
                outdir+f"/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
>>>>>>> origin/main


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=10, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt1",
        type=str,
<<<<<<< HEAD
        default="checkpoint_gob/550000.pt",
=======
        default="checkpoint/570000.pt",
>>>>>>> origin/main
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--ckpt2",
        type=str,
<<<<<<< HEAD
        default="checkpoint_gob/551000.pt",
=======
        default="checkpoint/550000.pt",
>>>>>>> origin/main
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

<<<<<<< HEAD
    g_base = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    g_trained = Generator(
=======
    g_ema1 = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    g_ema2 = Generator(
>>>>>>> origin/main
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    checkpoint1 = torch.load(args.ckpt1)
    checkpoint2 = torch.load(args.ckpt2)

<<<<<<< HEAD
    g_base.load_state_dict(checkpoint1["g_ema"])
    #print("Model's state_dict:")
    #for param_tensor in g_base.state_dict():
        #print(param_tensor, "\t", g_base.state_dict()[param_tensor].size())

    #print('-----------------')

    #g_trained.load_state_dict(checkpoint2["g_ema"], strict=False)
    #for param_tensor in g_trained.state_dict():
        #print(param_tensor, "\t", g_trained.state_dict()[param_tensor].size())

    #print('-----------------')
=======
    g_ema1.load_state_dict(checkpoint1["g"])
    print("Model's state_dict:")
    for param_tensor in g_ema1.state_dict():
            print(param_tensor, "\t", g_ema1.state_dict()[param_tensor].size())

    print('-----------------')

    g_ema2.load_state_dict(checkpoint2["g_ema"], strict=False)
    for param_tensor in g_ema2.state_dict():
        print(param_tensor, "\t", g_ema2.state_dict()[param_tensor].size())

    print('-----------------')
>>>>>>> origin/main


    if args.truncation < 1:
        with torch.no_grad():
<<<<<<< HEAD
            mean_latent1= g_base.mean_latent(args.truncation_mean)
            mean_latent2 = g_trained.mean_latent(args.truncation_mean)
    else:
        mean_latent1 = None
        mean_latent2 = None

    generate(args, g_base, g_trained, device, mean_latent1, mean_latent2,  "cat_sample_gob")
=======
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema1, device, mean_latent, "sample1")
    generate(args, g_ema2, device, mean_latent, "sample2")
>>>>>>> origin/main
