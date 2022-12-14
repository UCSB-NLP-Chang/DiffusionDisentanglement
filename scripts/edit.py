import argparse, os
import torch
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def final_encode_and_save_noise(opt):
    # input: image path, text (c1)
    # output: saved_noise in noise_saved_path
    seed_everything(opt.seed)
    input_image = opt.input
    text = opt.c1
    noise_saved_path = "noise"
    if not os.path.exists(noise_saved_path):
        os.makedirs(noise_saved_path, exist_ok=True)
    noise_saved_path += "/noise.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(
        OmegaConf.load("configs/stable-diffusion/v1-inference.yaml"),
        "models/ldm/stable-diffusion-v1/model.ckpt",
    ).to(device)
    sampler = PLMSSampler(model)
    num_samples = 1  # also known as batch size
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

    text_embed = model.get_learned_conditioning(num_samples * [text])
    input_image_source = load_img(input_image).to(device)
    noised_image_encode = model.get_first_stage_encoding(
        model.encode_first_stage(input_image_source)
    )
    shape = [4, 64, 64]
    start_code = None
    uc = model.get_learned_conditioning(num_samples * [""])

    with torch.no_grad():
        samples_ddim, _ = sampler.sample_encode_save_noise(
            S=50,
            conditioning=text_embed,
            batch_size=num_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=start_code,
            input_image=noised_image_encode,
            noise_save_path=noise_saved_path,
        )


def final_edit_image(opt):
    steps = 50
    seed_everything(opt.seed)
    original_text = opt.c1
    new_text = opt.c2
    noise_saved_path = "noise/noise.pt"
    lambda_t = [opt.lambda_t_default_1] * opt.lambda_t_star + [
        opt.lambda_t_default_2
    ] * (
        steps - opt.lambda_t_star
    )  # lambda_t initialization
    lambda_t = torch.tensor(lambda_t)
    # save intermediate optimization image and weight
    lambda_save_path = os.path.join(opt.outdir, "lambda")
    image_save_path = os.path.join(opt.outdir, "image")
    if not os.path.exists(lambda_save_path):
        os.makedirs(lambda_save_path, exist_ok=True)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
    image_output_path = opt.outdir
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path, exist_ok=True)
        # os.makedirs(image_output_path + "/each", exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model_from_config(
        OmegaConf.load("configs/stable-diffusion/v1-inference.yaml"),
        "models/ldm/stable-diffusion-v1/model.ckpt",
    ).to(device)
    sampler = PLMSSampler(model)
    num_samples = 1
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark("bytes", wm.encode("utf-8"))
    noised_image_encode = None
    shape = [4, 64, 64]
    c1 = model.get_learned_conditioning(num_samples * [original_text])
    c2 = model.get_learned_conditioning(num_samples * [new_text])
    start_code = torch.load(noise_saved_path + "_final_latent.pt")
    uc = model.get_learned_conditioning(num_samples * [""])
    sampler.sample_optimize_intrinsic_edit(
        S=50,
        conditioning1=c1,
        conditioning2=c2,
        batch_size=num_samples,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=7.5,
        unconditional_conditioning=uc,
        eta=0.0,
        x_T=start_code,
        input_image=noised_image_encode,
        noise_save_path=None,
        lambda_t=lambda_t,
        lambda_save_path=lambda_save_path,
        image_save_path=image_save_path,
        original_text=original_text,
        new_text=new_text,
        otext=original_text,
        noise_saved_path=noise_saved_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c1",
        type=str,
        nargs="?",
        default="A photo of person",
        help="The text to synthesize original image.",
    )
    parser.add_argument(
        "--c2",
        type=str,
        nargs="?",
        default="A photo of person, smiling",
        help="The text modifies from c1, containing target attribute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        default=42,
        help="The seed. Particularly useful to control original image.",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="?",
        default="input/test.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--lambda_t_star",
        type=int,
        nargs="?",
        default=20,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--lambda_t_default_1",
        type=float,
        nargs="?",
        default=1.0,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--lambda_t_default_2",
        type=float,
        nargs="?",
        default=0.0,
        help="lambda initialization.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/edit",
    )

    opt = parser.parse_args()

    final_encode_and_save_noise(opt)
    final_edit_image(opt)
