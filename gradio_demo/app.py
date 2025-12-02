import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_IDM_VTON_ROOT = _THIS_DIR.parent
_PROJECT_ROOT = _THIS_DIR.parents[3]
_REPO_ROOT = _THIS_DIR.parents[2]

_DENSEPOSE_CFG = _IDM_VTON_ROOT / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
_DENSEPOSE_CKPT = _IDM_VTON_ROOT / "ckpt" / "densepose" / "model_final_162be9.pkl"

for path in (_IDM_VTON_ROOT, _REPO_ROOT, _PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_vlm_model = None
_vlm_processor = None


def _vlm_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is None or _vlm_processor is None:
        dtype = _vlm_dtype()
        _vlm_processor = AutoProcessor.from_pretrained(
            VLM_MODEL_ID,
            trust_remote_code=True,
        )
        _vlm_model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            _vlm_model = _vlm_model.to(dtype=dtype)
        _vlm_model.eval()
    return _vlm_model, _vlm_processor


def _clean_vlm_response(text: str) -> str:
    if not text:
        return "a garment"
    cleaned = text.strip()
    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:")[-1].strip()
    if cleaned.lower().startswith("assistant"):
        cleaned = cleaned.split(":", 1)[-1].strip()
    if not cleaned:
        return "a garment"
    return cleaned.replace("\n", " ").strip()


def generate_garment_description(image: Image.Image | None) -> str:
    if image is None:
        return "a garment"
    try:
        model, processor = _load_vlm()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to load VLM: {exc}")
        return "a garment"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe the garment in this image for a virtual try-on system. "
                        "Mention color, garment type, sleeve/fit, and distinguishing details in ~15 words."
                    ),
                },
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        processed_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if value.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    processed_inputs[key] = value.to(model.device, dtype=model.dtype)
                else:
                    processed_inputs[key] = value.to(model.device)
            else:
                processed_inputs[key] = value
        inputs = processed_inputs
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.2,
                top_p=0.85,
            )
        prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        new_tokens = generated_ids[:, prompt_length:]
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
        return _clean_vlm_response(decoded[0] if decoded else "")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to generate garment description: {exc}")
        return "a garment"


def _extract_human_image(editor_state):
    if isinstance(editor_state, Image.Image):
        return editor_state.convert("RGB")
    if isinstance(editor_state, dict):
        for key in ("image", "background", "composite"):
            candidate = editor_state.get(key)
            if isinstance(candidate, Image.Image):
                return candidate.convert("RGB")
        layers = editor_state.get("layers")
        if isinstance(layers, list):
            for layer in layers:
                if isinstance(layer, Image.Image):
                    return layer.convert("RGB")
    raise ValueError("No person image supplied. Please upload a photo.")


def _extract_manual_mask(editor_state):
    if not isinstance(editor_state, dict):
        return None
    layers = editor_state.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, Image.Image):
                return layer.convert("L")
    mask = editor_state.get("mask")
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    return None


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def start_tryon(editor_state,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    garment_des = (garment_des or "").strip()
    if not garment_des:
        garment_des = generate_garment_description(garm_img)

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = _extract_human_image(editor_state)
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    manual_mask_img = _extract_manual_mask(editor_state)
    use_auto_mask = is_checked or manual_mask_img is None

    if use_auto_mask:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(manual_mask_img.resize((768, 1024)))
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = apply_net.create_argument_parser().parse_args(
        (
            'show',
            str(_DENSEPOSE_CFG),
            str(_DENSEPOSE_CKPT),
            'dp_segm',
            '-v',
            '--opts',
            'MODEL.DEVICE',
            'cuda',
        )
    )
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray, garment_des
    else:
        return images[0], mask_gray, garment_des
    # return images[0], mask_gray


def update_description(garm_img):
    return generate_garment_description(garm_img)

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources='upload',
                type="pil",
                label='Human (auto mask enabled by default). Draw only if you want a custom mask.',
                interactive=True,
            )
            with gr.Row():
                is_checked = gr.Checkbox(label="Use auto-generated mask", info="Recommended",value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            garment_desc = gr.Textbox(
                label="Auto garment description",
                placeholder="Description will appear after upload",
                interactive=False,
            )
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
            garm_img.change(fn=update_description, inputs=[garm_img], outputs=[garment_desc])
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img")
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img")




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)



    try_button.click(
        fn=start_tryon,
        inputs=[imgs, garm_img, garment_desc, is_checked, is_checked_crop, denoise_steps, seed],
        outputs=[image_out, masked_img, garment_desc],
        api_name='tryon'
    )

            


image_blocks.launch(server_name="0.0.0.0", server_port=9090, share=True)

