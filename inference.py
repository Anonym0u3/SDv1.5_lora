from diffusers import StableDiffusionPipeline
import torch

lora_model_path = "path_to_lora_model" #pytorch_lora_weights.safetensors
sd_model_path = "path_to_stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(lora_model_path, weight_name="pytorch_lora_weights.safetensors")
prompt = "a cute pokemon"
image = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5,seed=77).images[0]
image.save("pokemon.png")