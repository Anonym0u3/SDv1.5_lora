export MODEL_NAME="path_to_stable-diffusion-v1-5"
export OUTPUT_DIR="path_to_output_dir"
export DATASET_NAME="path_to_pokemon-blip-captions"

accelerate launch --mixed_precision="bf16" --num_processes=1 --num_machines=1 --dynamo_backend=no train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 \
  --rank=4 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpoints_total_limit=5 \
  --validation_prompt="A pokemon with blue eyes." \
  --num_validation_images=4 \
  --validation_epochs=30 \
  --seed=1337 \
  --enable_xformers_memory_efficient_attention \
  --report_to="wandb" \