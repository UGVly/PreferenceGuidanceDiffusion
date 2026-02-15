export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export PGD_UNET_PATH="tmp-sdxl-cpgd-win"
export LOSE_UNET_PATH="tmp-sdxl-cpgd-lose"
export PROMPT=""
export SAVE_PATH="tmp-sdxl-cpgd-win.png"
export PGD_SCALE=10

python inference/sdxl_cpgd_inference.py \
  --model_id=$MODEL_NAME \
  --pgd_unet_path=$PGD_UNET_PATH \
  --lose_unet_path=$LOSE_UNET_PATH \
  --prompt=$PROMPT \
  --save_path=$SAVE_PATH \
  --pgd_scale=$PGD_SCALE