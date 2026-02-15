export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export WIN_UNET_PATH="tmp-sdxl-cpgd-win"
export PROMPT=""
export SAVE_PATH="tmp-sdxl-pgd-win.png"
export PGD_SCALE=10

python inference/sdxl_pgd_inference.py \
  --model_id=$MODEL_NAME \
  --win_unet_path=$WIN_UNET_PATH \
  --prompt=$PROMPT \
  --save_path=$SAVE_PATH \
  --pgd_scale=$PGD_SCALE