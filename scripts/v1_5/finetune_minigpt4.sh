



python /scratch1/runhuixu/LLaVA/llava/train/finetune_minigpt4.py \
  --model_id deepseek-ai/deepseek-vl-7b-base \
  --data_path /project2/ruishanl_1185/SDP_for_VLM/datasets/physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/chat_train_p10_filtered.json \
  --image_folder /project2/ruishanl_1185/SDP_for_VLM/datasets/mimic-cxr-jpg/mimic-cxr-jpg/2.1.0/files \
  --output_dir /scratch1/runhuixu/outputs/minigpt4_mimic_fullft \
  --include_reason True \
  --filter_views True \
  --generation_methods gpt4 \
  --prompt_style minigpt4 \
  --image_token "<image>"
