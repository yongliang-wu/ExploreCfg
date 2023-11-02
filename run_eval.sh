LM_PATH="/home/wyl/work/code/open_flamingo/llama-7b" # llama model path
LM_TOKENIZER_PATH="/home/wyl/work/code/open_flamingo/llama-7b" # llama model path 
CKPT_PATH="/home/wyl/work/code/open_flamingo/openflamingo_checkpoint.pt"
# checkpoint model path you can run checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt") to get
DEVICE=0 # gpu num

COCO_IMG_PATH="/home/wyl/work/data/coco_data/train2014" # coco dataset
COCO_ANNO_PATH="/home/wyl/work/data/coco_data/annotations/captions_train2014.json" # coco dataset

RANDOM_ID="RS"
RESULTS_FILE="results_${RANDOM_ID}.json"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --device $DEVICE \
    --coco_image_dir_path $COCO_IMG_PATH \
    --coco_annotations_json_path $COCO_ANNO_PATH \
    --mgc_path  "MGC/wc_vis_135.json"\
    --mgca_path  "MGCA-idx/best_gt_WC(135).json"\
    --clip_ids_path "train_set_clip.json"
    --results_file $RESULTS_FILE \
    --num_samples 5000 --shots 4 8 16 32 --num_trials 1 --seed 5 --batch_size 8\
    --cross_attn_every_n_layers 4\
    --eval_coco
    
echo "evaluation complete! results written to $RESULTS_FILE"
