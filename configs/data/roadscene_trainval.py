# configs/data/roadscene_trainval.py
#
# Data config for fine-tuning JamMa on the RoadScene IR/Visible dataset.
# Run roadscene_to_megadepth.py first to generate the index files,
# then update the paths below to match your output_dir.
#
# Usage:
#   python train.py \
#       configs/data/roadscene_trainval.py \
#       configs/jamma/outdoor.py \
#       --exp_name roadscene_finetune \
#       --ckpt_path /path/to/jamma_pretrained.ckpt \
#       --gpus 1 --batch_size 2 --max_epochs 30

from configs.data.base import cfg

# ── Update these two paths to wherever you ran roadscene_to_megadepth.py ──
ROADSCENE_ROOT   = "/path/to/data/roadscene"   # <-- change this
TRAIN_BASE_PATH  = f"{ROADSCENE_ROOT}/index"
TEST_BASE_PATH   = f"{ROADSCENE_ROOT}/index"

# ── Training data ──────────────────────────────────────────────────────────
cfg.DATASET.TRAINVAL_DATA_SOURCE    = "MegaDepth"
cfg.DATASET.TRAIN_DATA_ROOT         = f"{ROADSCENE_ROOT}/train"
cfg.DATASET.TRAIN_NPZ_ROOT          = f"{TRAIN_BASE_PATH}/scene_info"
cfg.DATASET.TRAIN_LIST_PATH         = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0   # all pairs are fully overlapping

# ── Validation / Test data ─────────────────────────────────────────────────
cfg.DATASET.TEST_DATA_SOURCE        = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT           = \
cfg.DATASET.TEST_DATA_ROOT          = f"{ROADSCENE_ROOT}/test"
cfg.DATASET.VAL_NPZ_ROOT            = \
cfg.DATASET.TEST_NPZ_ROOT           = f"{TEST_BASE_PATH}/scene_info"
cfg.DATASET.VAL_LIST_PATH           = \
cfg.DATASET.TEST_LIST_PATH          = f"{TEST_BASE_PATH}/trainvaltest_list/test_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST  = 0.0

# ── Loader settings ────────────────────────────────────────────────────────
# Each "scene" has exactly 1 pair (IR + visible), so keep this at 1.
cfg.TRAINER.N_SAMPLES_PER_SUBSET    = 1

# Resize images to this size during training (must be divisible by 8).
cfg.DATASET.MGDPT_IMG_RESIZE        = 640
