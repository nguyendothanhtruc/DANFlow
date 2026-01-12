CHECKPOINT_DIR =  "ckpt"
EVAL_PATH = "eval" #Folder to store abnormaly map

#####################################################################
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

BTAD_CATEGORIES = [
    "01",
    "02",
    "03"
]

KLTSDD_CATEGORIES = [
    "transformed",
]

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum"

]
#####################################################################

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "resnet50"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_RESNET18,
]

#####################################################################
BATCH_SIZE = 32
NUM_EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 1
CHECKPOINT_INTERVAL = 1