import os

MODELS = [
    "edsr",
    "swinir",
    "srgan_resnet",
]

ROOT_DIR_PATH = os.path.abspath('/home/featurize') # modifiable

OUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, "out", "AI6126project2")
OUT_EDSR_DIR_PATH = os.path.join(OUT_DIR_PATH, "edsr")
OUT_SWINIR_DIR_PATH = os.path.join(OUT_DIR_PATH, "swinir")
OUT_SRGAN_DIR_PATH = os.path.join(OUT_DIR_PATH, "srgan_resnet")
OUT_MODEL_DIR_PATHS = dict(
    edsr = OUT_EDSR_DIR_PATH,
    swinri = OUT_SWINIR_DIR_PATH,
    srgan_resnet = OUT_SRGAN_DIR_PATH,
)

RES_DIR_PATH = os.path.join(ROOT_DIR_PATH, "res", "AI6126project2")
RES_EDSR_DIR_PATH = os.path.join(RES_DIR_PATH, "edsr")
RES_SWINIR_DIR_PATH = os.path.join(RES_DIR_PATH, "swinir")
RES_SRGAN_DIR_PATH = os.path.join(RES_DIR_PATH, "srgan_resnet")
RES_MODEL_DIR_PATHS = dict(
    edsr = RES_EDSR_DIR_PATH,
    swinir = RES_SWINIR_DIR_PATH,
    srgan_resnet = RES_SRGAN_DIR_PATH,
)

DATASET_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data", "ffhq")
DATASET_TRAIN_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train")
DATASET_VAL_DIR_PATH= os.path.join(DATASET_DIR_PATH, "val")
DATASET_TEST_DIR_PATH = os.path.join(DATASET_DIR_PATH, "test")
DATASET_TEST_REAL_DIR_PATH = os.path.join(DATASET_DIR_PATH, "test_real")


PROJECT_ROOT_DIR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), ".."
    )
)


MMAGIC_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, "mmagic")
MMAGIC_CONFIGS_DIR_PATH = os.path.join(MMAGIC_DIR_PATH, "configs")


SRC_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, "src")
SRC_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "models_analyses_dict.py")
SRC_FILTERED_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "filtered_models_analyses_dict.py")
SRC_DATA_CFG_FILE_PATH = os.path.join(SRC_DIR_PATH, "data_cfg.py")
SRC_VIS_CFG_FILE_PATH = os.path.join(SRC_DIR_PATH, "vis_cfg.py")


ASSETS_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, "assets")
ASSETS_CONFIGS_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "configs")
ASSETS_CONFIGS_EDSR_DIR_PATH = os.path.join(ASSETS_CONFIGS_DIR_PATH, "edsr")
ASSETS_CONFIGS_SWINIR_DIR_PATH = os.path.join(ASSETS_CONFIGS_DIR_PATH, "swinir")
ASSETS_CONFIGS_SRGAN_DIR_PATH = os.path.join(ASSETS_CONFIGS_DIR_PATH, "srgan_resnet")

ASSETS_CHECKPOINT_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "checkpoints")
ASSETS_CHECKPOINT_EDSR_DIR_PATH = os.path.join(ASSETS_CHECKPOINT_DIR_PATH, "edsr")
ASSETS_CHECKPOINT_SWINIR_DIR_PATH = os.path.join(ASSETS_CHECKPOINT_DIR_PATH, "swinir")
ASSETS_CHECKPOINT_SRGAN_DIR_PATH = os.path.join(ASSETS_CHECKPOINT_DIR_PATH, "srgan_resnet")

MAKEDIRS = [
    OUT_DIR_PATH,
    OUT_EDSR_DIR_PATH,
    OUT_SWINIR_DIR_PATH,
    OUT_SRGAN_DIR_PATH,
    ASSETS_DIR_PATH,
    ASSETS_CONFIGS_DIR_PATH,
    ASSETS_CHECKPOINT_DIR_PATH,
    ASSETS_CONFIGS_EDSR_DIR_PATH,
    ASSETS_CONFIGS_SWINIR_DIR_PATH,
    ASSETS_CONFIGS_SRGAN_DIR_PATH,
    ASSETS_CHECKPOINT_EDSR_DIR_PATH,
    ASSETS_CHECKPOINT_SWINIR_DIR_PATH,
    ASSETS_CHECKPOINT_SRGAN_DIR_PATH,
    RES_DIR_PATH,
    RES_EDSR_DIR_PATH,
    RES_SWINIR_DIR_PATH,
    RES_SRGAN_DIR_PATH,
]

for makedir in MAKEDIRS:
    if not os.path.exists(makedir):
        os.makedirs(makedir)
        print(f"Directories created:{makedir}")
            
    
