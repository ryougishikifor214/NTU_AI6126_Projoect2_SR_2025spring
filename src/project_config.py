import os

ROOT_DIR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), ".."
    )
)
MMAGIC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "mmagic")
MMAGIC_CONFIGS_DIR_PATH = os.path.join(MMAGIC_DIR_PATH, "configs")

SRC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "src")
SRC_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "models_analyses_dict.py")
SRC_FILTERED_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "filtered_models_analyses_dict.py")
SRC_DATA_CFG_FILE_PATH = os.path.join(SRC_DIR_PATH, "data_cfg.py")

DATASET_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data")
DATASET_TRAIN_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train")
DATASET_VAL_DIR_PATH= os.path.join(DATASET_DIR_PATH, "val")

OUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, "out")
OUT_EDSR_DIR_PATH = os.path.join(OUT_DIR_PATH, "edsr")

ASSETS_DIR_PATH = os.path.join(ROOT_DIR_PATH, "assets")
ASSETS_CONFIGS_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "configs")
ASSETS_CONFIGS_EDSR_DIR_PATH = os.path.join(ASSETS_CONFIGS_DIR_PATH, "edsr")
ASSETS_CHECKPOINT_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "checkpoints")
ASSETS_CHECKPOINT_EDSR_DIR_PATH = os.path.join(ASSETS_CHECKPOINT_DIR_PATH, "edsr")

MAKEDIRS = [
    OUT_DIR_PATH,
    OUT_EDSR_DIR_PATH,
    ASSETS_DIR_PATH,
    ASSETS_CONFIGS_DIR_PATH,
    ASSETS_CHECKPOINT_DIR_PATH,
    ASSETS_CONFIGS_EDSR_DIR_PATH,
    ASSETS_CHECKPOINT_EDSR_DIR_PATH,
]

for makedir in MAKEDIRS:
    if not os.path.exists(makedir):
        os.makedirs(makedir)
        print(f"Directories created:{makedir}")

# from dataclasses import dataclass

# @dataclass(frozen=True)
# class ProjectConfig:
    
#     ROOT_DIR_PATH = os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__), ".."
#         )
#     )
#     MMAGIC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "mmagic")
#     MMAGIC_CONFIGS_DIR_PATH = os.path.join(MMAGIC_DIR_PATH, "configs")
    
#     SRC_DIR_PATH = os.path.join(ROOT_DIR_PATH, "src")
#     SRC_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "models_analyses_dict.py")
#     SRC_FILTERED_MODELS_ANALYSES_FILE_PATH = os.path.join(SRC_DIR_PATH, "filtered_models_analyses_dict.py")
#     SRC_DATA_CFG_FILE_PATH = os.path.join(SRC_DIR_PATH, "data_cfg.py")
    
#     DATASET_DIR_PATH = os.path.join(ROOT_DIR_PATH, "data")
#     DATASET_TRAIN_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train")
#     DATASET_VAL_DIR_PATH= os.path.join(DATASET_DIR_PATH, "val")
    
#     OUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, "out")
#     OUT_EDSR_DIR_PATH = os.path.join(OUT_DIR_PATH, "edsr")
    
#     ASSETS_DIR_PATH = os.path.join(ROOT_DIR_PATH, "assets")
#     ASSETS_CONFIGS_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "configs")
#     ASSETS_CONFIGS_EDSR_DIR_PATH = os.path.join(ASSETS_CONFIGS_DIR_PATH, "edsr")
#     ASSETS_CHECKPOINT_DIR_PATH = os.path.join(ASSETS_DIR_PATH, "checkpoints")
#     ASSETS_CHECKPOINT_EDSR_DIR_PATH = os.path.join(ASSETS_CHECKPOINT_DIR_PATH, "edsr")
    
#     MAKEDIRS = [
#         OUT_DIR_PATH,
#         OUT_EDSR_DIR_PATH,
#         ASSETS_DIR_PATH,
#         ASSETS_CONFIGS_DIR_PATH,
#         ASSETS_CHECKPOINT_DIR_PATH,
#         ASSETS_CONFIGS_EDSR_DIR_PATH,
#         ASSETS_CHECKPOINT_EDSR_DIR_PATH,
#     ]
    
#     _instance = None
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(ProjectConfig, cls).__new__(cls)
#             cls._instance._initialize()
#         return cls._instance
    
#     def _initialize(self):
#         for makedir in self.MAKEDIRS:
#             if not os.path.exists(makedir):
#                 os.makedirs(makedir)
#                 print(f"Directories created:{makedir}")
            
    
