class Config:
    # --- 路径配置 ---
    BASE_DIR = ''
    MODEL_SAVE_DIR = ''

    # --- 数据配置 ---
    REGIONS = ['', '', '', '']
    IN_CHANNELS = 
    NUM_CLASSES = 

    # --- 训练超参 ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROUNDS = 
    EPOCHS = 
    BATCH_SIZE = 
    LR = 1e-4

    # --- 测试集比例 ---
    TEST_RATIO = 
    VAL_RATIO = 

    @staticmethod
    def mkdirs():
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

Config.mkdirs()