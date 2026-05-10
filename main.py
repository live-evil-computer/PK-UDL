MODE = 'train'

if __name__ == '__main__':
    if MODE == 'train':
        run_pipeline()
    elif MODE == 'predict':
        model_file = os.path.join(Config.MODEL_SAVE_DIR, "")
        input_tif = ""
        output_tif = ""
        if os.path.exists(input_tif):
            run_inference(model_file, input_tif, output_tif)
        else:
            print(f"❌ Input file not found: {input_tif}")