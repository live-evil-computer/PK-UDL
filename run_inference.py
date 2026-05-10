def run_inference(model_path, tif_path, output_path):
    print(f"\n🚀 Starting Inference on {os.path.basename(tif_path)}")
    model = UNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    with rasterio.open(tif_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    c, h, w = image.shape
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.uint16)

    patch_size = ……
    stride = ……
    print(f"   Image Shape: {h}x{w}, Patch: {patch_size}, Stride: {stride}")

    with torch.no_grad():
        for r in tqdm(range(0, h - patch_size + 1, stride), desc="Sliding Window"):
            for c in range(0, w - patch_size + 1, stride):
                img_patch = image[:, r:r+patch_size, c:c+patch_size]
                tensor = torch.from_numpy(img_patch).unsqueeze(0).to(Config.DEVICE)
                pred = torch.softmax(model(tensor), dim=1)[:, 1, :, :].cpu().numpy().squeeze()
                prob_map[r:r+patch_size, c:c+patch_size] += pred
                count_map[r:r+patch_size, c:c+patch_size] += 1

    count_map[count_map == 0] = 1
    result = prob_map / count_map

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result.astype(np.float32), 1)
    print(f"✅ Prediction Saved: {output_path}")
