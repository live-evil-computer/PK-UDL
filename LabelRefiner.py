class LabelRefiner:
    @staticmethod
    def update(model, image_paths, label_paths, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        device = Config.DEVICE

        with torch.no_grad():
            for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Updating Labels", leave=False):
                with rasterio.open(img_path) as src: image = src.read().astype(np.float32)
                with rasterio.open(lbl_path) as src: label = src.read(1).astype(np.uint8)

                img_t = torch.from_numpy(image).unsqueeze(0).to(device)
                prob_map = torch.sigmoid(model(img_t).squeeze(0)[1]).cpu().numpy()

                updated_label = label.copy()
                flat_probs = prob_map.ravel()

                mask1 = (label == 1)
                if mask1.sum() > 0:
                    thr1 = np.percentile(flat_probs[mask1.ravel()], ……)
                    updated_label[mask1 & (prob_map <= thr1) & (prob_map < ……)] = 2

                mask2 = (label == 2)
                if mask2.sum() > 0:
                    hi_thr = np.percentile(flat_probs[mask2.ravel()], ……)
                    lo_thr = np.percentile(flat_probs[mask2.ravel()], ……)
                    updated_label[mask2 & (prob_map >= hi_thr) & (prob_map > ……)] = 1
                    updated_label[mask2 & (prob_map <= lo_thr) & (prob_map < ……)] = 0

                mask0 = (label == 0)
                if mask0.sum() > 0:
                    hi_thr0 = np.percentile(flat_probs[mask0.ravel()], ……)
                    updated_label[mask0 & (prob_map >= hi_thr0) & (prob_map > ……)] = 2

                profile = rasterio.open(lbl_path).profile
                profile.update(count=1, dtype=rasterio.uint8)
                with rasterio.open(os.path.join(save_dir, os.path.basename(lbl_path)), 'w', **profile) as dst:
                    dst.write(updated_label.astype(np.uint8), 1)