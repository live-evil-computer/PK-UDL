class DataManager:
    @staticmethod
    def split_data_once():
        all_samples_with_gt = []
        all_samples_no_gt = []

        print("🔍 Scanning Data & Auto-Detecting GT...")
        for region in Config.REGIONS:
            img_dir = os.path.join(Config.BASE_DIR, f'{region}_image_patches')
            gt_dir  = os.path.join(Config.BASE_DIR, f'{region}_gt_patches')

            has_gt_folder = os.path.exists(gt_dir)
            if has_gt_folder:
                print(f"  -> [{region}] detected GT folder. Candidates for Test Set.")
            else:
                print(f"  -> [{region}] no GT detected. Training only.")

            if not os.path.exists(img_dir): continue

            valid_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
            for f in valid_files:
                fid = f.split('_')[1].split('.')[0]
                sample = {'region': region, 'id': fid}
                if has_gt_folder and os.path.exists(os.path.join(gt_dir, f'gt_{fid}.tif')):
                    all_samples_with_gt.append(sample)
                else:
                    all_samples_no_gt.append(sample)

        print(f"  -> Total samples with GT: {len(all_samples_with_gt)}")
        print(f"  -> Total samples w/o GT:  {len(all_samples_no_gt)}")

        if len(all_samples_with_gt) > 0:
            train_val_gt, test_set = train_test_split(
                all_samples_with_gt, test_size=Config.TEST_RATIO, random_state=42
            )
        else:
            train_val_gt, test_set = [], []
            print("⚠️ Warning: No GT data found! Test set is empty.")

        train_val_candidates = train_val_gt + all_samples_no_gt
        train_set, val_set = train_test_split(
            train_val_candidates, test_size=Config.VAL_RATIO, random_state=42
        )

        print(f"\n📊 Final Dataset Split:")
        print(f"  Train : {len(train_set)} (Uses Pseudo Labels)")
        print(f"  Val   : {len(val_set)} (Uses Pseudo Labels)")
        print(f"  Test  : {len(test_set)} (Uses GT) -> 🔒 ISOLATED")

        return train_set, val_set, test_set

    @staticmethod
    def get_paths_from_samples(sample_list, mode='pseudo', pseudo_ver='original'):
        img_paths, lbl_paths = [], []
        for s in sample_list:
            region, fid = s['region'], s['id']
            img_paths.append(os.path.join(Config.BASE_DIR, f'{region}_image_patches/image_{fid}.tif'))
            if mode == 'gt':
                lbl_paths.append(os.path.join(Config.BASE_DIR, f'{region}_gt_patches/gt_{fid}.tif'))
            else:
                folder = f'{region}_label_patches' if pseudo_ver == 'original' else f'{region}_label_pseudo'
                lbl_paths.append(os.path.join(Config.BASE_DIR, f'{folder}/label_{fid}.tif'))
        return img_paths, lbl_paths
