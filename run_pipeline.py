def run_pipeline():
    train_set, val_set, test_set = DataManager.split_data_once()

    test_loader = None
    if len(test_set) > 0:
        test_imgs, test_gts = DataManager.get_paths_from_samples(test_set, mode='gt')
        test_loader = DataLoader(
            MultiRegionDataset(test_imgs, test_gts, is_gt=True),
            batch_size=Config.BATCH_SIZE, shuffle=False
        )

    best_model_path = None

    for round_idx in range(1, Config.ROUNDS + 1):
        print(f"\n{'='*20} Round {round_idx} / {Config.ROUNDS} {'='*20}")

        ver = 'original' if round_idx == 1 else 'pseudo'
        train_imgs, train_lbls = DataManager.get_paths_from_samples(train_set, mode='pseudo', pseudo_ver=ver)
        val_imgs, val_lbls = DataManager.get_paths_from_samples(val_set, mode='pseudo', pseudo_ver=ver)

        train_loader = DataLoader(MultiRegionDataset(train_imgs, train_lbls), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MultiRegionDataset(val_imgs, val_lbls), batch_size=Config.BATCH_SIZE, shuffle=False)

        model = UNet(in_channels=Config.IN_CHANNELS).to(Config.DEVICE)
        trainer = Trainer(model, train_loader, val_loader)

        save_path = os.path.join(Config.MODEL_SAVE_DIR, f"unet_round{round_idx}.pth")
        trainer.run(save_path)
        best_model_path = save_path

        print("🔧 Updating Pseudo Labels (Train/Val only)...")
        for region in Config.REGIONS:
            region_samples = [s for s in (train_set + val_set) if s['region'] == region]
            if not region_samples: continue

            r_imgs, r_lbls = DataManager.get_paths_from_samples(region_samples, mode='pseudo', pseudo_ver=ver)
            pseudo_dir = os.path.join(Config.BASE_DIR, f'{region}_label_pseudo')

            LabelRefiner.update(model, r_imgs, r_lbls, pseudo_dir)

        del model, trainer, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    if test_loader:
        print("\n🏁 Training Finished. Loading best model for Final Test...")
        final_model = UNet(in_channels=Config.IN_CHANNELS).to(Config.DEVICE)
        final_model.load_state_dict(torch.load(best_model_path))
        FinalTester.evaluate(final_model, test_loader)
    else:
        print("❌ Final Test Skipped (No GT found).")