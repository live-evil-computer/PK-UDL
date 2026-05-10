class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # 传入验证集Loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.device = Config.DEVICE

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Train Ep {epoch}', leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            preds = self.model(imgs)[:, 1, :, :]

            valid_mask = (lbls != 2).float()
            binary_lbls = (lbls == 1).float()
            weights = torch.where(binary_lbls == 1, ……, ……)

            loss = (self.criterion(preds, binary_lbls) * weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return epoch_loss / len(self.train_loader)

    def validate(self):
        """新增：在每个 Epoch 结束时计算验证集指标"""
        self.model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for imgs, lbls in self.val_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)

                # 预测
                outputs = self.model(imgs)
                probs = torch.sigmoid(outputs[:, 1, :, :])
                preds = (probs > 0.5).long()

                # 同样应用 Mask (忽略2)
                mask = (lbls != 2)
                if mask.sum() > 0:
                    preds_all.append(preds[mask].cpu().numpy())
                    labels_all.append(lbls[mask].cpu().numpy())

        if len(preds_all) == 0:
            return 0, 0, 0

        y_pred = np.concatenate(preds_all)
        y_true = np.concatenate(labels_all)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return precision, recall, f1

    def run(self, save_path):
        print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val P':<8} | {'Val R':<8} | {'Val F1':<8}")
        print("-" * 55)

        for epoch in range(1, Config.EPOCHS + 1):
            # 1. 训练
            loss = self.train_epoch(epoch)

            # 2. 验证 (新增)
            val_p, val_r, val_f1 = self.validate()

            # 3. 打印
            print(f"{epoch:<6} | {loss:.4f}     | {val_p:.4f}   | {val_r:.4f}   | {val_f1:.4f}")

        torch.save(self.model.state_dict(), save_path)
        print(f"    -> Model saved: {os.path.basename(save_path)}")

class FinalTester:
    @staticmethod
    def evaluate(model, test_loader):
        model.eval()
        device = Config.DEVICE
        all_preds, all_gts = [], []

        print("\n🔒 Performing FINAL Evaluation on Ground Truth...")
        with torch.no_grad():
            for imgs, gts in tqdm(test_loader, desc="Testing on GT"):
                imgs, gts = imgs.to(device), gts.to(device)
                probs = torch.sigmoid(model(imgs)[:, 1, :, :])
                preds = (probs > 0.5).long()

                valid_mask = (gts != 255)
                if valid_mask.sum() > 0:
                    all_preds.append(preds[valid_mask].cpu().numpy())
                    all_gts.append(gts[valid_mask].cpu().numpy())

        if len(all_gts) == 0:
            print("❌ Error: No valid GT pixels found.")
            return

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_gts)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        oa = (tp + tn) / (tp + tn + fp + fn)

        print("\n" + "="*40)
        print("🏆 FINAL TEST RESULTS (Ground Truth)")
        print("="*40)
        print(f" Precision : {precision:.4f}")
        print(f" Recall    : {recall:.4f}")
        print(f" F1-Score  : {f1:.4f}")
        print(f" IoU       : {iou:.4f}")
        print(f" OA        : {oa:.4f}")
        print("="*40 + "\n")