import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Config.Config_AE import AEConfig
from Dataset.dataset import Load_train_val, VideoDataset
from utils.tools import PrepareSaveFile, EarlyStopping, PrepareModel, \
    PrepareOptimizer, adjust_learning_rate, leftBestModel
from utils.train_eval import train_epoch, val_epoch


def main(cfg):
    train_p, val_p = Load_train_val(cfg.train_txt, cfg.val_txt)
    for K in range(cfg.Kfold):
        ckpt_path, logp = PrepareSaveFile(cfg)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)

        train_dataset = VideoDataset(train_p)
        train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=4)

        valid_dataset = VideoDataset(val_p)
        valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size,
                                       shuffle=False, num_workers=4)
        model = PrepareModel(cfg)
        optimizer = PrepareOptimizer(cfg, model)
        writer = SummaryWriter(logdir=logp)
        for i in range(1, cfg.epoch + 1):
            adjust_learning_rate(cfg, optimizer, i)
            train_epoch(cfg, i, train_data_loader, model, optimizer, logs=True, writer=writer)
            val_loss = val_epoch(cfg, i, valid_data_loader, model, writer)
            save_file_path = os.path.join(ckpt_path, 'save_{}.pth'.format(i))
            earlystopping(val_loss, model, save_file_path, i, optimizer)
            if earlystopping.early_stop:
                writer.close()
                print('earlystopping!')
                break
            leftBestModel(ckpt_path)
        writer.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = AEConfig()
    main(cfg)
