import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.text_dataset import MyDataset
from config.engine_config import BasicEngineConfig, PretrainConfig
from utils.engine import set_random_seed
from model.transformer import CausalTransformer
from config.model_config import TransformerConfig


def train_one_epoch(
    model, optimizer, scheduler, train_loader, val_loader, device, epoch
):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, loss: {loss.item():.4f}")
        return total_loss


def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()

    return val_loss


def main(config: BasicEngineConfig):
    # check device, only support one card
    if config.device == "cuda":
        config.device = (
            torch.device(config.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    elif config.device == "mps":
        config.device = (
            torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
        )
    print(f"using {config.device}")

    # fix seed
    set_random_seed(config.seed)

    # create model
    model_cfg = TransformerConfig(
        hidden_dim=768,
        dropout_rate=0.1,
        scale=4,
        act=nn.GELU,
        num_head=12,
        head_dim=768 // 12,
        num_layer=12,
        voc_size=50257,
    )
    model = CausalTransformer(model_cfg)
    model = model.to(config.device)

    # calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # create dataset
    train_dataset = MyDataset(config.data_path)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [0.9, 0.1]
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False)

    for epoch in range(config.epoch):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, val_loader, config.device, epoch
        )
        val_loss = eval(model, val_loader, config.device)
        print(
            f"Epoch: {epoch}, train_loss: {train_loss/len(train_loader):.4f}, val_loss: {val_loss/len(val_loader):.4f}"
        )

        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
        }

        torch.save(checkpoint, "checkpoint.pth")


if __name__ == "__main__":
    config = PretrainConfig()
    main(config)
