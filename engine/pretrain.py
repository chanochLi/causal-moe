import torch



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

