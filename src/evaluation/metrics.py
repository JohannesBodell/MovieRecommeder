import torch

def mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets)).item()

def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def evaluate_model(model, data_loader):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for users, movies, ratings in data_loader:
            predictions = model(users, movies)
            all_predictions.append(predictions)
            all_targets.append(ratings)

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    return {
        "mae": mae(predictions, targets),
        "rmse": rmse(predictions, targets)
    }