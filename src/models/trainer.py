import torch.nn as nn
import torch
from tqdm import tqdm
import os
from src.dataset.batchdataloader import BatchDataloader
from src.dataset.utils import unpack_batch
from torch.nn import functional as F
from sklearn import metrics
from src.utils.metrics import *
import pandas as pd
import numpy as np



class Trainer(nn.Module):
    def __init__(self, loss_name, model:nn.Module) -> None:
        super().__init__()
        self.loss_name = loss_name
        self.optimizer = None
        self.model = model

        self.train_stats = ["epoch", "lr", "train_loss"]
        self.valid_metrics = []

    def create_optimizer(self, optimizer_type=torch.optim.Adam, **optimizer_kwargs):
        self.optimizer = optimizer_type(self.model.parameters(), **optimizer_kwargs)

    def create_scheduler(self, scheduler_type=torch.optim.lr_scheduler.ReduceLROnPlateau, **scheduler_kwargs):
        if not self.optimizer:
            raise ValueError("Please init an optimizer before initializing a scheduler")
        self.scheduler = scheduler_type(self.optimizer, **scheduler_kwargs)

    def compute_loss(self, traces:torch.Tensor, target:torch.Tensor, weights:torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        self.model.to(device)
        best_loss = np.Inf
        history = pd.DataFrame(columns=self.train_stats + self.valid_metrics)
        for ep in range(epochs):
            # compute train loss and metrics
            train_loss = self.train_epoch(train_loader, ep, device, weighted=weighted)
            valid_loss, valid_metrics = self.validate(ep, valid_loader, weighted=weighted)

            # Save best model
            if valid_loss < best_loss:
                # Save model
                self.save(folder)
                # Update best validation loss
                best_loss = valid_loss

            # Get learning rate
            learning_rate = self.optimizer.param_groups[-1]["lr"]

            # Interrupt for minimum learning rate
            if learning_rate < min_lr:
                print("Minimum learning rate is reached.")
                break

            # Update learning rate
            self.scheduler.step(valid_loss)

            # Print message
            tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                    '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                    .format(ep, train_loss, valid_loss, learning_rate))

            # Save history
            history = history.append({**{"epoch": ep, "lr": learning_rate, "train_loss": train_loss},  **valid_metrics}, ignore_index=True)
            history.to_csv(os.path.join(folder, 'history.csv'), index=False)


    def train_epoch(self, dataload:BatchDataloader, ep, device, weighted=False, loss_fun=None, meta=False):
        if not loss_fun:
            loss_fun = self.compute_loss
        if not self.optimizer:
            raise ValueError("Seems like the optimizer hasn't been created before training")
        self.model.train()
        total_loss = 0
        n_entries = 0
        weight_des = "weighted " if weighted else ""
        train_desc = "Epoch {:2d}: train - Loss(" + weight_des + self.loss_name + "): {:.6f}"
        train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                        desc=train_desc.format(ep, 0, 0), position=0)

        for batch in dataload:
            data, target, weights = unpack_batch(batch, weighted=weighted)
            # Reinitialize grad
            self.optimizer.zero_grad()
            # Forward pass
            loss = loss_fun(data, target, weights=weights)
            # Backward pass
            loss.backward()
            # Optimize
            self.optimizer.step()
            # Update
            bs = len(data) if not meta else len(data[0])
            # calculate tracking metrics
            with torch.no_grad():
                total_loss += loss.detach().cpu().numpy()
            
            n_entries += bs
            # Update train bar
            train_bar.desc = train_desc.format(ep, total_loss / n_entries)
            train_bar.update(1)
        train_bar.close()
        return total_loss / n_entries

    @torch.no_grad()
    def create_predictions(self, ep:int, dataloader:BatchDataloader, weighted=False):
        self.model.eval()
        prediction_list = []
        target_list = []
        weights_list = [] if weighted else None

        for batch in tqdm(dataloader, desc=f"Epoch {ep}: Validating"):
            data, target, weights = unpack_batch(batch, weighted=weighted)  
            batch_pred = self(data)
            prediction_list.append(batch_pred)
            target_list.append(target)
            if weighted:
                weights_list.append(weights)

        prediction_list = torch.cat(prediction_list, dim=0).cpu()
        target_list = torch.cat(target_list, dim=0).cpu()
        if weighted:
            weights_list = torch.cat(weights_list, dim=0).cpu()
        
        return prediction_list, target_list, weights_list


    def validate(self, ep, dataloader:BatchDataloader, weighted=False):
        raise NotImplementedError

    def save(self, path, name="model"):
        name = name + ".pth"
        torch.save({'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()},
            os.path.join(path, name))

    def load(self, path, strict=True, model_name="model"):
        ckpt = torch.load(os.path.join(path, model_name + '.pth'), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if not self.optimizer:
            print("Optimizer not initialized, therefore skipping loading of state dict")
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])



class RegressionTrainer(Trainer):
    def __init__(self,model: nn.Module) -> None:
        loss_name = "MSE"
        super().__init__(loss_name, model)
        self.valid_metrics = ["mse", "mae"]

    def compute_l1(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        pred = self(traces)
        pred = pred.flatten()
        target = target.flatten()
        loss = self.l1_loss(pred, target)
        return loss

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        # forward pass
        predictions = self(traces)
        if torch.is_tensor(weights):
            loss = mse(target, predictions, weights=weights, reduction=torch.sum)
        else:
            loss = mse(target, predictions, weights=None, reduction=torch.sum)
        return loss

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        if weighted:
            self.valid_metrics += ["weighted_mse", "weighted_mae"]
        return super().train(epochs, train_loader, valid_loader, min_lr, folder, device, weighted)


    def validate(self, ep, dataloader: BatchDataloader, weighted=False):
        predictions, targets, weights = self.create_predictions(ep, dataloader, weighted)
        mse_loss = mse(predictions, targets, weights=None, reduction=torch.mean)
        mae_loss = mae(predictions, targets, weights=None, reduction=torch.mean)
        metrics_dict = {"mse": mse_loss.item(), "mae": mae_loss.item()}
        if weighted:
            metrics_dict["weighted_mse"] = mse(predictions, targets, weights, reduction=torch.mean)
            metrics_dict["weighted_mae"] = mae(predictions, targets, weights, reduction=torch.mean)
        print(metrics_dict)
        return mse_loss, metrics_dict


class ProbRegressionTrainer(RegressionTrainer):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.loss_name = "NLL"
        self.valid_metrics = ["nll"] + self.valid_metrics
        self.burnin = 1

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.mean) -> torch.Tensor:
        # forward pass
        predictions = self(traces)
        pred_mean = predictions[:, 0]
        pred_log_var = predictions[:, 1]
        if torch.is_tensor(weights):
            loss, exp, log_var = gaussian_nll(pred_mean, target, pred_log_var, weights=weights, reduction=reduction)
        else:
            loss, exp, log_var = gaussian_nll(pred_mean, target, pred_log_var, weights=None, reduction=reduction)
        return loss   

    def compute_mse_loss(self, traces, ages, weights):
        prediction = self.model.forward_mean(traces)
        if torch.is_tensor(weights):
            loss = mse(ages, prediction, weights=weights, reduction=torch.sum)
        else:
            loss = mse(ages, prediction, weights=None, reduction=torch.sum)
        return loss

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        if weighted:
            self.valid_metrics += ["weighted_nll"]
        return super().train(epochs, train_loader, valid_loader, min_lr, folder, device, weighted)

    def set_burnin(self, burnin):
        self.burnin = burnin

    def train_epoch(self, dataload: BatchDataloader, ep, device, weighted=False, loss_fun=None, meta=False):
        if self.burnin > 0:
            if ep < self.burnin:
                self.loss_name = "MSE"
                loss_fun = self.compute_mse_loss
            if ep == self.burnin:
                self.loss_name = "NLL"
        return super().train_epoch(dataload, ep, device, weighted, loss_fun, meta)

    
    def validate(self, ep, dataloader: BatchDataloader, weighted=False):
        predictions, targets, weights = self.create_predictions(ep, dataloader, weighted)
        prediction_mean = predictions[:, 0]
        prediction_log_var = predictions[:, 1]
        mse_loss = mse(prediction_mean, targets, weights=None, reduction=torch.mean)
        mae_loss = mae(prediction_mean, targets, weights=None, reduction=torch.mean)
        nll_loss, _, _ = gaussian_nll(prediction_mean, targets, prediction_log_var, weights=None, reduction=torch.mean)
        
        
        metrics_dict = {"mse": mse_loss.item(), "mae": mae_loss.item(), "nll": nll_loss.item()}
        if weighted:
            metrics_dict["weighted_mse"] = mse(predictions, targets, weights, reduction=torch.mean)
            metrics_dict["weighted_mae"] = mae(predictions, targets, weights, reduction=torch.mean)
            metrics_dict["weighted_nll"] = gaussian_nll(prediction_mean, targets, prediction_log_var, weights=weights, reduction=torch.mean)

        print(metrics_dict)
        return nll_loss, metrics_dict

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        # Redefined since we save both best NLL and best MSE
        self.model.to(device)
        best_mse = np.Inf
        best_nll = np.Inf
        history = pd.DataFrame(columns=self.train_stats + self.valid_metrics)
        for ep in range(epochs):
            # compute train loss and metrics
            train_loss = self.train_epoch(train_loader, ep, device, weighted=weighted)
            valid_mse, valid_metrics = self.validate(ep, valid_loader, weighted=weighted)

            # Save best model for different criteria
            if valid_metrics["nll"] < best_nll:
                # Save model
                self.save(folder, "best_valid_nll")
                # Update best validation loss
                best_nll = valid_metrics["nll"]

            if valid_metrics["mse"] < best_mse:
                self.save(folder, "best_valid_mse")
                # Update best validation loss
                best_mse = valid_metrics["mse"]


            # Get learning rate
            learning_rate = self.optimizer.param_groups[-1]["lr"]

            # Interrupt for minimum learning rate
            if learning_rate < min_lr:
                print("Minimum learning rate is reached.")
                break

            # Update learning rate
            self.scheduler.step(valid_metrics["mse"])

            # Print message
            tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                    '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                    .format(ep, train_loss, valid_metrics["mse"], learning_rate))

            # Save history
            history = history.append({**{"epoch": ep, "lr": learning_rate, "train_loss": train_loss},  **valid_metrics}, ignore_index=True)
            history.to_csv(os.path.join(folder, 'history.csv'), index=False)



    



class ClassificationTrainer(Trainer):
    def __init__(self, model) -> None:
        loss_name = "CE"
        super().__init__(loss_name, model)
        self.valid_metrics = ["cross_entropy", "accuracy", "auc"]

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.mean) -> torch.Tensor:
        pred = self(traces)
        target = target.long()
        if torch.is_tensor(weights):
            loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction="none")
        else:
            loss = F.cross_entropy(pred, target, reduction="none")
        return reduction(loss)

    def init_class_weights(self, class_weights: torch.Tensor):
        self.class_weights = class_weights

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        # check that class weights exist for weighted training
        if weighted:
            assert self.class_weights
            self.valid_metrics += ["weighted_cross_entropy"]
        return super().train(epochs, train_loader, valid_loader, min_lr, folder, device, weighted)

    def validate(self, ep, dataloader: BatchDataloader, weighted=False):
        predictions, targets, weights = self.create_predictions(ep, dataloader, weighted)
        targets = targets.long()
        ce = F.cross_entropy(predictions, targets)
        acc = accuracy(predictions, targets, reduction=torch.mean)
        auc = metrics.roc_auc_score(F.one_hot(targets), predictions, multi_class="ovo") # multi class one vs one auc score
        metrics_dict = {"cross_entropy": ce.item(), "accuracy": acc.item(), "auc": auc}
        if weighted:
            wce = F.cross_entropy(predictions, targets, weight=self.class_weights)
            metrics_dict["weighted_cross_entropy"] = wce
        print(metrics_dict)
        return ce, metrics_dict

class BinaryClassificationTrainer(Trainer):
    def __init__(self, model: nn.Module) -> None:
        loss_name = "BCE"
        super().__init__(loss_name, model)
        self.valid_metrics = ["bce", "accuracy", "auc"]

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.mean) -> torch.Tensor:
        pred = self.forward(traces)
        pred = pred.flatten().sigmoid() # convert to probs
        target = target.flatten()
        if torch.is_tensor(weights):
            loss = F.binary_cross_entropy(pred, target, weight=weights.flatten(), reduction="none")
        else:
            loss = F.binary_cross_entropy(pred, target, reduction="none")
        return reduction(loss)

    def train(self, epochs, train_loader, valid_loader, min_lr, folder, device, weighted=False):
        if weighted:
            self.valid_metrics += ["weighted_bce"]
        return super().train(epochs, train_loader, valid_loader, min_lr, folder, device, weighted)
        
    def validate(self, ep, dataloader: BatchDataloader, weighted=False):
        predictions, targets, weights = self.create_predictions(ep, dataloader, weighted)
        predictions, targets = predictions.flatten(), targets.flatten()
        predictions = predictions.sigmoid()
        bce = F.binary_cross_entropy(predictions, targets)
        acc = torch.mean(((predictions > 0.5) == targets).float())
        fpr, tpr, thresholds = metrics.roc_curve(targets, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        metrics_dict = {"bce": bce.item(), "accuracy": acc.item(), "auc": auc}
        if weighted:
            metrics_dict["weighted_bce"] = F.binary_cross_entropy(predictions, targets, weight=weights.flatten())
        print(metrics_dict)
        return bce, metrics_dict
    

class OrdinalRegressionTrainer(Trainer):
    def __init__(self, model) -> None:
        loss_name = "CORAL"
        super().__init__(loss_name, model)
        self.valid_metrics = ["coral", "accuracy"]

    def compute_loss(self, traces: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, reduction=torch.sum) -> torch.Tensor:
        predictions = self.forward(traces)
        predictions = predictions.sigmoid() # convert to probs
        return coral(predictions, target, weights, reduction)

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        prediction_logit = self.forward(x)
        return self.logits_to_prediction(prediction_logit)

    def prediction_class_to_regression(self, pred_class, buckets, mean=True):
        if mean:
            buckets = torch.cat((buckets[0].unsqueeze(dim=0), buckets.unfold(dimension=0, size=2, step=1).mean(dim=-1), buckets[-1].unsqueeze(dim=0)))
        return buckets[pred_class]

    def logits_to_prediction(self, logits:torch.Tensor) -> torch.Tensor:
        return (logits.sigmoid() > 0.5).sum(dim=-1).long()

    def validate(self, ep, dataloader: BatchDataloader, weighted=False):
        prediction_list, target_list, weights = self.create_predictions(ep, dataloader, weighted)
        prediction_class = self.logits_to_prediction(prediction_list)
        accuracy = (prediction_class == target_list.sum(dim=-1)).float().mean()
        prediction_list = prediction_list.sigmoid() # convert to probs
        coral_loss = coral(prediction_list, target_list, None, torch.mean)
        metrics_dict = {"coral": coral_loss.item(), "accuracy": accuracy.item()}
        print(metrics_dict)
        return coral_loss, metrics_dict

