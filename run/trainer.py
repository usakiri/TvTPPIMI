import torch
import torch.nn as nn
import copy
import os
import numpy as np
from utils import set_seed
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_score
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, seed, model, optim, device, train_dataloader, val_dataloader, test_dataloader, output, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.seed = seed

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = output

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Precision", "Threshold", "Test_loss"]
        
        train_metric_header = ["# Epoch", "Train_loss"]
        
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        set_seed(self.seed)
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1

            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
        
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc and auprc >= self.best_auprc:
                self.best_model = copy.deepcopy(self.model)
                self.best_model.eval()
                self.best_auroc = auroc
                self.best_auprc = auprc
                self.best_epoch = self.current_epoch
                torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"best_model_epoch.pth"))
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision, y_pred, y_label = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, precision, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Precision " + str(precision) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
    
        return self.test_metrics, y_pred, y_label

    def save_result(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        torch.save(self.best_model.state_dict(),
                   os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
       
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())
    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score= self.model(v_d, v_p, v_d_mask, v_p_mask)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            if dataloader == "test" and self.best_model is not None:
                self.best_model.eval()
            for i, (v_d, v_p, labels, v_d_mask, v_p_mask) in enumerate(tqdm(data_loader)):
                v_d, v_p, labels, v_d_mask, v_p_mask = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device), v_d_mask.to(self.device), v_p_mask.to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p, v_d_mask, v_p_mask)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p, v_d_mask, v_p_mask)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        y_label = np.asarray(y_label, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            positives = np.count_nonzero(y_label == 1)
            negatives = np.count_nonzero(y_label == 0)
            denom = (tpr * positives) + (fpr * negatives)
            precision = np.divide(
                tpr * positives,
                denom,
                out=np.zeros_like(tpr),
                where=denom > 0,
            )
            recall = tpr
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) > 0,
            )

            start_idx = 5 if thresholds.size > 5 else 0
            f1_segment = f1[start_idx:]
            thresholds_segment = thresholds[start_idx:]
            if f1_segment.size == 0:
                best_idx = int(np.argmax(f1))
                best_f1 = float(f1[best_idx])
                thred_optim = float(thresholds[best_idx])
            else:
                rel_idx = int(np.argmax(f1_segment))
                best_f1 = float(f1_segment[rel_idx])
                thred_optim = float(thresholds_segment[rel_idx])

            if not np.isfinite(thred_optim):
                thred_optim = 0.5

            y_pred_s = (y_pred >= thred_optim).astype(int)
            y_true = y_label.astype(int)
            cm1 = confusion_matrix(y_true, y_pred_s, labels=[0, 1])
            tn, fp, fn, tp = cm1.ravel()
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            precision1 = precision_score(y_true, y_pred_s, zero_division=0)
            return auroc, auprc, best_f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision1, y_pred, y_label
        else:
            return auroc, auprc, test_loss
