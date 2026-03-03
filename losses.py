import torch
import torch.nn as nn


class OurLoss(nn.Module):
    def __init__(self, lambda_con=0.3, lambda_super=0.3, lambda_super_con=0.3):
        super(OurLoss, self).__init__()
        self.lambda_con = lambda_con
        self.lambda_super = lambda_super
        self.lambda_super_con = lambda_super_con
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def cross_modal_loss(self, features1, features2, tau=1.0, q=0.01):
        fea = [features1, features2]
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()

        sim_sum1 = sum(
            [sim[:, v * batch_size : (v + 1) * batch_size] for v in range(n_view)]
        )
        diag1 = torch.cat(
            [
                sim_sum1[v * batch_size : (v + 1) * batch_size].diag()
                for v in range(n_view)
            ]
        )
        p1 = diag1 / sim.sum(1)
        loss1 = (1 - q) * (1.0 - (p1) ** q).div(q) + q * (1 - p1)

        sim_sum2 = sum(
            [sim[v * batch_size : (v + 1) * batch_size] for v in range(n_view)]
        )
        diag2 = torch.cat(
            [
                sim_sum2[:, v * batch_size : (v + 1) * batch_size].diag()
                for v in range(n_view)
            ]
        )
        p2 = diag2 / sim.sum(1)
        loss2 = (1 - q) * (1.0 - (p2) ** q).div(q) + q * (1 - p2)
        return loss1.mean() + loss2.mean()

    def supervise_loss(self, logit1, logit2, labels, weight):
        img_loss = self.cls_criterion(logit1, labels)  # [B, num_classes]
        text_loss = self.cls_criterion(logit2, labels)
        sample_loss = img_loss + text_loss  # [B]
        cls_loss = (weight * sample_loss).mean()
        return cls_loss

    def super_cross_modal_loss(
        self, features1, features2, labels, weight, tau=1.0, eps=1e-8
    ):

        B = features1.shape[0]
        device = features1.device
        all_features = torch.cat([features1, features2], dim=0)  # [2B, D]
        all_labels = torch.cat([labels, labels], dim=0)  # [2B, C]
        all_weights = torch.cat([weight, weight], dim=0)  # [2B]

        sim_matrix = torch.matmul(all_features, all_features.T) / tau  # [2B, 2B]
        sim_matrix = sim_matrix - torch.eye(2 * B, device=device) * 1e9

        label_sim = torch.matmul(all_labels.float(), all_labels.float().T)  # [2B, 2B]
        positive_mask = label_sim > 0

        sim_exp = sim_matrix.exp()  # [2B, 2B]
        sim_exp_sum = sim_exp.sum(dim=1, keepdim=True) + eps
        prob = sim_exp / sim_exp_sum

        p_pos = (prob * positive_mask.float()).sum(dim=1)

        loss = -torch.log(p_pos + eps) * all_weights  # weighted log loss
        return loss.mean()

    def forward(
        self,
        features1,
        features2,
        view1_predict_logit,
        view2_predict_logit,
        label,
        weight,
    ):
        loss_cross = (
            self.cross_modal_loss(features1, features2, 1.0, 0.01) * self.lambda_con
        )
        loss_supervise = (
            self.supervise_loss(view1_predict_logit, view2_predict_logit, label, weight)
            * self.lambda_super
        )
        loss_super_con = (
            self.super_cross_modal_loss(features1, features2, label, weight)
            * self.lambda_super_con
        )

        return loss_cross + loss_supervise + loss_super_con
