from functools import partial
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from torchmetrics.functional import accuracy


from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights


@dataclass
class SSMConfig:
    d_code = 1024
    d_model = 2048
    n_ssm = 1
    n_classes = 400
    lr = 1.4e-4
    lr_min = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.02
    scheduler = "plateau"


class VideoMamba(L.LightningModule):
    """
    Simple SSM model built on Mamba for video classification.
    Videos must be pre-processed by some encoder and embeddings
    should be feed instead of original frames.
    """

    @staticmethod
    def get_default_config():
        return SSMConfig()

    def __init__(self, config: SSMConfig, omit_in_proj: bool = False):
        super().__init__()
        self.save_hyperparameters()
        # in_proj -> Mamba -> out_proj
        self.ssms = nn.ModuleList(
            [create_block(config.d_model, d_intermediate=0,layer_idx=i) for i in range(config.n_ssm)]
        )
        # self.out_proj = nn.Linear(config.d_model, config.n_classes)
        self.norm_fn = nn.LayerNorm(config.d_model)
        # self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(partial(_init_weights, n_layer=config.n_ssm))

    def load_checkpoint(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            "loaded checkpoint {}, missing: {}, unexpected: {}".format(
                ckpt_path, missing, unexpected
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: ssm.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, ssm in enumerate(self.ssms)
        }

    def forward(self, embeds, inference_params=None):
        """
        input: torch.Tensor with shape (batch_size, seqlen, d_code)
        """
        # (batch_size, seqlen, d_code) -> (batch_size, seqlen, d_model)

        hidden_states = embeds
        residual = None

        # (batch_size, seqlen, d_model) -> (batch_size, seqlen, d_model)
        for ssm in self.ssms:
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=inference_params
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))

        # # method 1, global pooling
        # hidden_states = torch.mean(
        #     hidden_states, dim=1
        # )  # (batch_size, seqlen, d_model) -> (batch_size, d_model)
        # method 2, use only the last token
        # hidden_states = hidden_states[
        #     :, -1, :
        # ]  # (batch_size, seqlen, d_model) -> (batch_size, d_model)

        # logits = self.out_proj(
        #     hidden_states
        # )  # (batch_size, d_model) -> (batch_size, n_classes)
        # return self.softmax(logits)
        logits = hidden_states
        return logits

    def on_after_backward(self):
        param_norm_dict = {}
        grad_norm_dict = {}
        for pn, p in self.named_parameters():
            param_norm_dict["train_param/" + pn] = p.norm()
            grad_norm_dict["train_grad/" + pn] = p.grad.norm()
        self.log_dict(
            param_norm_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log_dict(
            grad_norm_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_acc5": acc5},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        # self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict(
            {"val_loss": loss, "val_acc": acc, "val_acc5": acc5}, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_acc5": acc5})

    def configure_optimizers(self):
        cfg = self.hparams.config
        optimizer = optim.AdamW(
            self.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
        )
        scheduler_fn = {
            "plateau": partial(
                optim.lr_scheduler.ReduceLROnPlateau,
                optimizer,
                "min",
                0.1,
                10,
                min_lr=cfg.lr_min,
            ),
            "cosine": partial(
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                optimizer,
                50,
                1,
                cfg.lr_min,
            ),
        }[cfg.scheduler]
        scheduler = scheduler_fn()
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, 10, min_lr=cfg.lr_min)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1, cfg.lr_min)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": "learning_rate",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor("step"),
        ]
        return callbacks
