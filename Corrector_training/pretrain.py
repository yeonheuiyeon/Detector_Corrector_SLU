"""
Reference: https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-
Next ref: https://github.com/ceshine/finetuning-t5/tree/master/paraphrase
"""
import enum
import os
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
import pytorch_lightning_spells as pls
import typer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# from Cord19Dataset import Cord19Dataset, DATASET_DIR, Parts
from T5Dataset import T5Dataset, DATASET_DIR, Parts
from t2t import BaseConfig, T5BaseModel, masked_cross_entropy_loss


class Corpus(enum.Enum):
    T5 = 't5'

@dataclass
class Config(BaseConfig):
    dataset: Corpus = Corpus.T5


class T5Model(T5BaseModel):
    def __init__(self, config: Config, **kwargs):
        model = T5ForConditionalGeneration.from_pretrained(config.base_t5_model)
        tokenizer = T5Tokenizer.from_pretrained(config.base_t5_model, model_max_length=512)
        super().__init__(config, model, tokenizer)
        self.config = config
        # log the config values
        self.save_hyperparameters(asdict(config))
        self.train_dataset = T5Dataset(Parts.TRAIN)
        print("Train dataset: ", len(self.train_dataset))
        self.valid_dataset = T5Dataset(Parts.VALID)
        print("Valid dataset: ", len(self.valid_dataset))


def main(
        t5_model: str = "dataset_cache_with_hyp_word_prediction_newmodel_onlygoogle_final/t5-base_last_epoch30", lr: float = 1e-4,  # 3^4
        epochs: int = 20, fp16: bool = False,
        dataset: Corpus = Corpus.T5, batch_size: int = 160,
        max_len: int = 512, grad_accu: int = 1,
        num_gpus: int = 8
):
    pl.seed_everything(int(os.environ.get("SEED", 738)))
    config = Config(
        base_t5_model=t5_model,
        learning_rate=lr,
        epochs=epochs,
        dataset=dataset,
        max_len=max_len,
        grad_accu=grad_accu,
        batch_size=batch_size,
        fp16=fp16,
        weight_decay=0,
        num_gpus=num_gpus,
        loss_fn=masked_cross_entropy_loss
    )

    pl_module = T5Model(config)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(DATASET_DIR / "model_checkpoints"),
            monitor='val_loss',
            mode="min",
            filename='{epoch:02d}-{step:06d}-{val_loss:.4f}',
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    trainer = pl.Trainer(
        accelerator='cuda' if num_gpus > 1 else None,
        # amp_backend="apex", amp_level='O2',
        precision=16 if config.fp16 else 32,
        gpus=config.num_gpus,
        val_check_interval=1.0,
        gradient_clip_val=10,
        max_epochs=epochs,
        # max_steps=steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_accu,
        # auto_scale_batch_size='power' if batch_size is None else None,
        logger=[
            # pls.loggers.ScreenLogger()
            # pl.loggers.TensorBoardLogger(str(DATASET_DIR / "tb_logs"), name=""),
            pl.loggers.WandbLogger(project="t5-pretrain-final")
        ],
        log_every_n_steps=100,
        replace_sampler_ddp=False
    )

    trainer.fit(pl_module)

    pl_module.model.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_last_epoch50")
    pl_module.tokenizer.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_last_epoch50")
    print("Last model saved")

    # assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
    # print(callbacks[0].best_model_path)
    # pl_module = T5Model.load_from_checkpoint(
    #     callbacks[0].best_model_path,
    #     config=config
    # )
    # pl_module.model.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_best")
    # pl_module.tokenizer.save_pretrained(DATASET_DIR / f"{config.base_t5_model}_best")
    # print("Best model saved")


if __name__ == "__main__":
    typer.run(main)
