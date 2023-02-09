from torch.optim import Adam
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, config):
    """
    Parameters
    ----------
    model: Model
    config: ConfigTree

    Returns
    -------
    [transformers.AdamW, torch.optim.Adam]
    """
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param, task_param = model.get_params(named=True)
    grouped_bert_param = [
        {
            "params": [p for n,p in bert_param if not any(nd in n for nd in no_decay)],
            "lr": config["bert_learning_rate"],
            "weight_decay": config["adam_weight_decay"],
        },
        {
            "params": [p for n,p in bert_param if any(nd in n for nd in no_decay)],
            "lr": config["bert_learning_rate"],
            "weight_decay": 0.0,
        }
    ]
    optimizers = [
        AdamW(grouped_bert_param, lr=config["bert_learning_rate"], eps=config["adam_eps"]),
        Adam(model.get_params()[1], lr=config["task_learning_rate"], eps=config["adam_eps"], weight_decay=0)
    ]
    return optimizers


def get_scheduler(optimizers, total_update_steps, warmup_steps):
    """
    Parameters
    ----------
    optimizers: [transformers.AdamW, torch.optim.Adam]
    total_update_steps: int
    warmup_steps: int

    Returns
    -------
    list[torch.optim.lr_scheduler.LambdaLR]
    """
    # Only warm up bert lr
    def lr_lambda_bert(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
               )

    def lr_lambda_task(current_step):
        return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

    schedulers = [
        LambdaLR(optimizers[0], lr_lambda_bert),
        LambdaLR(optimizers[1], lr_lambda_task)
    ]
    return schedulers

