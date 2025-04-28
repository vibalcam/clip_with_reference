# %%
import pandas as pd

# %%
def enter_result(
        image_net_acc: float,
        mscoco_recall: float,
        notes: str
):
    return dict(
        image_net_acc=image_net_acc,
        mscoco_recall=mscoco_recall,
        notes=notes
    )
# %%
results = [
    enter_result(
        0.0795, 0.0261, "Baseline, no teacher"
    ),
    enter_result(
        0.1108, 0.0347, "0.5 Cross Entropy"
    ),
    enter_result(
        0.1162, 0.0376, "0.5 KL, lock logit scale 0.01"
    ),
    enter_result(
        0.1143, 0.0411, "0.5 Interactive, lock logit scale 0.01"
    ),
    enter_result(
        0.3141, 0.1447, "0.5 Feature"
    ),
    enter_result(
        0.3206, 0.1564, "0.5 Feature, lock logit scale 0.01"
    ),
    enter_result(
        0.3624, 0.183, "0.5 Feature, lock logit scale 0.01, 2xLR"
    ),
    enter_result(
        0.367, 0.1883, "0.5 Feature, lock logit scale 0.01, 3xLR"
    ),
    enter_result(
        0.3906, 0.2138, "0.5 Feature, lock logit scale 0.01, 4xLR, 3xWD"
    ),
    enter_result(
        0.3982, 0.2126, "0.5 Feature, lock logit scale 0.01, 3xLR, 2xWD"
    ),
    enter_result(
        0.3992, 0.2186, "0.5 Feature, lock logit scale 0.01, 3xLR, 3xWD, stop@epoch29"
    ),
    enter_result(
        0.5685, 0.3742, "0.5 Feature, init from teacher model"
    ),
    enter_result(
        0.6333, 0.4028, "Teacher"
    )
]