# %%
import sys
sys.path.append("../src")
from fast_clip import create_model_and_transforms
# %%

model = "ViT-B-16" 
distill_model = "ViT-B-32"
model, _, _ = create_model_and_transforms(
    model, 
)

dist_model, _, _ = create_model_and_transforms(
    distill_model, 
)
# %%
model_nparams = sum(p.numel() for p in model.parameters())
dist_model_nparams = sum(p.numel() for p in dist_model.parameters())
model_nparams, dist_model_nparams
# %%
model_named_params = {k: v for k, v in model.named_parameters()}
dist_model_named_params = {k: v for k, v in dist_model.named_parameters()}
# %%
assert list(model_named_params.keys()) == list(dist_model_named_params.keys())
# %%
assert sum(v.numel() for v in model_named_params.values()) == model_nparams
assert sum(v.numel() for v in dist_model_named_params.values()) == dist_model_nparams
# %%
unmatched_params = []
for (k, v), (dist_k, dist_v) in zip(model_named_params.items(), dist_model_named_params.items()):
    if v.shape != dist_v.shape:
        unmatched_params.append(k)
# %%
len(unmatched_params)
# %%
unmatched_params
# %%
model_named_params["visual.positional_embedding"].shape
# %%
dist_model_named_params["visual.positional_embedding"].shape
# %%
model_named_params["visual.conv1.weight"].shape
# %%
dist_model_named_params["visual.conv1.weight"].shape