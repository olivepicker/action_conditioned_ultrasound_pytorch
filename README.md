## Action-Conditioned Ultrasound World Model(wip)

Unofficial PyTorch implementation of [Action-Conditioned World Model for Goal Plane Guidance in Robotic Neck Ultrasound](https://arxiv.org/abs/2607.21918) from Fan et al.

## Todo

##### Stage 0: Data & VAE
- [x] Implement dataset and preprocessing
- [x] Verify 6DoF pose and action semantics
- [x] Implement and pretrain Gaussian VAE

##### Stage 1: World Model
- [x] Implement conditional diffusion transformer
- [x] Implement diffusion schedule and DDIM sampler
- [x] Implement world model training pipeline
- [x] Complete world model training
- [x] Evaluate future prediction and action conditioning (sample data, we use SonATA)
- [ ] LPIPS, SSIM and Forward-Inverse Consistency

##### Stage 2: Action Prediction
- [ ] Verify goal-label semantics
- [ ] Implement goal-conditioned action predictor
- [ ] Train imitation policy
- [ ] Implement frozen-world-model actor-critic training
- [ ] Evaluate goal-conditioned policy

## Usage


```python
# Stage 1

# 1. Train VAE (Prepare your dataset, I used sonATA dataset for implementation)
from networks.vae import VAE
from trainer import VAETrainer

vae = VAE(
    in_channels=3,
    hidden_channels=32,
    latent_dim=4,
    kl_weight=1e-4,
    recon_loss_type='l1',
    downsample_factor=8
)

trainer = VAETrainer(
    vae=vae,
    optimizer=torch.optim.AdamW(vae.parameters(), lr=1e-4),
    train_ds=train_vae_ds,
    valid_ds=valid_vae_ds,
    num_epochs=20,
    batch_size=16,
    device='cuda',
    save_path='vae_output'
)

trainer.train()


# 2. Train Conditional Diffusion World Model
from networks.dit import ConditionalDiT
from diffusion import DiffusionSchedule
from model import WorldModel
from trainer import WorldModelTrainer

vae.load_state_dict(torch.load('vae_output/best.pt')['model'])
vae.eval().requires_grad_(False)

dit = ConditionalDiT(
    latent_channels=4,
    dim=256,
    depth=6,
    context_length=3,
    translation_scale=translation_scale,
    rotation_scale=rotation_scale,
    patch_size=2,
    dim_head=32,
    num_heads=8
)

world_model = WorldModel(
    dit=dit,
    diffusion=DiffusionSchedule(num_timesteps=1000)
)

trainer = WorldModelTrainer(
    model=world_model,
    vae=vae,
    optimizer=torch.optim.AdamW(world_model.parameters(), lr=1e-4),
    train_ds=train_ds,
    valid_ds=valid_ds,
    num_epochs=10,
    batch_size=16,
    device='cuda',
    save_path='world_model_output'
)

trainer.train()


# 3. Generate Test
from infer import WorldModelInfer

world_model.load_state_dict(
    torch.load('world_model_output/best.pt')['model']
)

infer = WorldModelInfer(
    vae=vae,
    world_model=world_model,
    num_steps=50
).to('cuda').eval()

sample = valid_ds[0]

with torch.inference_mode():
    output = infer(
        context_images=sample['context_images'][None].to('cuda'),
        action=sample['action'][None].to('cuda'),
        time_offset=sample['time_offset'][None].to('cuda')
    )

future_image = output['future_image']

# Predictions are compared with ground-truth frames at horizons of 0.5, 1, 2, 3, and 5 seconds using MAE with zero, shuffled, and opposite actions.
# This is simple version of evaluation, LPIPS, SSIM and Forward-Inverse Consistency will be added later.

```

```python
# Stage 2(wip)

```

## Citations

```bibtex
@misc{fan2026actionconditionedworldmodel,
      title={Action-Conditioned World Model for Goal Plane Guidance in Robotic Neck Ultrasound},
      author={Siqi Fan and Ran Liu and Zixuan Yang and Xiaoyu Fu and Xiaoqing Gao and Yingqi Xing and Yunhui Liu and Hongbin Liu and Mingcong Chen},
      year={2026},
      eprint={2607.21918},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.21918},
}
```
