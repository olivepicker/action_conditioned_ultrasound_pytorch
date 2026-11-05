## Action-Conditioned Ultrasound World Model(wip)

Unofficial PyTorch implementation of [Action-Conditioned World Model for Goal Plane Guidance in Robotic Neck Ultrasound](https://arxiv.org/abs/2607.21918) from Fan et al.

## Todo
- [x] Implement US-PPNR episode metadata loader
- [ ] Implement relative 6DoF probe motion
- [ ] Implement ultrasound frame dataset
- [ ] Integrate pretrained VAE
- [ ] Implement action and temporal embeddings
- [ ] Implement conditional latent diffusion transformer
- [ ] Train world model
- [ ] Sample future ultrasound frames (sanity check)
- [ ] Evaluate action conditioning
- [ ] Implement goal-conditioned action predictor

## Usage
```python
from action_conditioned_ultrasound_pytorch.dataset import USPPNRMetadata

dataset = USPPNRMetadata('../dataset/orig')
episode = dataset[0]

print(episode['probe_pose'].shape)
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
