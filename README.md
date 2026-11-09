## Action-Conditioned Ultrasound World Model(wip)

Unofficial PyTorch implementation of [Action-Conditioned World Model for Goal Plane Guidance in Robotic Neck Ultrasound](https://arxiv.org/abs/2607.21918) from Fan et al.

## Todo
- [x] Implement US-VLA episode metadata loader
- [ ] Verify action and goal-label semantics
- [x] Implement relative 6DoF probe motion
- [x] Define world model dataset interface
- [x] Implement ultrasound image loading and preprocessing
- [x] Implement Gaussian VAE
- [ ] Pretrain and freeze VAE
- [x] Implement conditional diffusion transformer
- [ ] Train action-conditioned world model
- [ ] Implement goal-conditioned action predictor
- [ ] Train imitation policy
- [ ] Implement frozen-world-model actor-critic training
- [ ] Evaluate world model and policy

## Usage
```python
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
