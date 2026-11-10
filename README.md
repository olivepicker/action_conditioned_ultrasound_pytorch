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
- [ ] Complete world model training
- [ ] Evaluate future prediction and action conditioning (sample data, we use US-VLA)

##### Stage 2: Action Prediction
- [ ] Verify goal-label semantics
- [ ] Implement goal-conditioned action predictor
- [ ] Train imitation policy
- [ ] Implement frozen-world-model actor-critic training
- [ ] Evaluate goal-conditioned policy

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
