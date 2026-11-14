from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .networks.context import ContextEncoder


class WorldModelTrainer:
    def __init__(
        self,
        model,
        vae,
        optimizer,
        train_ds,
        valid_ds,
        num_epochs=100,
        batch_size=4,
        num_workers=4,
        device='cuda',
        save_path='world_model_output'
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.context_encoder = ContextEncoder(vae).to(self.device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.epoch = 0
        self.best_valid_loss = float('inf')

        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        self.valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def encode_batch(self, batch):
        context_images = batch['context_images'].to(self.device)
        target_image = batch['target_image'].to(self.device)

        context_latents = self.context_encoder(context_images)
        future_latent = self.context_encoder.vae.encode(target_image, sample=False)

        return {
            'future_latent': future_latent,
            'context_latents': context_latents,
            'action': batch['action'].to(self.device),
            'time_offset': batch['time_offset'].to(self.device)
        }

    def train_one_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        inputs = self.encode_batch(batch)
        out = self.model(**inputs)
        out['loss'].backward()
        self.optimizer.step()

        return {'loss': out['loss'].detach()}

    @torch.no_grad()
    def valid_one_step(self, batch):
        self.model.eval()
        inputs = self.encode_batch(batch)
        out = self.model(**inputs)

        return {'loss': out['loss'].detach()}

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss
        }, self.save_path / filename)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']

    def train(self):
        for epoch in range(self.epoch, self.num_epochs):
            train_loss = 0.0
            for batch in tqdm(self.train_dl, desc=f'Train {epoch + 1}/{self.num_epochs}'):
                train_loss += self.train_one_step(batch)['loss'].item()
            train_loss /= len(self.train_dl)

            valid_loss = 0.0
            for batch in tqdm(self.valid_dl, desc=f'Valid {epoch + 1}/{self.num_epochs}'):
                valid_loss += self.valid_one_step(batch)['loss'].item()
            valid_loss /= len(self.valid_dl)

            self.epoch = epoch + 1
            is_best = valid_loss < self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss

            self.save_checkpoint('last.pt')
            if is_best:
                self.save_checkpoint('best.pt')

            print(f'Epoch {self.epoch}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}')


class VAETrainer:
    def __init__(
        self,
        vae,
        optimizer,
        train_ds,
        valid_ds,
        num_epochs=100,
        batch_size=4,
        num_workers=4,
        device='cuda',
        save_path='vae_output'
    ):
        self.device = torch.device(device)
        self.vae = vae.to(self.device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.epoch = 0
        self.best_valid_loss = float('inf')

        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        self.valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def train_one_step(self, batch):
        self.vae.train()
        self.optimizer.zero_grad(set_to_none=True)

        images = batch.to(self.device, non_blocking=True)
        out = self.vae(images)
        out['loss'].backward()
        self.optimizer.step()

        return {
            'loss': out['loss'].detach(),
            'recon_loss': out['recon_loss'].detach(),
            'kl_loss': out['kl_loss'].detach()
        }

    @torch.no_grad()
    def valid_one_step(self, batch):
        self.vae.eval()
        images = batch.to(self.device, non_blocking=True)
        out = self.vae(images)

        return {
            'loss': out['loss'].detach(),
            'recon_loss': out['recon_loss'].detach(),
            'kl_loss': out['kl_loss'].detach()
        }

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'vae': self.vae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss
        }, self.save_path / filename)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.vae.load_state_dict(checkpoint['vae'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']

    def run_epoch(self, dataloader, training):
        totals = {'loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0}
        step = self.train_one_step if training else self.valid_one_step
        phase = 'Train' if training else 'Valid'
        count = 0

        for batch in tqdm(dataloader, desc=f'{phase} {self.epoch + 1}/{self.num_epochs}'):
            out = step(batch)
            batch_size = batch.shape[0]
            count += batch_size

            for key in totals:
                totals[key] += out[key].item() * batch_size

        return {key: value / count for key, value in totals.items()}

    def train(self):
        for epoch in range(self.epoch, self.num_epochs):
            train_metrics = self.run_epoch(self.train_dl, training=True)
            valid_metrics = self.run_epoch(self.valid_dl, training=False)

            self.epoch = epoch + 1
            is_best = valid_metrics['loss'] < self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_metrics['loss']

            self.save_checkpoint('last.pt')
            if is_best:
                self.save_checkpoint('best.pt')

            print(
                f'Epoch {self.epoch}: '
                f'train_loss={train_metrics["loss"]:.6f}, '
                f'valid_loss={valid_metrics["loss"]:.6f}, '
                f'valid_recon={valid_metrics["recon_loss"]:.6f}, '
                f'valid_kl={valid_metrics["kl_loss"]:.6f}'
            )


class PolicyTrainer:
    def __init__(
        self,
        policy,
        vae,
        optimizer,
        train_ds,
        valid_ds,
        num_epochs=100,
        batch_size=16,
        num_workers=4,
        device='cuda',
        save_path='policy_output'
    ):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.vae = vae.to(self.device).eval().requires_grad_(False)
        self.context_encoder = ContextEncoder(self.vae).to(self.device).eval()
        self.context_encoder.requires_grad_(False)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.epoch = 0
        self.best_valid_loss = float('inf')

        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        self.valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def encode_batch(self, batch):
        context_images = batch['context_images'].to(self.device)
        goal_image = batch['goal_image'].to(self.device)

        return {
            'context_latents': self.context_encoder(context_images),
            'goal_latent': self.vae.encode(goal_image, sample=False),
            'previous_actions': batch['previous_actions'].to(self.device),
            'time_offset': batch['time_offset'].to(self.device)
        }

    @staticmethod
    def euler_to_quaternion(euler):
        roll, pitch, yaw = (euler / 2).unbind(-1)

        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        return torch.stack([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ], dim=-1)

    @torch.no_grad()
    def action_metrics(self, prediction, target):
        translation_mm = (
            (prediction[:, :3] - target[:, :3])
            .norm(dim=-1)
            .mean()
            * 1000
        )

        predicted_q = self.euler_to_quaternion(prediction[:, 3:])
        target_q = self.euler_to_quaternion(target[:, 3:])

        cosine = (
            (predicted_q * target_q)
            .sum(dim=-1)
            .abs()
            .clamp(0, 1)
        )

        rotation_deg = torch.rad2deg(
            2 * torch.atan2(
                torch.sqrt((1 - cosine.square()).clamp_min(0)),
                cosine
            )
        ).mean()

        return {
            'translation_mm': translation_mm,
            'rotation_deg': rotation_deg
        }

    def train_one_step(self, batch):
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)

        inputs = self.encode_batch(batch)
        target = batch['expert_action'].to(self.device)
        normalized_target = target / self.policy.action_scales

        out = self.policy(**inputs)
        loss = F.smooth_l1_loss(
            out['normalized_action'],
            normalized_target
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        metrics = self.action_metrics(out['action'].detach(), target)
        metrics['loss'] = loss.detach()

        return metrics

    @torch.no_grad()
    def valid_one_step(self, batch):
        self.policy.eval()

        inputs = self.encode_batch(batch)
        target = batch['expert_action'].to(self.device)
        normalized_target = target / self.policy.action_scales

        out = self.policy(**inputs)
        no_goal_out = self.policy(**inputs, use_goal=False)

        loss = F.smooth_l1_loss(
            out['normalized_action'],
            normalized_target
        )

        no_goal_loss = F.smooth_l1_loss(
            no_goal_out['normalized_action'],
            normalized_target
        )

        metrics = self.action_metrics(out['action'], target)
        metrics['loss'] = loss
        metrics['no_goal_loss'] = no_goal_loss

        return metrics

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.epoch,
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss
        }, self.save_path / filename)

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.best_valid_loss = checkpoint['best_valid_loss']

    def run_epoch(self, dataloader, training):
        totals = {}
        count = 0

        step = self.train_one_step if training else self.valid_one_step
        phase = 'Train' if training else 'Valid'

        for batch in tqdm(
            dataloader,
            desc=f'{phase} {self.epoch + 1}/{self.num_epochs}'
        ):
            metrics = step(batch)
            batch_size = batch['expert_action'].shape[0]
            count += batch_size

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value.item() * batch_size

        return {
            key: value / count
            for key, value in totals.items()
        }

    def train(self):
        for epoch in range(self.epoch, self.num_epochs):
            train_metrics = self.run_epoch(self.train_dl, training=True)
            valid_metrics = self.run_epoch(self.valid_dl, training=False)

            self.epoch = epoch + 1

            is_best = valid_metrics['loss'] < self.best_valid_loss

            if is_best:
                self.best_valid_loss = valid_metrics['loss']

            self.save_checkpoint('last.pt')

            if is_best:
                self.save_checkpoint('best.pt')

            goal_advantage = (
                valid_metrics['no_goal_loss']
                - valid_metrics['loss']
            )

            print(
                f'Epoch {self.epoch}: '
                f'train_loss={train_metrics["loss"]:.6f}, '
                f'valid_loss={valid_metrics["loss"]:.6f}, '
                f'translation_mm={valid_metrics["translation_mm"]:.3f}, '
                f'rotation_deg={valid_metrics["rotation_deg"]:.3f}, '
                f'goal_advantage={goal_advantage:.6f}'
            )