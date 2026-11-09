
from pathlib import Path

import torch
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
