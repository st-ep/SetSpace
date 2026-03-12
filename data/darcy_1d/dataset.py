from __future__ import annotations

import torch
from datasets import load_from_disk


def load_darcy_dataset(data_path):
    dataset = load_from_disk(data_path)
    print(f"Loaded Darcy 1D dataset from: {data_path}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    return dataset


class DarcyDataGenerator:
    """
    Minimal training data wrapper for the Darcy 1D set-encoder case study.
    """

    def __init__(self, dataset, sensor_indices, query_indices, device, batch_size, grid_points):
        self.device = device
        self.batch_size = batch_size
        train_data = dataset["train"]
        n_train = len(train_data)
        n_grid = len(train_data[0]["u"])

        self.u_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        for i in range(n_train):
            self.u_data[i] = torch.tensor(train_data[i]["u"], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(train_data[i]["s"], device=device, dtype=torch.float32)

        self.sensor_indices = sensor_indices.to(device)
        self.query_indices = query_indices.to(device)
        self.grid_points = grid_points.to(device)
        self.n_train = n_train
        self.sensor_x = self.grid_points[self.sensor_indices].view(-1, 1)
        self.query_x = self.grid_points[self.query_indices].view(-1, 1)
        self.u_sensors = self.u_data[:, self.sensor_indices]
        self.s_queries = self.s_data[:, self.query_indices]

    def sample(self):
        indices = torch.randint(0, self.n_train, (self.batch_size,), device=self.device)
        xs = self.sensor_x.unsqueeze(0).expand(self.batch_size, -1, -1)
        us = self.u_sensors[indices].unsqueeze(-1)
        ys = self.query_x.unsqueeze(0).expand(self.batch_size, -1, -1)
        targets = self.s_queries[indices].unsqueeze(-1)
        return xs, us, ys, targets, None


def create_sensor_points(sensor_size, device, grid_points):
    sensor_indices = torch.linspace(0, len(grid_points) - 1, sensor_size, dtype=torch.long)
    sensor_x = grid_points[sensor_indices].to(device).view(-1, 1)
    return sensor_x, sensor_indices


def create_query_points(device, grid_points, n_query_points):
    query_indices = torch.linspace(0, len(grid_points) - 1, n_query_points, dtype=torch.long)
    query_x = grid_points[query_indices].to(device).view(-1, 1)
    return query_x, query_indices
