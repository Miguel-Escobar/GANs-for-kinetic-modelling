import os, sys
import math
import time
import pickle
import matplotlib
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from random import sample
from functools import partial


import helper as hp


# check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU")
    device = torch.device("cpu")


class SinusoidalPositionEmbedding(nn.Module):
    """Implements sinusoidal positional embeddings"""

    def __init__(self, max_len: int, model_dim: int) -> None:
        super(SinusoidalPositionEmbedding, self).__init__()
        pos = torch.arange(0.0, max_len).unsqueeze(1).repeat(1, model_dim)
        dim = torch.arange(0.0, model_dim).unsqueeze(0).repeat(max_len, 1)
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / model_dim))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Updates the input embedding with positional embedding
        Arguments:
            x {torch.Tensor} -- Input tensor
        Returns:
            torch.Tensor -- Input updated with positional embeddings
        """
        t = self.pe[:, : x.size(1), :]
        return x + t


class TransformerLayer(nn.Module):
    """
    Basic transformer layer with layer normalization.

    Args:
        embedding_dim (int): Dimension of the inner transformer space.
        num_heads (int): Number of self-attention heads.
        MLP_dim (int): Dimension of the MLP hidden layers in the transformer.
        dropout (float): Dropout rate for the attention and MLP layers. Default is 0.0.
        attention_dropout (float): Dropout rate for the attention mechanism. Default is 0.0.
        activation_dropout (float): Dropout rate for the activation function in the MLP. Default is 0.0.
        activation_fn (callable): Activation function to use in the MLP. Default is F.relu.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        MLP_dim,
        activation_fn=F.relu,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn

        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=self.attention_dropout,
        )

        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)

        # MLP for after the attention mechanism:
        self.first_mlp = nn.Linear(self.embedding_dim, MLP_dim)
        self.second_mlp = nn.Linear(MLP_dim, self.embedding_dim)
        self.mlp_layer_norm = torch.nn.LayerNorm(self.embedding_dim)

        self.init_parameters()  # Explicit parameter initialization.

    def init_parameters(self):
        nn.init.xavier_uniform_(self.first_mlp.weight)
        nn.init.constant_(self.first_mlp.bias, 0.0)
        nn.init.xavier_uniform_(self.second_mlp.weight)
        nn.init.constant_(self.second_mlp.bias, 0.0)

    def forward(self, x):
        skip_layer = x

        x = self.self_attention_layer_norm(x)
        x, _ = self.self_attention(query=x, key=x, value=x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = skip_layer + x  # Residual connection
        skip_layer = x

        x = self.mlp_layer_norm(x)
        x = self.activation_fn(self.first_mlp(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.second_mlp(x)  # Sin activación final
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = skip_layer + x  # Residual connection

        return x


class Transformer(nn.Module):
    """
    Transformer module implementation with sinusoidal position embeddings but no masking. Dropout is not implemented yet.

    Args:
        input_dim (int): Size of the input vectors.
        embedding_dim (int): Dimension of the embedding space.
        max_input_length (int): Maximum length of input sequences.
        num_heads (int): Number of self-attention heads.
        num_layers (int): Number of transformer layers.
        MLP_dim (int): Dimension of the MLP hidden layers in the transformer.
        positional_embedding (bool): Whether to use positional embeddings. Default is True.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        max_input_length,
        num_heads,
        num_layers,
        MLP_dim,
        positional_embedding=True,
    ):
        super().__init__()

        # First we provide an embedding layer to the internal transformer dimension:
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_input_length
        self.embedding_scale = math.sqrt(self.embedding_dim)
        self.embed_positions = (
            SinusoidalPositionEmbedding(
                max_len=self.max_sequence_length + 1, model_dim=self.embedding_dim
            )
            if positional_embedding
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=num_heads,
                    MLP_dim=MLP_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.init_parameters()  # Explicit parameter initialization.

    def init_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding_dim**-0.5)

    def forward(self, input_tokens):
        # Apply embedding to input:
        x = self.embedding_scale * self.embedding(input_tokens)

        if self.embed_positions is not None:
            x = self.embed_positions(x)

        x = x.transpose(
            0, 1
        )  # Transpose to (sequence_length, batch_size, embedding_dim)

        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)

        x = x.transpose(
            0, 1
        )  # Transpose back to (batch_size, sequence_length, embedding_dim)
        return x


class Generator(nn.Module):
    """
    Takes in a noise sequence and a label as inputs, outputs a set of kinetic model parameters.

    Args:
        embedding_dim (int): Dimension of the vectors that we feed to the transformer in sequences.
        n_parameters (int): Number of parameters needed for the kinetic model.
        noise_len (int): Length of each noise input sequence that we feed as a seed for parameter generation.
        noise_dim (int): Dimension of each element of the noise input sequence.
        n_transformer_layers (int): Number of transformer layers in the generator.
        MLP_dim (int): Dimension of the MLP layers in the transformer.
        num_heads (int): Number of self-attention heads in the transformer. Default is 1.
        num_classes (int): Number of classes for the labels. Default is 2, for binary classification (Biologically Relevant and Biologically Irrelevant).
    """

    def __init__(
        self,
        embedding_dim,
        n_parameters,
        noise_len,
        noise_dim,
        n_transformer_layers,
        MLP_dim,
        num_heads=1,
    ):
        super(Generator, self).__init__()

        self.transformer = Transformer(
            input_dim=noise_dim,
            embedding_dim=embedding_dim,
            max_input_length=noise_len + 1,  # +1 for the label info.
            num_heads=num_heads,
            num_layers=n_transformer_layers,
            MLP_dim=MLP_dim,
        )

        self.output_linear = nn.Linear((noise_len + 1) * embedding_dim, n_parameters)

    def forward(self, noise_sequence_and_label):
        """
        Forward pass of the generator.

        Args:
            noise_sequence_and_label (torch.Tensor): Input noise sequence of shape (batch_size, noise_len + 1, noise_dim). The extra dimension contains the label information.

        Returns:
            torch.Tensor: Generated kinetic model parameters of shape (batch_size, n_parameters).
        """

        # print(
        #     "Calling generator forward pass on tensor of shape: ",
        #     noise_sequence_and_label.shape,
        # )

        # Pass through the transformer
        transformer_output_sequence = self.transformer(noise_sequence_and_label)

        # Reshape the output to match the expected input of the linear layer
        transformer_output = transformer_output_sequence.reshape(
            transformer_output_sequence.size(0), -1
        )

        # Pass through the output linear layer to generate parameters
        generated_parameters = self.output_linear(transformer_output)

        return generated_parameters


class Discriminator(nn.Module):
    """
    Module that takes in a set of kinetic model parameters and a label as inputs, and outputs logits indicating the probability of the parameters being real (instead of generated). Used for training the generator module.
    """

    def __init__(
        self,
        embedding_dim,
        n_parameters,
        n_transformer_layers,
        MLP_dim,
        num_heads=1,
    ):
        super(Discriminator, self).__init__()

        self.transformer = Transformer(
            input_dim=1,
            embedding_dim=embedding_dim,
            max_input_length=n_parameters + 1,  # +1 for the label info.
            num_heads=num_heads,
            num_layers=n_transformer_layers,
            MLP_dim=MLP_dim,
        )

        self.output_linear = nn.Linear((n_parameters + 1) * embedding_dim, 1)

    def forward(self, parameters_and_label):
        """
        Forward pass of the discriminator.

        Args:
            parameters_and_label (torch.Tensor): Input parameters and label of shape (batch_size, n_parameters + 1, 1). The extra dimension contains the label information.

        Returns:
            torch.Tensor: Logits indicating the probability of the parameters being real.
        """
        # print(
        #     "Calling discriminator forward pass on tensor of shape: ",
        #     parameters_and_label.shape,
        # )
        # Ensure the input has three dimensions:
        if parameters_and_label.dim() == 2:
            parameters_and_label = parameters_and_label.unsqueeze(2)

        # Pass through the transformer
        transformer_output_sequence = self.transformer(parameters_and_label)

        # Reshape the output to match the expected input of the linear layer
        transformer_output = transformer_output_sequence.reshape(
            transformer_output_sequence.size(0), -1
        )

        # Pass through the output linear layer to generate logits
        logits = self.output_linear(transformer_output)

        return logits


class CGAN:
    def __init__(
        self,
        X_train,
        y_train,
        latent_dim,
        batch_size,
        path_generator,
        savepath,
        num_classes=2,
        verbose=False,
    ):
        """
        Conditional Generative Adversarial Network (CGAN) for sampling
        kinetic model parameters.

        X_train: torch.tensor
            Kinetic model parameters obtained via traditional methods.

        y_train: torch.tensor
            Labels for the parameters in X_train, classified as biologically relevant or irrelevant.
        """

        self.param_shape = X_train.shape[1]
        self.label_shape = 1
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batchsize = batch_size
        self.savepath = savepath
        self.verbose = verbose
        self.min_x = None
        self.max_x = None
        self.informed_labelling = False

        self.noise_dim = 3  # TODO: Make an informed decision.

        if path_generator == None:
            self.transfer_learning = False
        else:
            self.transfer_learning = True
            self.path_generator = path_generator

        # data
        self.X_train = X_train
        self.y_train = y_train

        self.discriminator = Discriminator(
            embedding_dim=4,  # TODO : Make an informed decision.
            n_parameters=self.param_shape,
            n_transformer_layers=2,  # TODO : Make an informed decision.
            MLP_dim=64,  # TODO : Make an informed decision.
            num_heads=4,  # TODO : Make an informed decision.
        )

        # d_trainable_count = np.sum([K.count_params(w) for w in self.discriminator.trainable_weights])

        d_trainable_count = sum(
            p.numel() for p in self.discriminator.parameters() if p.requires_grad
        )

        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Build and compile the generator
        # self.generator = self.build_generator()
        # self.generator.compile(loss=['binary_crossentropy'],
        #                        optimizer=optimizer, metrics='accuracy')
        self.generator = Generator(
            embedding_dim=30,  # TODO : Make an informed decision.
            n_parameters=self.param_shape,
            noise_len=self.latent_dim,
            noise_dim=self.noise_dim,  # TODO : Make an informed decision.
            n_transformer_layers=2,  # TODO : Make an informed decision.
            MLP_dim=64,  # TODO : Make an informed decision.
            num_heads=3,  # TODO : Make an informed decision.
        )

        g_trainable_count = sum(
            p.numel() for p in self.generator.parameters() if p.requires_grad
        )

        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        print(f"Total trainable parameters: {g_trainable_count + d_trainable_count}")

    def train(
        self, epochs: int, sample_interval: int, n_samples: int
    ) -> tuple[list, list, list]:
        """
        Trains the GAN.

        Args:
        - epochs (int): The number to epochs
        - sample_interval (int): Every 'sample_interval' epochs, we sample parameters and store them.
        - n_samples (int): The number of samples to generate at each sample interval.

        Returns:
        - all_d_loss (list): List of discriminator losses for each epoch.
        - all_g_loss (list): List of generator losses for each epoch.
        - all_acc (list): List of accuracies for each epoch.
        """

        # We move the models to the device an set training mode
        self.generator.to(device)
        self.discriminator.to(device)
        self.discriminator.train()
        self.generator.train()

        # Rescale the input between -1 to 1
        # This is needed because we use the Tanh as output
        # function, therefore we need to match the domain
        # of definitino of that function
        X_train, min_x, max_x = hp.scale_range(self.X_train, -1.0, 1.0)
        self.min_x = min_x
        self.max_x = max_x

        # save for future sampling
        d_scaling = {"min_x": min_x, "max_x": max_x}
        hp.save_pkl(f"{self.savepath}d_scaling.pkl", d_scaling)

        batchsize = self.batchsize
        half_batch = int(batchsize / 2)

        all_d_loss = []
        all_g_loss = []
        all_acc = []

        # Compute how many samples will
        # go in a batch
        samples_per_epoch = X_train.shape[0]
        number_of_batches = int(samples_per_epoch / batchsize)

        criterion = criterion = torch.nn.BCELoss()

        # In each epoch we train once on each batch
        for epoch in range(epochs):

            epoch_g_loss = []
            epoch_d_loss = []
            epoch_acc = []

            """
            On each iteration, we train on a batch.
            That is:
            - Train the discriminator,
            - Train
            """
            for i in range(number_of_batches):
                X_batch = np.array(X_train[batchsize * i : batchsize * (i + 1)])
                y_batch = np.array(self.y_train[batchsize * i : batchsize * (i + 1)])

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.discriminator.zero_grad()
                self.generator.zero_grad()

                # Select a random half batch
                idx = np.random.randint(0, X_batch.shape[0], half_batch)
                params_np, labels_np = X_batch[idx], y_batch[idx]

                params = torch.tensor(params_np, dtype=torch.float32, device=device)
                labels = torch.tensor(labels_np, dtype=torch.float32, device=device)
                noise = torch.randn(
                    half_batch, self.latent_dim, self.noise_dim, device=device
                )

                labels_formatted = labels.unsqueeze(1).repeat(1, self.noise_dim)

                labels_formatted = labels_formatted.unsqueeze(1)

                # print("Good labels shape:", labels_formatted.shape)
                # print("Good noise shape: ", noise.shape)

                data = torch.cat([noise, labels_formatted], dim=1)

                # Generate a half batch
                gen_params = self.generator(data)

                # DISCRIMINATE:
                valid = torch.ones((half_batch, 1), device=device, dtype=torch.float32)
                fake = torch.zeros((half_batch, 1), device=device, dtype=torch.float32)

                labels = labels.unsqueeze(1)

                # Reales
                real_inputs = torch.cat((params, labels), dim=1)
                pred_real = self.discriminator(real_inputs)
                d_loss_real = criterion(pred_real, valid)

                # Fakes
                fake_inputs = torch.cat((gen_params, labels), dim=1)
                pred_fake = self.discriminator(fake_inputs)
                d_loss_fake = criterion(pred_fake, fake)

                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                self.optimizer_discriminator.step()
                d_loss_value = d_loss.item()

                # ---------------------
                #  Train Generator
                # ---------------------
                self.generator.zero_grad()
                self.discriminator.zero_grad()

                noise = torch.randn(
                    batchsize, self.latent_dim, self.noise_dim, device=device
                )
                valid = torch.ones((batchsize, 1), device=device, dtype=torch.float32)

                # Generator wants discriminator to label the generated images as the intended
                # stability

                data_labels = [1, -1]
                sampled_labels = np.random.choice(data_labels, batchsize)

                # Add noise
                sampled_labels = np.array(
                    [j + np.random.normal(0, 0) for j in sampled_labels]
                )

                # sampled_labels = sampled_labels.reshape((-1, 1))

                # PyTorchify
                sampled_labels = torch.tensor(
                    sampled_labels, dtype=torch.float32, device=device
                )

                sampled_labels_formatted = sampled_labels.unsqueeze(1).repeat(
                    1, self.noise_dim
                )

                sampled_labels_formatted = sampled_labels_formatted.unsqueeze(1)

                # Train the generator
                # print()
                # print("Train the generator:")

                # print()
                # print("My 'sampled_labels_formatted':")
                # print(sampled_labels_formatted.shape)

                # print()
                # print("My 'noise':")
                # print(noise.shape)

                gen_data = torch.cat((noise, sampled_labels_formatted), dim=1)
                gen_params = self.generator(gen_data)
                gen_inputs = torch.cat((gen_params, sampled_labels.unsqueeze(1)), dim=1)
                pred_gen = self.discriminator(gen_inputs)
                g_loss = criterion(pred_gen, valid)
                g_loss.backward()
                self.optimizer_generator.step()
                g_loss_value = g_loss.item()

                epoch_d_loss.append(d_loss.to("cpu").detach().numpy())
                epoch_g_loss.append(g_loss.to("cpu").detach().numpy())

            # epoch_d_loss = torch.tensor(epoch_d_loss).float()
            # epoch_g_loss = torch.tensor(epoch_g_loss).float()

            all_d_loss.append(np.mean(epoch_d_loss))
            all_g_loss.append(np.mean(epoch_g_loss))
            # all_acc.append(np.mean(epoch_acc))

            """# Discriminator overpower check
            if epoch >= 200:
                moving_average = np.mean(all_acc[-200:])
                if moving_average >= 90:
                    print(f'Moving average: {moving_average}')
                    break"""

            # Plot the progress
            mean_d_loss = np.mean(epoch_d_loss)
            # mean_acc = np.mean(epoch_acc)
            mean_g_loss = np.mean(epoch_g_loss)

            print(f"Epoch {epoch}, D loss: {mean_d_loss}, G loss: {mean_g_loss}")

            # Generate data at every sample interval

            if epoch % sample_interval == 0:
                if self.num_classes == 2:
                    self.sample_parameters(epoch, n_samples, cond_class=-1)
                else:
                    raise ValueError("The current code works with two classes for now")

        return all_d_loss, all_g_loss, all_acc

    def sample_parameters(self, epoch, n_samples, cond_class):

        noise = torch.randn(n_samples, self.latent_dim, self.noise_dim, device=device)

        # Create the conditional label for cond_class
        sampled_labels = np.ones(n_samples) * cond_class
        sampled_labels = torch.tensor(
            sampled_labels, dtype=torch.float32, device=device
        )

        sampled_labels_formatted = sampled_labels.unsqueeze(1).repeat(1, self.noise_dim)

        sampled_labels_formatted = sampled_labels_formatted.unsqueeze(1)

        data = torch.cat([noise, sampled_labels_formatted], dim=1)

        gen_par = self.generator(data).to("cpu").detach().numpy()

        # Rescale parameters according to previous scaling on X_train
        x_new, new_min, new_max = hp.unscale_range(
            gen_par, -1.0, 1.0, self.min_x, self.max_x
        )
        class_label = "r" if cond_class == -1 else "nr"
        np.save(f"{self.savepath}{epoch}_{class_label}.npy", x_new)

        # and save the corresponding generator and descriminator
        path_models = f"{self.savepath}saved_models/"
        os.makedirs(path_models, exist_ok=True)

        t
        # self.generator.save(f"{path_models}generator_{epoch}.h5")
        #    self.discriminator.save(f'{path_models}discriminator_{epoch}.h5')
