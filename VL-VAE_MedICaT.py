import os
import time
import json
import torch
import numpy as np
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from torchvision.models import vgg16
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()

# Print the device information
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

class Config:
    def __init__(self):
        # General settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_dir = "../Datasets/DeepEyeNet"
        self.train_json = "../Datasets/DeepEyeNet/DeepEyeNet_train.json"
        self.val_json = "../Datasets/DeepEyeNet/DeepEyeNet_valid.json"
        self.test_json = "../Datasets/DeepEyeNet/DeepEyeNet_test.json"
        self.corpus_file = "corpus.txt"

        # Model settings
        self.enc_type = 'EfficientNetB0'
        self.embed_dim = 1024
        self.hidden_dim = 2048
        self.num_heads = 8
        self.num_layers = 6
        self.num_kv_heads = 2
        self.vocab_size = 3000
        self.max_seq_len = 50
        self.dropout = 0.1  # Dropout rate for regularization
        self.norm_eps = 1e-5
        self.latent_dim = 512

        # Training settings
        self.freeze_encoder = False
        self.epochs = 25
        self.batch_size = 4
        self.lr = 1e-4
        self.weight_decay = 1e-5
        self.beam_size = 5
        self.save_best_model = True
        self.lambda1 = 1
        self.lambda2 = 1
        self.lambda3 = 1

        # Image transformation settings
        self.image_size = (256, 256)  # Resize images to (224, 224)
        self.mean = [0.5, 0.5, 0.5]  # Pre-trained ResNet mean
        self.std = [0.25, 0.25, 0.25]  # Pre-trained ResNet std

        # Model saving settings
        self.model_save_path = "../DeepEyeNet/vlvae_den_best_model_25e.pth"

    def __repr__(self):
        return f"Config(device={self.device}, embed_dim={self.embed_dim}, " \
               f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, " \
               f"num_layers={self.num_layers}, vocab_size={self.vocab_size}, " \
               f"max_seq_len={self.max_seq_len}, epochs={self.epochs}, " \
               f"batch_size={self.batch_size}, lr={self.lr}, weight_decay={self.weight_decay}, " \
               f"beam_width={self.beam_size}, dropout={self.dropout}, " \
               f"save_best_model={self.save_best_model}, " \
               f"lambda1={self.lambda1}, lambda2={self.lambda2}, lambda3={self.lambda3})"

    def _process_data(self,file_path):
        with open(file_path) as file:
            data_dict = json.load(file)
            
        paths, keywords, descs = [], [], []
        for item in data_dict:
            paths.append(list(item.keys())[0])
            keyword = list(item.values())[0]['keywords']
            keywords.append(keyword)
            descs.append(list(item.values())[0]['clinical-description'])
            
        data = pd.DataFrame({"image_path": paths, "keywords": keywords, "caption": descs})
        
        return data
        
    def data_prep(self):        
        train_data = self._process_data(self.train_json)
        val_data = self._process_data(self.val_json)
        train_data = pd.concat([train_data, val_data])
        test_data = self._process_data(self.test_json)

        return train_data, test_data

    def get_image_transforms(self):
        """
        Returns the transformation pipeline to be applied to images.
        Uses self.image_size, self.mean, and self.std from config.
        """
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_optimizer(self, model):
        """
        Returns the optimizer (AdamW) for the given model.
        """
        from torch.optim import AdamW
        return AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def load_checkpoint(self, model, checkpoint_path=None):
        """
        Load model checkpoint if available.
        If no path is given, default to loading from self.model_save_path.
        """
        checkpoint_path = checkpoint_path or self.model_save_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

    def save_checkpoint(self, model, epoch, loss):
        """
        Save model checkpoint at the specified path.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, self.model_save_path)
        print(f"Checkpoint saved to {self.model_save_path}")

cfg = Config()

# Read the data from CSV
train_df, test_df = cfg.data_prep()

# concatenate to create corpus for entire dataset
report_df = pd.concat([train_df, test_df])

# Define a function to process text: remove special characters and extra spaces
def clean_text(text):
    # Check if the text is NaN or not a string, if so return an empty string
    if not isinstance(text, str):
        text = ''
    # Replace non-alphanumeric characters (including punctuation) with space
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    # Strip leading/trailing spaces and collapse multiple spaces
    text = ' '.join(text.split()).lower()
    return text

# Process the 'MeSH' (keywords) and 'findings' (description) columns
processed_keywords = report_df['keywords'].apply(clean_text)
processed_caption = report_df['caption'].apply(clean_text)

# Concatenate processed keywords and descriptions with a space
combined = processed_keywords + " " + processed_caption

# Save the combined text to a .txt file
output_file = cfg.corpus_file
combined.to_csv(output_file, index=False, header=False)

print(f"Processed data saved to {output_file}")

#### Tokenizer ####
class Tokenizer:
    def __init__(self, corpus_file, vocab_size=3000):
        
        from collections import Counter
        from itertools import chain

        with open(corpus_file, 'r') as f:
            captions = [line.strip() for line in f]

        # Calculate word frequency
        word_freq = Counter(chain.from_iterable(caption.split() for caption in captions))
        # Filter words with at least 3 occurrences
        filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= 3}
        # Get the most common words
        most_common = Counter(filtered_word_freq).most_common(vocab_size - 4)

        self.word2idx = {w: i + 4 for i, (w, _) in enumerate(most_common)}
        self.word2idx.update({"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3})
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text):
        tokens = ["<start>"] + text.split() + ["<end>"]
        return [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokens]

    def decode(self, indices):
        return " ".join(self.idx2word[idx] for idx in indices if idx > 3)
    
#### Dataset ####
class DeepEyeNetDataset(Dataset):
    def __init__(self, cfg, tokenizer, data, transform=None):

        self.image_dir = cfg.image_dir
        self.data = data
        self.max_seq_len = cfg.max_seq_len
        self.tokenizer = tokenizer
        # Load image IDs
        self.image_ids = self.data['image_path'].values
        # Apply transformations
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_id)

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        filtered_row = self.data[self.data['image_path'] == img_id]
        keywords = filtered_row['keywords'].apply(clean_text)
        caption = filtered_row['caption'].apply(clean_text)

        # Select one random caption for training
        keyword = self.tokenizer.encode(keywords.iloc[0])
        caption = self.tokenizer.encode(caption.iloc[0])

        # Pad the caption to max_seq_len
        keyword = keyword[:self.max_seq_len]  # Trim if longer than max_seq_len
        keyword += [self.tokenizer.word2idx["<pad>"]] * (self.max_seq_len - len(keyword))  # Pad if shorter

        # Pad the caption to max_seq_len
        caption = caption[:self.max_seq_len]  # Trim if longer than max_seq_len
        caption += [self.tokenizer.word2idx["<pad>"]] * (self.max_seq_len - len(caption))  # Pad if shorter

        return image, torch.tensor(keyword), torch.tensor(caption)

#### Guided Context Attention ####
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GlobalContextAttention(nn.Module):
    def __init__(self, filters, reduction_ratio=8, kernel = 1, transform_output_activation='linear'):
        super(GlobalContextAttention, self).__init__()
        self.channels = filters
        self.kernel = kernel

        # Context Formulation block
        self.context_conv = nn.Conv2d(self.channels, 1, kernel_size=self.kernel, stride=1, padding=0, bias=False)

        # Channel Correlation / Transform block
        self.transform_bottleneck = nn.Conv2d(self.channels, self.channels // reduction_ratio, kernel_size=self.kernel, stride=1, padding=0, bias=False)
        self.transform_activation = nn.ReLU(inplace=True)
        
        self.layer_norm = nn.LayerNorm([self.channels // reduction_ratio])
    
        self.transform_conv = nn.Conv2d(self.channels // reduction_ratio, self.channels, kernel_size=self.kernel, stride=1, padding=0, bias=False)
        self.transform_output_activation = nn.Identity(self.channels)
        
        # Module's output
        self.output = None


    def forward(self, x):
        batch_size = x.size(0)

        # Context Formulation block
        input_context_1 = x.view(x.size(0), x.size(1), -1)   # (B, C, H*W)
        
        input_context_2 = self.context_conv(x)  # (B, 1, H, W)
        input_context_2 = F.softmax(input_context_2, dim=-1)   # (B, 1, H, W)
        # Reshape to (B, H*W, 1)
        batch, _, height, width = input_context_2.size()
        input_context_2 = input_context_2.view(batch, height*width, 1)   # (B, H*W, 1)

        # Compute context block outputs
        context = torch.bmm(input_context_1, input_context_2)  # (B, C, 1)
        context = context.view(context.size(0), context.size(1), -1, 1) # (B, C, 1, 1)

        # Channel Correlation / Transform block
        transform = self.transform_bottleneck(context)
        transform = self.transform_activation(transform)  # (B, C/k, 1, 1)
        transform = transform.squeeze(-1).squeeze(-1)     # (B, C/k)
        transform = self.layer_norm(transform)
        transform = transform.view(transform.size(0), transform.size(1), 1, 1)   # (B, C/k, 1, 1)
        transform = self.transform_conv(transform)        # (B, C, 1, 1)
        transform = self.transform_output_activation(transform)

        # Apply context transform
        self.output = x + transform    # (B, C, H, W)
        return self.output

class AttentionGate(nn.Module):
    def __init__(self, filters):
        super(AttentionGate, self).__init__()
        
        self.conv_x = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, padding='same', bias=False)
        self.conv_g = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, padding='same', bias=False)
        self.psi = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=1, stride=1, padding='same', bias=False)
        self.layer_norm = nn.LayerNorm(filters)
        self.bxg = nn.Parameter(torch.zeros(filters))
        self.bpsi = nn.Parameter(torch.zeros(1))
        
        # Module's output
        self.output = None

    def forward(self, x, g):
        # Apply convolutional operations
        x_conv = self.conv_x(x)
        g_conv = self.conv_g(g)

        # Compute additive attention
        att = F.relu(x_conv + g_conv + self.bxg.view(1, -1, 1, 1))
        
        att = att.permute(0, 2, 3, 1)   # Permuting (B, H, W, C) 'cause layer_norm expects last dimension to be C
        att = self.layer_norm(att)
        att = att.permute(0, 3, 1, 2)   # Permuting back to (B, C, H, W) as PyTorch expects
        att = self.psi(att) + self.bpsi.view(1, -1, 1, 1)
        att = torch.sigmoid(att)

        self.output = att * x
        return self.output
    
#### Image Encoder ####

# Step 1: Image Encoder Placeholder (can be EfficientNet)
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, freeze_encoder=False):
        super(ImageEncoder, self).__init__()

        # Load the pre-trained EfficientNetB0 model
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        
        # Remove the fully connected layer (last layer)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        # Dynamically find the last convolutional layer
        last_conv_layer = self._get_last_conv_layer(self.model)
        
        # Get the output channels of the last convolutional layer
        conv_output_channels = last_conv_layer.out_channels

        # Global Context Attention
        self.gc_attention = GlobalContextAttention(filters=conv_output_channels)
        
        # Attention Gate Block
        self.attention_gate = AttentionGate(filters=conv_output_channels)
        
        # Define the final fully connected layer for embedding
        self.fc = nn.Linear(conv_output_channels, embed_dim)

        # Optionally freeze the encoder layers
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Pass through the convolutional part of the model to get features
        features = self.model(x)  # (Batch, C, H, W)

        # Apply Global Context Attention
        gc_features = self.gc_attention(features)
        att_features = self.attention_gate(features, gc_features)
        
        # Reshape to (batch_size, H * W, channels)
        batch_size, channels, height, width = att_features.size()   
        att_features = att_features.view(batch_size, -1, channels,)  # (Batch, H*W, C)
        
        # Optionally pass through the final embedding layer to reduce dimensionality
        image_features = self.fc(att_features)  # (Batch, H*W, embed_dim)
        
        return image_features

    def _get_last_conv_layer(self, model):
        """
        Helper method to get the last convolutional layer in the ResNet model.
        It iterates through the blocks and returns the final convolutional layer.
        """
        # ResNet is composed of several blocks of layers
        for layer in model:
            if isinstance(layer, nn.Conv2d):  # Check if the layer is a Conv2d layer
                last_conv_layer = layer
        return last_conv_layer

#### Text Encoder ####
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_length):
        """
        Args:
            vocab_size (int): Size of the vocabulary for the embedding layer.
            embed_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads in the multi-head attention block.
            hidden_dim (int): Dimension of the feed-forward network inside the transformer encoder.
            num_layers (int): Number of transformer encoder layers.
            max_seq_length (int): Maximum sequence length for positional encoding.
        """
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(max_seq_length, embed_dim), requires_grad=False)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
    
    def _generate_positional_encoding(self, max_seq_length, embed_dim):
        """
        Generate sinusoidal positional encoding.
        """
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (Tensor): Input tensor of shape (batch_size, seq_length), containing token indices.
            attention_mask (Tensor, optional): Mask tensor of shape (batch_size, seq_length), where 1 indicates
                                               tokens to be attended to and 0 indicates tokens to be ignored.
        Returns:
            Tensor: Output embeddings of shape (batch_size, seq_length, embed_dim).
        """
        batch_size, seq_length = input_ids.shape
        
        # Embedding lookup
        embeddings = self.embedding(input_ids)  # (batch_size, seq_length, embed_dim)
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:, :seq_length, :]
        
        # Prepare mask for transformer encoder
        if attention_mask is not None:
            # Convert attention mask to the format required by nn.TransformerEncoder
            # Shape should be (seq_length, seq_length) per nn.TransformerEncoder requirements
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)  # (batch_size, seq_length, seq_length)
            attention_mask = attention_mask.transpose(0, 1).transpose(1, 2)  # (seq_length, batch_size, seq_length)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)
        
        # Pass through transformer encoder
        embeddings = embeddings.transpose(0, 1)  # Transformer expects (seq_length, batch_size, embed_dim)
        output = self.transformer_encoder(embeddings, src_key_padding_mask=None)  # (seq_length, batch_size, embed_dim)
        output = output.transpose(0, 1)  # Back to (batch_size, seq_length, embed_dim)
        
        return output

#### Multi-modal VAE ####
class VisionLanguageVAE(nn.Module):
    def __init__(self, config, pad_idx=0):
        super(VisionLanguageVAE, self).__init__()

        self.config = config
        self.latent_dim = config.latent_dim
        ####### IMAGE VAE SETUP ######
        # Load pretrained VGG16
        vgg = vgg16(pretrained=True)
        self.img_encoder = nn.Sequential(*list(vgg.features.children())[:-2]) # Taking the output of last conv layer 14x14x512

        # Dynamically find the last convolutional layer
        self.target_layer = self._get_last_conv_layer(self.img_encoder)
        
        # Global Context Attention
        self.gc_attention = GlobalContextAttention(filters=self.target_layer.out_channels)
        
        # Attention Gate Block
        self.attention_gate = AttentionGate(filters=self.target_layer.out_channels)
        
        # Register forward hook to capture the output
        self.feature_hook = {}
        self.target_layer.register_forward_hook(self._capture_features_hook)

        # Latent layers
        self.flatten_size = None # To be computed dynamically 
        self.img_fc_mu = None # To be instantiated dynamically
        self.img_fc_logvar = None # To be instantiated dynamically
        
        # self.img_fc_mu = nn.Linear(self.flatten_size, latent_dim)
        # self.img_fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Softmax layer for output (image decoder)
        self.sigmoid = nn.Sigmoid()
        
        # Decoder setup
        self.img_fc_decoder = None # To be instantiated dynamically
        # self.img_fc_decoder = nn.Linear(latent_dim, self.flatten_size)
        
        self.img_decoder = self._img_build_decoder()

        ####### TEXT VAE SETUP ######
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=pad_idx)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads)
        self.txt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Latent layers
        self.txt_fc_mu = nn.Linear(config.embed_dim * config.max_seq_len, config.latent_dim)
        self.txt_fc_logvar = nn.Linear(config.embed_dim * config.max_seq_len, config.latent_dim)
        
        # Latent space back to embedding space
        self.txt_fc_decoder_input = nn.Linear(config.latent_dim, config.embed_dim * config.max_seq_len)
        
        # Transformer Decoder (mirrors encoder)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.embed_dim, nhead=config.num_heads)
        self.txt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        
        # Final output layer to vocabulary size
        self.txt_fc_output = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Other parameters
        self.max_seq_len = config.max_seq_len
        self.embed_dim = config.embed_dim

        ###### JOINT LATENT SPACE SETUP ######
        self.joint_latents_layer = nn.Linear(2 * config.latent_dim, config.latent_dim)
        self.relu = nn.ReLU()
        self.joint_fc_mu = nn.Linear(config.latent_dim, config.latent_dim)
        self.joint_fc_logvar = nn.Linear(config.latent_dim, config.latent_dim)

    def _img_build_decoder(self):
        """
        Dynamically constructs a decoder that mirrors the encoder layers.
        Adjusts the ConvTranspose2d parameters to ensure proper upsampling.
        """
        layers = []
        decoder_layers = list(self.img_encoder.children())[::-1]  # Reverse encoder layers
        for layer in decoder_layers:
            if isinstance(layer, nn.Conv2d):
                # Correctly configure ConvTranspose2d
                layers.append(nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                ))
            elif isinstance(layer, nn.ReLU):
                layers.append(nn.ReLU(inplace=True))
            elif isinstance(layer, nn.MaxPool2d):
                # Replace MaxPool with Upsample
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        return nn.Sequential(*layers)

    def _get_last_conv_layer(self, model):
        """
        Helper method to get the last convolutional layer in the model.
        It iterates through the blocks and returns the final convolutional layer.
        """
        for layer in model:
            if isinstance(layer, nn.Conv2d):  # Check if the layer is a Conv2d layer
                last_conv_layer = layer
        return last_conv_layer

    def _capture_features_hook(self, module, input, output):
        """
        Hook function to capture the output features of the last convolutional layer.
        """
        self.feature_hook['features'] = output  # Save the output for dynamic computation


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_img, x_txt, src_mask=None):
        """
        Forward pass: encodes img and txt inputs, samples latent vector, and decodes.
        """
        #### IMAGE ENCODE ####
        # Encode
        x_img = self.img_encoder(x_img)

        # Dynamically compute the flattened size of the last convolutional layer output
        if self.flatten_size is None:
            B, C, H, W = x_img.shape  # Extract batch size, channels, height, and width
            self.flatten_size = C * H * W  # Compute flattened size
            
            # Define the fully connected layers (mu, logvar & decoder) based on the dynamic flattened size
            self.img_fc_mu = nn.Linear(self.flatten_size, self.latent_dim).to(self.config.device)
            self.img_fc_logvar = nn.Linear(self.flatten_size, self.latent_dim).to(self.config.device)
            self.img_fc_decoder = nn.Linear(self.latent_dim, self.flatten_size).to(self.config.device)
            
        x_encoded = torch.flatten(x_img, start_dim=1)
        img_mu = self.img_fc_mu(x_encoded)
        img_logvar = self.img_fc_logvar(x_encoded)
        
        # Reparameterization trick
        z_img = self.reparameterize(img_mu, img_logvar)

        #### TEXT ENCODE ####
        # Embedding and positional encoding
        x_txt_embd = self.embedding(x_txt) + self.positional_encoding[:, :x_txt.size(1), :]

        # Encode
        txt_encoded = self.txt_encoder(x_txt_embd.permute(1, 0, 2), src_key_padding_mask=src_mask)  # (seq_len, batch_size, embed_dim)
        txt_encoded = txt_encoded.permute(1, 0, 2).contiguous()  # (batch_size, seq_len, embed_dim)
        txt_encoded_flat = txt_encoded.view(txt_encoded.size(0), -1)  # Flatten (batch_size, seq_len * embed_dim)
        
        # Compute latent space
        txt_mu = self.txt_fc_mu(txt_encoded_flat)
        txt_logvar = self.txt_fc_logvar(txt_encoded_flat)
        z_txt = self.reparameterize(txt_mu, txt_logvar)

        #### Joint Latent Space Modelling ####
        joint_latents = torch.cat((z_img, z_txt), dim=1)
        joint_latents = self.relu(self.joint_latents_layer(joint_latents))
        joint_mu = self.joint_fc_mu(joint_latents)
        joint_logvar = self.joint_fc_logvar(joint_latents)
        z_shared = self.reparameterize(joint_mu, joint_logvar)
        
        #### IMAGE DECODE ####
        # Map latent space back to embedding space
        txt_decoder_input = self.txt_fc_decoder_input(z_txt).view(-1, self.max_seq_len, self.embed_dim)  # Reshape to (batch_size, seq_len, embed_dim)
        
        # Decode
        x_img_decoded = self.img_fc_decoder(z_img)
        x_img_decoded = x_img_decoded.view(-1, x_img.shape[1], x_img.shape[2], x_img.shape[3])  # Reshape to match encoder output
        x_img_decoded = self.img_decoder(x_img_decoded)

        # Apply sigmoid to normalize output
        x_img_output = self.sigmoid(x_img_decoded)

        #### TEXT DECODE ####
        # Decode
        x_txt_decoded = self.txt_decoder(
            txt_decoder_input.permute(1, 0, 2),
            txt_decoder_input.permute(1, 0, 2),
            memory_key_padding_mask=src_mask
        )  # (seq_len, batch_size, embed_dim)
        x_txt_decoded = x_txt_decoded.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        
        # Project to vocabulary space
        x_txt_output = self.txt_fc_output(x_txt_decoded)  # (batch_size, seq_len, vocab_size)
        
        return x_img_output, img_mu, img_logvar, z_img, x_txt_output, txt_mu, txt_logvar, z_txt, joint_mu, joint_logvar, z_shared
    

#### Transformer Decoder ####
# Step 2: RMSNorm Layer
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Step 3: Utility Functions (Rotery Positional Embeddings - Q & K)

# a: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# b: Reshape frequency tensor for broadcasting it with another tensor.
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# c: Apply rotary embeddings to input tensors using the given frequency tensor.
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Step 4: Grouped Query Attention

# a: repeat kv for specified intervals
def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# b: Attention layer
class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, config):
        super().__init__()
        self.n_kv_heads = config.num_kv_heads
        self.n_heads = config.num_heads
        self.n_rep = config.num_heads // config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.wq = nn.Linear(
            config.embed_dim,
            config.num_heads * self.head_dim,
            bias=False,
        )
        
        self.wk = nn.Linear(
            config.embed_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        
        self.wv = nn.Linear(
            config.embed_dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        
        self.wo = nn.Linear(
            config.num_heads * self.head_dim,
            config.embed_dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (
                config.batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                config.batch_size,
                config.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

    def forward(self,x,start_pos,freqs_cis,mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq.device)
        self.cache_v = self.cache_v.to(xq.device)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)
        
        # scaled dot product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# c: Cross-Attention Layer
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections of query, key, value
        Q = self.q_linear(query)  # [batch_size, seq_len, embed_dim]
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # Apply attention weights to the values
        output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate the heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final linear layer
        output = self.out_linear(output)  # [batch_size, seq_len, embed_dim]

        return output, attention_weights

# Step 5: FeedForward SwiGLU
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim,):
        super().__init__()

        self.w1 = nn.Linear(
            embed_dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, embed_dim, bias=False,
        )
        self.w3 = nn.Linear(
            embed_dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))        

# Step 6: Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.n_heads = config.num_heads
        self.dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        
        self.self_attention = Attention(config)

        self.cross_attention = MultiheadAttention(config.embed_dim, config.num_heads) 
    
        self.feed_forward = FeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.embed_dim,
        )
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

    def forward(self, lang_seq, img_features, start_pos, freqs_cis, mask,):
        h1 = lang_seq + self.self_attention(
            self.attention_norm(lang_seq), start_pos, freqs_cis, mask
        )

        h2, attn_weights = self.cross_attention(
            self.attention_norm(h1), self.attention_norm(img_features), self.attention_norm(img_features)
        )
        
        h2 = h1 + h2
        out = h2 + self.feed_forward(self.ffn_norm(h2))
        
        return out

# Step 7: Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_layers
        self.head_dim = config.embed_dim // config.num_heads

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.embed_dim,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.num_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.output = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.head_dim, self.config.max_seq_len * 2
        )

    def forward(self, tokens, img_embs, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, img_embs, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

#### Fusion Blocks ####

class TransFusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.V2L = MultiheadAttention(cfg.embed_dim, cfg.num_heads)
        self.L2V = MultiheadAttention(cfg.embed_dim, cfg.num_heads)
        self.rms_norm_image = RMSNorm(cfg.embed_dim, cfg.norm_eps)
        self.rms_norm_text = RMSNorm(cfg.embed_dim, cfg.norm_eps)

    def forward(self, image, text):
        image = self.rms_norm_image(image)
        text = self.rms_norm_text(text)
        v2l, _ = self.V2L(image, text, text) # (B, 64, emded_dim)
        v2l = v2l + image
        l2v, _ = self.L2V(text, image, image) # (B, 50, emded_dim)
        l2v = l2v + text
        concat_features = torch.cat((v2l, l2v), dim=1) # (B, 114, emded_dim)
        return concat_features

class LatentTransFusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, zi, zt, zs):
        zi = zi.unsqueeze(1) # Adds a dimension at position 1
        zt = zt.unsqueeze(1) # Adds a dimension at position 1
        zs = zs.unsqueeze(1) # Adds a dimension at position 1

        zit_featurs = torch.cat((zi, zt), dim=2)   # (B, 1, latent_dim*2)
        #zs_featurs = torch.cat((zs, zs), dim=2)   # (B, 1, latent_dim*2)
        
        #concat_features = torch.cat((zit_featurs, zs_featurs), dim=1)   # (B, 2, emded_dim)
        
        return zit_featurs

class FullFusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1, f2):
        concat_features = torch.cat((f1, f2), dim=1)   # (B, 50+64+2, emded_dim)
        return concat_features

#### The Model ####

# Step 8: Image Captioning Model with Attention Map Handling
class ImageCaptioningModel(nn.Module):
    def __init__(self, config, tokenizer):
        super(ImageCaptioningModel, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Initialize the image encoder (EfficientNet)
        self.image_encoder = ImageEncoder(config.embed_dim, config.freeze_encoder)
        self.text_encoder = TextEncoder(config.vocab_size, config.embed_dim, config.num_heads, config.hidden_dim, config.num_layers, config.max_seq_len)
        self.transfusion_encoder = TransFusionEncoder()

        # Instantiate the VisionLanguageVAE
        self.vl_vae = VisionLanguageVAE(config=self.config, pad_idx=0)
        
        self.latent_tfe = LatentTransFusionEncoder()

        # Initialize the full fusion block
        self.full_fusion_encoder = FullFusionEncoder()

        # Initialize the transformer decoder
        self.decoder = TransformerDecoder(config)

    def forward(self, images, keywords, captions):

        #### Embeddings from Attention Models ####
        features = self.image_encoder(images)
        embeddings = self.text_encoder(keywords)
        
        #### Latent Embeddings from VL-VAE ####
        x_img_output, img_mu, img_logvar, z_img, x_txt_output, txt_mu, txt_logvar, z_txt, joint_mu, joint_logvar, z_shared = self.vl_vae(images, keywords)

        #### Implicit Fusion of Attention Features ####
        transfused_features = self.transfusion_encoder(features, embeddings)

        #### Explicit Fusion of Latent Features ####
        latent_transfused_features = self.latent_tfe(z_img, z_txt, z_shared)

        #### Full Fusion ####
        fused_features = self.full_fusion_encoder(transfused_features, latent_transfused_features)
        
        logits = self.decoder(captions, fused_features, start_pos=0)
        return logits, x_img_output, x_txt_output, img_mu, txt_mu, joint_mu, img_logvar, txt_logvar, joint_logvar, z_img, z_txt, z_shared 

#### VLVAE Loss ####
import torch
import torch.nn as nn
import torch.nn.functional as F

class VLVAELoss(nn.Module):
    def __init__(self, pad_idx=0):
        super(VLVAELoss, self).__init__()
        self.pad_idx = pad_idx
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    def forward(self, x_img, x_img_bar, x_txt, x_txt_bar, img_mu, img_logvar, txt_mu, txt_logvar, joint_mu, joint_logvar):
        # Image reconstruction loss (MSE)
        mse_img_recon_loss = F.mse_loss(x_img_bar, x_img, reduction='sum')

        # Image-specific regularization loss (KL Divergence)
        kld_img_reg_loss = -0.5 * torch.sum(1 + img_logvar - img_mu.pow(2) - img_logvar.exp())

        # Text reconstruction loss (CrossEntropy)
        x_txt_bar = x_txt_bar.view(-1, x_txt_bar.size(-1))  # (batch_size * seq_len, vocab_size)
        x_txt = x_txt.view(-1)  # (batch_size * seq_len)
        ce_txt_recon_loss = self.cross_entropy(x_txt_bar, x_txt)

        # Text-specific regularization loss (KL Divergence)
        kld_txt_reg_loss = -0.5 * torch.sum(1 + txt_logvar - txt_mu.pow(2) - txt_logvar.exp())

        # Shared modality regularization loss (KL Divergence)
        kld_shared_reg_loss = -0.5 * torch.sum(1 + joint_logvar - joint_mu.pow(2) - joint_logvar.exp())

        # Total VL-VAE loss
        total_vlvae_loss = mse_img_recon_loss + kld_img_reg_loss + ce_txt_recon_loss + kld_txt_reg_loss #+ kld_shared_reg_loss

        return total_vlvae_loss
    
#### Representation Orthogonality Constraint ####

class OrthogoDiffLoss(nn.Module):

    def __init__(self):
        super(OrthogoDiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
    
#### Shared Information Alignment Loss ####

class SharedInformationAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        Initializes the SharedInformationAlignmentLoss.
        
        Args:
            temperature (float): Temperature parameter \( \tau \) for scaling similarities.
        """
        super(SharedInformationAlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_img, z_txt, z_shared):
        """
        Computes the alignment loss.

        Args:
            z_img (torch.Tensor): Image embeddings, shape (batch_size, embedding_dim).
            z_txt (torch.Tensor): Text embeddings, shape (batch_size, embedding_dim).
            z_shared (torch.Tensor): Shared embeddings, shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The alignment loss.
        """
        # Normalize the embeddings to unit vectors
        z_img = F.normalize(z_img, dim=1)
        z_txt = F.normalize(z_txt, dim=1)
        z_shared = F.normalize(z_shared, dim=1)

        # Compute similarities
        sim_s_i = torch.mm(z_shared, z_img.t()) / self.temperature  # (batch_size, batch_size)
        sim_s_t = torch.mm(z_shared, z_txt.t()) / self.temperature  # (batch_size, batch_size)

        # Create ground-truth labels for similarity (diagonal should match across batches)
        labels = torch.arange(z_shared.size(0), device=z_shared.device)

        # Compute cross-entropy losses
        loss_s_i = F.cross_entropy(sim_s_i, labels)
        loss_s_t = F.cross_entropy(sim_s_t, labels)

        # Combine losses
        loss = loss_s_i + loss_s_t
        return loss
    
#### Training & Evaluation through Loaders ####

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Initialize model, criterion, optimizer, and dataloaders
config = Config()
train_data, test_data = config.data_prep()
tokenizer = Tokenizer(corpus_file=config.corpus_file, vocab_size=config.vocab_size)
model = ImageCaptioningModel(config, tokenizer).to(config.device)
transform = config.get_image_transforms()

# Optimizer and criterion
optimizer = config.get_optimizer(model)
vl_vae_criterion = VLVAELoss()
ortho_diff_loss = OrthogoDiffLoss()
shared_info_align_loss = SharedInformationAlignmentLoss()
decoder_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<pad>"])

# Dataset and DataLoaders
train_dataset = DeepEyeNetDataset(cfg=cfg, tokenizer=tokenizer, data=train_data, transform=transform)
test_dataset = DeepEyeNetDataset(cfg=cfg, tokenizer=tokenizer, data=test_data, transform=transform)

# print("Train Size: ", len(train_dataset))
# print("Test Size: ", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=4)

# Track loss history
train_losses, val_losses = [], []

# Helper function to compute accuracy during validation
def compute_accuracy(outputs, captions):
    # Get the predicted tokens (argmax of logits)
    predicted = outputs.argmax(dim=-1)
    # Exclude <pad> tokens and calculate accuracy
    correct = (predicted == captions).float()
    correct = correct[captions != tokenizer.word2idx["<pad>"]]  # Ignore padding tokens
    accuracy = correct.sum() / correct.size(0)  # Accuracy as percentage
    return accuracy.item()

# Training and validation loop
best_loss = float('inf')
# Record training_start_time
training_start_time =  time.time()
for epoch in range(config.epochs):
    
    # Record train epoch start time
    train_epoch_start_time = time.time()
    
    model.train()
    train_loss = 0

    # TQDM for training
    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{config.epochs}") as pbar_train:
        for images, keywords, captions in train_loader:
            images, keywords, captions = images.to(config.device), keywords.to(config.device), captions.to(config.device)

            # Forward pass
            optimizer.zero_grad()
            outputs, x_img_output, x_txt_output, img_mu, txt_mu, joint_mu, img_logvar, txt_logvar, joint_logvar, z_img, z_txt, z_shared = model(images, keywords, captions[:, :-1])  # Exclude <end> token 

            # Loss Terms
            decoder_loss = decoder_criterion(outputs.reshape(-1, config.vocab_size), captions[:, 1:].reshape(-1))  # Shift target by 1
            
            vl_vae_loss = vl_vae_criterion(images, x_img_output, keywords, x_txt_output, img_mu, img_logvar, txt_mu, txt_logvar, joint_mu, joint_logvar)

            # Apply Orthogonality Constraint on z_img, z_txt, z_shared
            zi_zt_loss = ortho_diff_loss(z_img, z_txt)
            zi_zs_loss = ortho_diff_loss(z_img, z_shared)
            zt_zs_loss = ortho_diff_loss(z_txt, z_shared)

            # Compute Shared Information Alignment Loss
            align_loss = shared_info_align_loss(z_img, z_txt, z_shared)
            
            # Compute total  Orthogonality Constraint
            total_diff_loss = zi_zt_loss + zi_zs_loss + zt_zs_loss

            # Total loss
            loss = decoder_loss + (config.lambda1 * vl_vae_loss) + (config.lambda2 * total_diff_loss) + (config.lambda3 * align_loss)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            pbar_train.set_postfix({"Batch Loss": loss.item()})
            pbar_train.update(1)

    # Record train epoch end time
    train_epoch_end_time = time.time()
    # Find total train epoch time
    train_epoch_total_time = train_epoch_end_time - train_epoch_start_time
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss, val_accuracy = 0, 0

    # Record val epoch start time
    val_epoch_start_time = time.time()
    
    # TQDM for validation
    with tqdm(total=len(val_loader), desc=f"Validating Epoch {epoch + 1}/{config.epochs}") as pbar_val:
        for images, keywords, captions in val_loader:
            images, keywords, captions = images.to(config.device), keywords.to(config.device), captions.to(config.device)

            # Forward pass
            outputs, x_img_output, x_txt_output, img_mu, txt_mu, joint_mu, img_logvar, txt_logvar, joint_logvar, z_img, z_txt, z_shared = model(images, keywords, captions[:, :-1])  # Exclude <end> token

            decoder_loss = decoder_criterion(outputs.reshape(-1, config.vocab_size), captions[:, 1:].reshape(-1))  # Shift target by 1
            vl_vae_loss = vl_vae_criterion(images, x_img_output, keywords, x_txt_output, img_mu, img_logvar, txt_mu, txt_logvar, joint_mu, joint_logvar)
            
            # Apply Orthogonality Constraint on z_img, z_txt, z_shared
            zi_zt_loss = ortho_diff_loss(z_img, z_txt)
            zi_zs_loss = ortho_diff_loss(z_img, z_shared)
            zt_zs_loss = ortho_diff_loss(z_txt, z_shared)
            
            # Compute total  Orthogonality Constraint
            total_diff_loss = zi_zt_loss + zi_zs_loss + zt_zs_loss

            # Compute Shared Information Alignment Loss
            align_loss = decoder_loss + (config.lambda1 * vl_vae_loss) + (config.lambda2 * total_diff_loss) + (config.lambda3 * align_loss)

            # Total loss
            loss = (config.lambda1 * decoder_loss) + (config.lambda2 * vl_vae_loss) + (config.lambda3 * total_diff_loss)
            
            val_loss += loss.item()

            # Compute accuracy
            accuracy = compute_accuracy(outputs, captions[:, 1:])
            val_accuracy += accuracy

            pbar_val.set_postfix({"Batch Loss": loss.item(), "Accuracy": accuracy})
            pbar_val.update(1)

    # Record val epoch end time
    val_epoch_end_time = time.time()
    # Find total val epoch time
    val_epoch_total_time = val_epoch_end_time - val_epoch_start_time
    
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)  # Average accuracy across all batches
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Time: {train_epoch_total_time} sec, Val Time: {val_epoch_total_time} sec")

    # Save the best model based on validation loss
    if val_loss < best_loss and config.save_best_model:
        config.save_checkpoint(model, epoch + 1, val_loss)
        best_loss = val_loss

# Record training_end_time
training_end_time =  time.time()

total_time = training_end_time - training_start_time
print(f"Total training time: {total_time:.2f} seconds")

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, config.epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, config.epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig('acc_loss_plot.png')  # Saves the plot as an image file
plt.show()
    
#### Compute Model Params ####

# Function to compute trainable, non-trainable, and total parameters
def compute_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # Include registered buffers as non-trainable
    buffer_params = sum(b.numel() for b in model.buffers())
    non_trainable_params += buffer_params
    total_params = trainable_params + non_trainable_params
    return trainable_params, non_trainable_params, total_params

# Compute parameters
trainable, non_trainable, total = compute_parameters(model)

print(f"Trainable Parameters: {trainable}")
print(f"Non-Trainable Parameters: {non_trainable}")
print(f"Total Parameters: {total}")


#### Generate Caption Method ####
# Step 9: Generate Captions Method
def generate_captions(model, images, keywords):
    """
    Generates captions for a batch of images using the trained model.

    Args:
        model (nn.Module): The trained image captioning model.
        images (torch.Tensor): A batch of input image tensors (preprocessed).

    Returns:
        list of str: List of generated captions for each image in the batch.
    """
    model.eval()  # Set the model to evaluation mode
    generated_captions = []

    with torch.no_grad():
        # Ensure images are on the same device as the model
        images = images.to(model.config.device)
        keywords = keywords.to(model.config.device)

        # Start with the <start> token for each image in the batch
        batch_size = images.size(0)
        caption_tokens = torch.full((batch_size, 1), model.tokenizer.word2idx["<start>"], dtype=torch.long, device=model.config.device)

        for _ in range(model.config.max_seq_len):
            # Forward pass through the model for the entire batch
            outputs, _, _, _, _, _, _, _, _, _, _, _ = model(images, keywords, caption_tokens)  # Shape: (batch_size, seq_len, vocab_size)
            next_token_logits = outputs[:, -1, :]  # Get the logits for the last predicted token in each sequence
            
            # Get the next token (greedy decoding: pick the token with the highest probability)
            next_token = next_token_logits.argmax(dim=-1)  # Shape: (batch_size,)
            
            # Append the predicted token to the captions for each image
            caption_tokens = torch.cat([caption_tokens, next_token.unsqueeze(1)], dim=1)

            # Stop if the <end> token is generated for each image in the batch
            if (next_token == model.tokenizer.word2idx["<end>"]).all():
                break

        # Convert token indices to words and join them into sentences
        for i in range(batch_size):
            caption = " ".join(
                model.tokenizer.idx2word[token.item()] 
                for token in caption_tokens[i, 1:] 
                if token.item() != model.tokenizer.word2idx["<end>"]
            )
            generated_captions.append(caption)
        
        return generated_captions
    
#### Compute Metrics ####

import nltk
import pandas as pd

from nltk.translate.bleu_score import corpus_bleu
#!pip install rouge
from rouge import Rouge
#!pip install bert_score
from bert_score import score

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Initialize BLEU
nltk.download('punkt')

def evaluate_model(model, val_loader):
    model.eval()
    references, hypotheses = [], []
    actual_predicted_samples = []

    to_pil = ToPILImage()  # Convert tensor to PIL Image

    with torch.no_grad():
        for images, keywords, captions in tqdm(val_loader, desc="Evaluating"):
            images = images.to(config.device)
            keywords = keywords.to(config.device)
            captions = captions.to(config.device)
    
            # Generate captions for the batch
            generated_captions = generate_captions(model, images, keywords)
    
            # Prepare actual captions
            actual_captions = [
                " ".join(
                    model.tokenizer.idx2word[token.item()] 
                    for token in captions[i, 1:] 
                    if token.item() not in [model.tokenizer.word2idx["<pad>"], model.tokenizer.word2idx["<end>"]]
                )
                for i in range(len(captions))
            ]
    
            hypotheses.extend(generated_captions)
            references.extend(actual_captions)

            # Collect sample pairs with image (or index)
            for i in range(len(generated_captions)):
                # Convert the image from tensor to PIL for displaying
                image = to_pil(images[i].cpu())  # Move to CPU and convert to PIL Image
                kwds = model.tokenizer.decode(keywords[i].tolist())
                
                # Append sample information
                actual_predicted_samples.append({
                    "image": image,  # Store the image as a PIL image
                    "keywords": kwds,
                    "actual_caption": actual_captions[i],
                    "predicted_caption": generated_captions[i]
                })
    
    return references, hypotheses, actual_predicted_samples

references, hypotheses, actual_predicted_samples = evaluate_model(model, val_loader)

def compute_metrics(references, hypotheses, actual_predicted_samples):

    # Save the actual and predicted captions to CSV using Pandas
    df = pd.DataFrame(actual_predicted_samples)
    df.to_csv('results.csv', index=False)

    # Tokenize the captions
    actual_tokens = [tokens.split() for tokens in references]
    predicted_tokens = [tokens.split() for tokens in hypotheses]

    # Create reference list for corpus_bleu (list of lists of references)
    reference_tokens = [[actual] for actual in actual_tokens]

    # Compute BLEU scores
    b1 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.1, 0, 0, 0))
    b2 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    # ROUGE score
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(hypotheses, references, avg=True)
    rouge_score = rouge_scores['rouge-1']['f']

    # Compute BERTScore
    reference_captions = [tokens for tokens in references]
    prediction_captions = [tokens for tokens in hypotheses]
    P, R, F1 = score(prediction_captions, reference_captions, lang="en", verbose=False)

    # Display BLEU scores
    print('BLEU-1: %f' % b1)
    print('BLEU-2: %f' % b2)
    print('BLEU-3: %f' % b3)
    print('BLEU-4: %f' % b4)
    print('BLEU Avg: %f' % ((b1 + b2 + b3 + b4) / 4))

    print("ROUGE Score:", rouge_score)

    # Display average BERTScore for Precision, Recall, and F1
    print("BERTScore Precision:", P.mean().item())
    print("BERTScore Recall:", R.mean().item())
    print("BERTScore F1:", F1.mean().item())

    # Display 5 sample captions with images
    print("\nSample Captions:")
    for sample in actual_predicted_samples[:5]:
        # Display the image along with the actual and predicted captions
        print(f"Actual: {sample['actual_caption']}\nPredicted: {sample['predicted_caption']}")
        plt.figure(figsize=(5, 5))
        plt.imshow(sample["image"])
        plt.axis('off')
        #plt.title()
        plt.show()

compute_metrics(references, hypotheses, actual_predicted_samples)

#### Complute FLOPS ####

# !pip install thop
from thop import profile, clever_format

# Function to compute FLOPS
def compute_flops(model):
    image_input = torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(config.device)
    text_input = torch.randint(0, config.vocab_size, (1, config.max_seq_len))  # Dummy text sequences
    text_output = torch.randint(0, config.vocab_size, (1, config.max_seq_len))  # Dummy text sequences
    flops, _ = profile(model, inputs=((image_input.to(config.device), text_input.to(config.device), text_output.to(config.device))))
    return flops

flops = compute_flops(model)
print("FLOPs: ", flops / 1e9, "GFLOPs")