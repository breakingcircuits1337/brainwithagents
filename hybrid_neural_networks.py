import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

class BinarizeFunction(torch.autograd.Function):
    """Custom function for binarizing weights and activations"""
    
    @staticmethod
    def forward(ctx, input):
        # Binarize to -1 and +1
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient through unchanged
        return grad_output

def binarize(input):
    return BinarizeFunction.apply(input)

class BinarizedLinear(nn.Module):
    """Binarized fully connected layer"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization for better training stability
        self.bn = nn.BatchNorm1d(out_features, momentum=0.1, affine=False)
    
    def forward(self, x):
        # Binarize input activations
        if self.training:
            # During training, use straight-through estimator
            binary_input = binarize(x)
        else:
            # During inference, hard binarization
            binary_input = torch.sign(x)
        
        # Binarize weights
        if self.training:
            binary_weight = binarize(self.weight)
        else:
            binary_weight = torch.sign(self.weight)
        
        # Linear transformation with binarized weights and inputs
        output = F.linear(binary_input, binary_weight, self.bias)
        
        # Apply batch normalization
        if output.dim() > 1:
            output = self.bn(output)
        
        return output

class BinarizedLSTMCell(nn.Module):
    """Binarized LSTM Cell with binary weights but full precision states"""
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(BinarizedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden weights (will be binarized)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.1)
        # Hidden-to-hidden weights (will be binarized)  
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.1)
        
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(4 * hidden_size)
    
    def forward(self, input, hidden):
        hx, cx = hidden
        
        # Binarize weights
        if self.training:
            binary_weight_ih = binarize(self.weight_ih)
            binary_weight_hh = binarize(self.weight_hh)
        else:
            binary_weight_ih = torch.sign(self.weight_ih)
            binary_weight_hh = torch.sign(self.weight_hh)
        
        # Linear transformations with binary weights
        gi = F.linear(input, binary_weight_ih, self.bias_ih)
        gh = F.linear(hx, binary_weight_hh, self.bias_hh)
        i_r, i_i, i_n, i_c = gi.chunk(4, 1)
        h_r, h_i, h_n, h_c = gh.chunk(4, 1)
        
        # Apply layer normalization
        gates = self.layer_norm(gi + gh)
        resetgate, inputgate, newgate, cellgate = gates.chunk(4, 1)
        
        # LSTM computations (keep states in full precision)
        resetgate = torch.sigmoid(resetgate)
        inputgate = torch.sigmoid(inputgate)
        newgate = torch.tanh(newgate)
        cellgate = torch.sigmoid(cellgate)
        
        cy = (cellgate * cx) + (inputgate * newgate)
        hy = resetgate * torch.tanh(cy)
        
        return hy, cy

class BinarizedLSTM(nn.Module):
    """Binarized LSTM layer"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(BinarizedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        num_directions = 2 if bidirectional else 1
        
        self.lstm_cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                self.lstm_cells.append(BinarizedLSTMCell(layer_input_size, hidden_size, bias))
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, input, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        seq_len = input.size(1) if self.batch_first else input.size(0)
        
        if not self.batch_first:
            input = input.transpose(0, 1)  # Convert to batch_first
        
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = (torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device),
                  torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=input.device))
        
        h, c = hx
        outputs = []
        
        for t in range(seq_len):
            x = input[:, t, :]
            new_h = []
            new_c = []
            
            for layer in range(self.num_layers):
                if self.bidirectional:
                    # Forward direction
                    idx = layer * 2
                    h_forward, c_forward = self.lstm_cells[idx](x, (h[idx], c[idx]))
                    
                    # Backward direction (process in reverse)
                    idx = layer * 2 + 1
                    x_backward = input[:, seq_len - 1 - t, :] if layer == 0 else x
                    h_backward, c_backward = self.lstm_cells[idx](x_backward, (h[idx], c[idx]))
                    
                    x = torch.cat([h_forward, h_backward], dim=1)
                    new_h.extend([h_forward, h_backward])
                    new_c.extend([c_forward, c_backward])
                else:
                    h_new, c_new = self.lstm_cells[layer](x, (h[layer], c[layer]))
                    x = h_new
                    new_h.append(h_new)
                    new_c.append(c_new)
                
                if self.dropout_layer and layer < self.num_layers - 1:
                    x = self.dropout_layer(x)
            
            outputs.append(x)
            h = torch.stack(new_h, dim=0)
            c = torch.stack(new_c, dim=0)
        
        output = torch.stack(outputs, dim=1)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h, c)

class HybridSequenceModel(nn.Module):
    """Hybrid model combining CNN, LSTM, BNN, and ANN layers for sequence data"""
    
    def __init__(self, input_size, hidden_sizes, lstm_hidden_size, num_classes,
                 sequence_length, use_cnn=True, use_bnn_lstm=False, 
                 bnn_layers=None, dropout_rate=0.2, num_lstm_layers=2):
        super(HybridSequenceModel, self).__init__()
        
        self.use_cnn = use_cnn
        self.use_bnn_lstm = use_bnn_lstm
        self.sequence_length = sequence_length
        
        if bnn_layers is None:
            bnn_layers = []
        
        # CNN feature extractor (optional)
        if use_cnn:
            self.cnn_features = nn.Sequential(
                nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(sequence_length)
            )
            lstm_input_size = 128
        else:
            self.cnn_features = None
            lstm_input_size = input_size
        
        # LSTM layer (regular or binarized)
        if use_bnn_lstm:
            self.lstm = BinarizedLSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
                bidirectional=True
            )
            classifier_input_size = lstm_hidden_size * 2  # Bidirectional
        else:
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
                bidirectional=True
            )
            classifier_input_size = lstm_hidden_size * 2  # Bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.Tanh(),
            nn.Linear(classifier_input_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Hybrid classifier layers
        self.classifier_layers = nn.ModuleList()
        self.classifier_activations = nn.ModuleList()
        self.classifier_dropouts = nn.ModuleList()
        
        prev_size = classifier_input_size
        all_sizes = [classifier_input_size] + hidden_sizes + [num_classes]
        
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            
            # Decide whether to use BNN or ANN layer
            if i in bnn_layers:
                layer = BinarizedLinear(prev_size, current_size)
                activation = nn.Hardtanh(inplace=True)
            else:
                layer = nn.Linear(prev_size, current_size)
                activation = nn.ReLU(inplace=True)
            
            self.classifier_layers.append(layer)
            self.classifier_activations.append(activation)
            
            if i < len(all_sizes) - 2:
                self.classifier_dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.classifier_dropouts.append(nn.Identity())
            
            prev_size = current_size
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        if self.use_cnn:
            # x should be (batch, sequence, features)
            x = x.transpose(1, 2)  # Convert to (batch, features, sequence)
            x = self.cnn_features(x)
            x = x.transpose(1, 2)  # Convert back to (batch, sequence, features)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Classifier
        x = attended_output
        for i, (layer, activation, dropout) in enumerate(zip(
            self.classifier_layers, self.classifier_activations, self.classifier_dropouts)):
            
            x = layer(x)
            
            if i < len(self.classifier_layers) - 1:
                x = activation(x)
                x = dropout(x)
        
        return x

class HybridTimeSeries(nn.Module):
    """Hybrid model specifically designed for time series prediction"""
    
    def __init__(self, input_features, lstm_hidden_size, output_features, 
                 sequence_length, forecast_horizon=1, use_bnn_lstm=False, 
                 num_lstm_layers=2, dropout_rate=0.1):
        super(HybridTimeSeries, self).__init__()
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Feature extraction with hybrid conv layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Add binarized conv layer for efficiency
        self.bnn_conv = BinarizedConv1d(32, 64, kernel_size=3, padding=1)
        
        # LSTM layers
        if use_bnn_lstm:
            self.lstm = BinarizedLSTM(
                input_size=64,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        else:
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout_rate
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, output_features * forecast_horizon)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = self.feature_extractor(x)
        x = self.bnn_conv(x)
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]
        
        # Generate predictions
        output = self.output_projection(last_output)
        
        # Reshape to (batch, forecast_horizon, features)
        output = output.view(batch_size, self.forecast_horizon, -1)
        
        return output

class BinarizedConv1d(nn.Module):
    """Binarized 1D convolutional layer for time series"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(BinarizedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.1, affine=False)
    
    def forward(self, x):
        if self.training:
            binary_input = binarize(x)
            binary_weight = binarize(self.weight)
        else:
            binary_input = torch.sign(x)
            binary_weight = torch.sign(self.weight)
        
        output = F.conv1d(binary_input, binary_weight, self.bias,
                         self.stride, self.padding)
        output = self.bn(output)
        return output

class BinarizedConv2d(nn.Module):
    """Binarized convolutional layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(BinarizedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 
                                              kernel_size, kernel_size) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=False)
    
    def forward(self, x):
        # Binarize input activations
        if self.training:
            binary_input = binarize(x)
        else:
            binary_input = torch.sign(x)
        
        # Binarize weights
        if self.training:
            binary_weight = binarize(self.weight)
        else:
            binary_weight = torch.sign(self.weight)
        
        # Convolution with binarized weights and inputs
        output = F.conv2d(binary_input, binary_weight, self.bias,
                         self.stride, self.padding)
        
        # Apply batch normalization
        output = self.bn(output)
        
        return output

class HybridANNBNN(nn.Module):
    """Hybrid Neural Network combining ANN and BNN layers"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, 
                 bnn_layers=None, dropout_rate=0.2):
        super(HybridANNBNN, self).__init__()
        
        # If bnn_layers not specified, binarize middle layers
        if bnn_layers is None:
            bnn_layers = list(range(1, len(hidden_sizes)))
        
        self.bnn_layers = bnn_layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Input layer dimensions
        prev_size = input_size
        all_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # Build layers
        for i in range(len(all_sizes) - 1):
            current_size = all_sizes[i + 1]
            
            # Decide whether to use BNN or ANN layer
            if i in bnn_layers:
                # Use binarized layer
                layer = BinarizedLinear(prev_size, current_size)
                # Binarized layers use hard tanh activation
                activation = nn.Hardtanh(inplace=True)
            else:
                # Use full precision layer
                layer = nn.Linear(prev_size, current_size)
                # Regular layers use ReLU
                activation = nn.ReLU(inplace=True)
            
            self.layers.append(layer)
            self.activations.append(activation)
            
            # Add dropout (except for output layer)
            if i < len(all_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
            
            prev_size = current_size
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass through all layers
        for i, (layer, activation, dropout) in enumerate(zip(
            self.layers, self.activations, self.dropouts)):
            
            x = layer(x)
            
            # Don't apply activation to output layer
            if i < len(self.layers) - 1:
                x = activation(x)
                x = dropout(x)
        
        return x

class HybridCNN(nn.Module):
    """Hybrid CNN combining ANN and BNN convolutional layers"""
    
    def __init__(self, input_channels=3, num_classes=10, bnn_conv_layers=None):
        super(HybridCNN, self).__init__()
        
        if bnn_conv_layers is None:
            bnn_conv_layers = [1, 2]  # Binarize middle conv layers
        
        self.bnn_conv_layers = bnn_conv_layers
        
        # Define architecture
        self.features = nn.ModuleList()
        
        # Layer 0: Regular conv (preserve input precision)
        if 0 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(input_channels, 32, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(input_channels, 32, 3, padding=1))
            self.features.append(nn.BatchNorm2d(32))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(2, 2))
        
        # Layer 1: Potentially binarized
        if 1 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(32, 64, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(32, 64, 3, padding=1))
            self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(2, 2))
        
        # Layer 2: Potentially binarized
        if 2 in bnn_conv_layers:
            self.features.append(BinarizedConv2d(64, 128, 3, padding=1))
        else:
            self.features.append(nn.Conv2d(64, 128, 3, padding=1))
            self.features.append(nn.BatchNorm2d(128))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        # Classifier (keep full precision for final layer)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Example usage and training utilities
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_model_size(model):
    """Estimate model size in MB"""
    total_size = 0
    for name, param in model.named_parameters():
        # Binarized weights effectively use 1 bit per weight
        if any(bnn_layer in name for bnn_layer in ['BinarizedLinear', 'BinarizedConv2d']):
            # For binarized layers, weights are effectively 1-bit
            size = param.numel() * 0.125 / (1024 * 1024)  # 1 bit per parameter
        else:
            # Regular layers use 32-bit floats
            size = param.numel() * 4 / (1024 * 1024)  # 4 bytes per parameter
        total_size += size
    return total_size

# Example models
if __name__ == "__main__":
    # Example 1: Hybrid fully connected network
    hybrid_fc = HybridANNBNN(
        input_size=784,  # 28x28 for MNIST
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        bnn_layers=[1],  # Only middle layer is binarized
        dropout_rate=0.3
    )
    
    # Example 2: Hybrid CNN
    hybrid_cnn = HybridCNN(
        input_channels=3,  # RGB images
        num_classes=10,
        bnn_conv_layers=[1, 2]  # Middle conv layers are binarized
    )
    
    # Example 3: Hybrid Sequence Model (CNN + LSTM + BNN + ANN)
    hybrid_sequence = HybridSequenceModel(
        input_size=50,  # Feature dimension
        hidden_sizes=[256, 128],  # Classifier hidden layers
        lstm_hidden_size=128,
        num_classes=10,
        sequence_length=100,
        use_cnn=True,
        use_bnn_lstm=True,  # Use binarized LSTM
        bnn_layers=[0],  # First classifier layer is binarized
        num_lstm_layers=2
    )
    
    # Example 4: Time Series Prediction Model
    hybrid_timeseries = HybridTimeSeries(
        input_features=5,  # Number of input features per time step
        lstm_hidden_size=64,
        output_features=1,  # Predicting 1 feature
        sequence_length=50,
        forecast_horizon=10,  # Predict 10 steps ahead
        use_bnn_lstm=False,  # Use regular LSTM for better accuracy
        num_lstm_layers=2
    )
    
    # Example 5: Pure Binarized LSTM
    bnn_lstm = BinarizedLSTM(
        input_size=20,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
        bidirectional=True
    )
    
    # Test with dummy data
    dummy_input_fc = torch.randn(32, 784)  # Batch of 32, flattened MNIST
    dummy_input_cnn = torch.randn(32, 3, 32, 32)  # Batch of 32, 32x32 RGB
    dummy_input_seq = torch.randn(32, 100, 50)  # Batch of 32, sequence of 100, 50 features
    dummy_input_ts = torch.randn(32, 50, 5)  # Batch of 32, 50 time steps, 5 features
    dummy_input_lstm = torch.randn(32, 25, 20)  # Batch of 32, 25 time steps, 20 features
    
    print("=== HYBRID ANN-BNN-LSTM NEURAL NETWORKS ===\n")
    
    print("1. Hybrid FC Network:")
    print(f"   Output shape: {hybrid_fc(dummy_input_fc).shape}")
    total, trainable = count_parameters(hybrid_fc)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_fc):.2f} MB")
    
    print("\n2. Hybrid CNN:")
    print(f"   Output shape: {hybrid_cnn(dummy_input_cnn).shape}")
    total, trainable = count_parameters(hybrid_cnn)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_cnn):.2f} MB")
    
    print("\n3. Hybrid Sequence Model (CNN + LSTM + BNN):")
    print(f"   Output shape: {hybrid_sequence(dummy_input_seq).shape}")
    total, trainable = count_parameters(hybrid_sequence)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_sequence):.2f} MB")
    
    print("\n4. Hybrid Time Series Model:")
    print(f"   Output shape: {hybrid_timeseries(dummy_input_ts).shape}")
    total, trainable = count_parameters(hybrid_timeseries)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(hybrid_timeseries):.2f} MB")
    
    print("\n5. Binarized LSTM:")
    lstm_output, (h, c) = bnn_lstm(dummy_input_lstm)
    print(f"   Output shape: {lstm_output.shape}")
    print(f"   Hidden state shape: {h.shape}")
    print(f"   Cell state shape: {c.shape}")
    total, trainable = count_parameters(bnn_lstm)
    print(f"   Parameters: {total:,} total, {trainable:,} trainable")
    print(f"   Estimated size: {estimate_model_size(bnn_lstm):.2f} MB")
    
    print("\n=== HYBRID ARCHITECTURE BENEFITS ===")
    print("✓ ANN Layers: Full precision for critical computations")
    print("✓ BNN Layers: 32x memory reduction, faster inference")
    print("✓ LSTM Layers: Sequential pattern learning and memory")
    print("✓ CNN Layers: Local feature extraction")
    print("✓ Attention: Focus on important sequence elements")
    print("✓ Flexible: Configure which layers to binarize")
    
    print("\n=== USE CASES ===")
    print("• Natural Language Processing (text classification, sentiment)")
    print("• Time Series Forecasting (stock prices, sensor data)")
    print("• Video Analysis (action recognition, object tracking)")
    print("• Audio Processing (speech recognition, music classification)")
    print("• Sensor Data Analysis (IoT, medical signals)")
    print("• Financial Modeling (algorithmic trading, risk assessment)")
    
    print("\n=== MEMORY EFFICIENCY COMPARISON ===")
    
    # Create comparable models for comparison
    full_precision_lstm = nn.LSTM(20, 64, 2, batch_first=True, bidirectional=True)
    total_fp, _ = count_parameters(full_precision_lstm)
    total_bnn, _ = count_parameters(bnn_lstm)
    
    print(f"Full Precision LSTM: {estimate_model_size(full_precision_lstm):.2f} MB")
    print(f"Binarized LSTM: {estimate_model_size(bnn_lstm):.2f} MB")
    print(f"Memory Reduction: {estimate_model_size(full_precision_lstm) / estimate_model_size(bnn_lstm):.1f}x smaller")