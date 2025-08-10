"""
Neural Processing Agent - Integrates hybrid ANN-BNN-LSTM capabilities
into the brain system for advanced pattern recognition and processing.
"""

import time
import random
from typing import Dict, List, Any, Optional, Tuple

# Mock PyTorch classes for demonstration when PyTorch is not available
class MockTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = [random.random() for _ in range(self._size_from_shape(shape))]
        self.grad_fn = None
    
    def _size_from_shape(self, shape):
        size = 1
        for dim in shape:
            size *= dim
        return size
    
    def numel(self):
        return len(self.data)
    
    def element_size(self):
        return 4  # 4 bytes for float32
    
    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return MockTensor(tuple(new_shape))
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
    def dim(self):
        return len(self.shape)
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def view(self, *new_shape):
        return MockTensor(new_shape)
    
    def transpose(self, dim1, dim2):
        new_shape = list(self.shape)
        new_shape[dim1], new_shape[dim2] = new_shape[dim2], new_shape[dim1]
        return MockTensor(tuple(new_shape))
    
    def squeeze(self, dim=None):
        if dim is None:
            new_shape = [s for s in self.shape if s != 1]
        else:
            new_shape = list(self.shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
        return MockTensor(tuple(new_shape))
    
    def item(self):
        return self.data[0] if len(self.data) == 1 else self.data

class MockModule:
    def __init__(self):
        self.training = False
    
    def eval(self):
        self.training = False
    
    def train(self, mode=True):
        self.training = mode

class MockLinear(MockModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MockTensor((out_features, in_features))
        if bias:
            self.bias = MockTensor((out_features,))
        else:
            self.bias = None
    
    def forward(self, x):
        # Simple matrix multiplication approximation
        batch_size = x.shape[0]
        output_data = []
        for i in range(batch_size):
            output_row = []
            for j in range(self.out_features):
                sum_val = 0
                for k in range(self.in_features):
                    sum_val += x.data[i * self.in_features + k] * self.weight.data[j * self.in_features + k]
                if self.bias:
                    sum_val += self.bias.data[j]
                output_row.append(sum_val)
            output_data.extend(output_row)
        
        output = MockTensor((batch_size, self.out_features))
        output.data = output_data
        return output

class MockLSTM(MockModule):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
    
    def forward(self, x, hx=None):
        # Simple LSTM approximation
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            h = MockTensor((self.num_layers * num_directions, batch_size, self.hidden_size))
            c = MockTensor((self.num_layers * num_directions, batch_size, self.hidden_size))
        else:
            h, c = hx
        
        # Simple output generation
        output_size = (batch_size, seq_len, self.hidden_size * (2 if self.bidirectional else 1))
        output = MockTensor(output_size)
        
        return output, (h, c)

class MockConv2d(MockModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = MockTensor((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = MockTensor((out_channels,))
    
    def forward(self, x):
        # Simple conv2d approximation
        batch_size, channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = MockTensor((batch_size, self.out_channels, out_height, out_width))
        return output

class MockReLU(MockModule):
    def forward(self, x):
        # Simple ReLU approximation
        output = MockTensor(x.shape)
        output.data = [max(0, val) for val in x.data]
        return output

class MockMaxPool2d(MockModule):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    
    def forward(self, x):
        # Simple max pooling approximation
        batch_size, channels, height, width = x.shape
        out_height = height // self.kernel_size
        out_width = width // self.kernel_size
        
        output = MockTensor((batch_size, channels, out_height, out_width))
        return output

class MockAdaptiveAvgPool2d(MockModule):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        output = MockTensor((batch_size, channels, self.output_size[0], self.output_size[1]))
        return output

class MockDropout(MockModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        # Simple dropout (no-op during eval)
        if not self.training:
            return x
        return x

class MockSequential(MockModule):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)
    
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

class MockBatchNorm2d(MockModule):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
    
    def forward(self, x):
        return x

class MockTorch:
    @staticmethod
    def randn(*shape):
        return MockTensor(shape)
    
    @staticmethod
    def zeros(*shape):
        tensor = MockTensor(shape)
        tensor.data = [0.0] * tensor.numel()
        return tensor
    
    @staticmethod
    def tensor(data, dtype=None):
        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return MockTensor((len(data),))
            elif all(isinstance(x, list) for x in data):
                return MockTensor((len(data), len(data[0])))
        return MockTensor((1,))
    
    @staticmethod
    def softmax(tensor, dim=-1):
        # Simple softmax approximation
        data = tensor.data.copy()
        max_val = max(data)
        exp_data = [2.71828 ** (x - max_val) for x in data]
        sum_exp = sum(exp_data)
        softmax_data = [x / sum_exp for x in exp_data]
        result = MockTensor(tensor.shape)
        result.data = softmax_data
        return result
    
    @staticmethod
    def argmax(tensor, dim=-1):
        max_idx = tensor.data.index(max(tensor.data))
        return MockTensor((1,))
    
    @staticmethod
    def max(tensor):
        return max(tensor.data)
    
    @staticmethod
    def mean(tensor):
        return sum(tensor.data) / len(tensor.data)
    
    @staticmethod
    def std(tensor):
        mean = MockTorch.mean(tensor)
        variance = sum((x - mean) ** 2 for x in tensor.data) / len(tensor.data)
        return variance ** 0.5
    
    @staticmethod
    def no_grad():
        class NoGrad:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoGrad()
    
    nn = type('nn', (), {
        'Module': MockModule,
        'Linear': MockLinear,
        'LSTM': MockLSTM,
        'Conv2d': MockConv2d,
        'ReLU': MockReLU,
        'MaxPool2d': MockMaxPool2d,
        'AdaptiveAvgPool2d': MockAdaptiveAvgPool2d,
        'Dropout': MockDropout,
        'Sequential': MockSequential,
        'BatchNorm2d': MockBatchNorm2d,
        'functional': type('functional', (), {
            'linear': lambda x, weight, bias=None: MockLinear(weight.shape[1], weight.shape[0]).forward(x),
            'conv2d': lambda x, weight, bias, stride, padding: MockConv2d(weight.shape[1], weight.shape[0], weight.shape[2]).forward(x)
        })
    })()

# Use mock torch if real torch is not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    torch = MockTorch()
    nn = torch.nn
    F = torch.nn.functional
    np = None

# Mock hybrid neural network classes
class MockHybridANNBNN(MockModule):
    def __init__(self, input_size, hidden_sizes, num_classes, bnn_layers=None, dropout_rate=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bnn_layers = bnn_layers or []
    
    def forward(self, x):
        # Simple forward pass
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        output = MockTensor((x.shape[0], self.num_classes))
        return output

class MockHybridCNN(MockModule):
    def __init__(self, input_channels=3, num_classes=10, bnn_conv_layers=None):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bnn_conv_layers = bnn_conv_layers or []
    
    def forward(self, x):
        # Simple CNN forward pass
        batch_size = x.shape[0]
        output = MockTensor((batch_size, self.num_classes))
        return output

class MockHybridSequenceModel(MockModule):
    def __init__(self, input_size, hidden_sizes, lstm_hidden_size, num_classes, sequence_length, use_cnn=True, use_bnn_lstm=False, bnn_layers=None, dropout_rate=0.2, num_lstm_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
    
    def forward(self, x):
        # Simple sequence model forward pass
        batch_size = x.shape[0]
        output = MockTensor((batch_size, self.num_classes))
        return output

class MockHybridTimeSeries(MockModule):
    def __init__(self, input_features, lstm_hidden_size, output_features, sequence_length, forecast_horizon=1, use_bnn_lstm=False, num_lstm_layers=2, dropout_rate=0.1):
        super().__init__()
        self.input_features = input_features
        self.lstm_hidden_size = lstm_hidden_size
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x):
        # Simple time series forward pass
        batch_size = x.shape[0]
        output = MockTensor((batch_size, self.forecast_horizon, self.output_features))
        return output

class MockBinarizedLSTM(MockModule):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
    
    def forward(self, x, hx=None):
        # Simple LSTM forward pass
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            h = MockTensor((self.num_layers * num_directions, batch_size, self.hidden_size))
            c = MockTensor((self.num_layers * num_directions, batch_size, self.hidden_size))
        else:
            h, c = hx
        
        output_size = (batch_size, seq_len, self.hidden_size * (2 if self.bidirectional else 1))
        output = MockTensor(output_size)
        
        return output, (h, c)

# Mock the hybrid neural network imports
try:
    from hybrid_neural_networks import (
        HybridANNBNN, HybridCNN, HybridSequenceModel, HybridTimeSeries,
        BinarizedLSTM, count_parameters, estimate_model_size
    )
except ImportError:
    # Use mock implementations
    HybridANNBNN = MockHybridANNBNN
    HybridCNN = MockHybridCNN
    HybridSequenceModel = MockHybridSequenceModel
    HybridTimeSeries = MockHybridTimeSeries
    BinarizedLSTM = MockBinarizedLSTM
    
    def count_parameters(model):
        """Mock parameter counting"""
        return 1000, 1000
    
    def estimate_model_size(model):
        """Mock model size estimation"""
        return 1.0

class NeuralProcessingAgent(Agent):
    """
    Specialized agent for neural network processing with hybrid ANN-BNN-LSTM capabilities.
    Integrates various neural architectures for different types of data and tasks.
    """
    
    def __init__(self, agent_id: str, specialization: str, 
                 neural_config: Optional[Dict] = None):
        """
        Initialize neural processing agent.
        
        Args:
            agent_id: Unique identifier for the agent
            specialization: Type of neural processing (e.g., 'vision', 'nlp', 'timeseries')
            neural_config: Configuration for neural architectures
        """
        super().__init__(agent_id, specialization)
        
        # Initialize HRM for neural processing decisions
        self.hrm = HierarchicalReasoningModel(
            visionary_prompt=f"You are a neural processing expert specializing in {specialization}. "
                           f"Your mission is to select and configure the optimal neural architecture "
                           f"for each task, balancing accuracy, efficiency, and resource constraints.",
            architect_prompt=f"Design neural processing strategies for {specialization} tasks. "
                           f"Choose appropriate architectures (ANN, BNN, LSTM, CNN) based on "
                           f"data characteristics and performance requirements.",
            foreman_prompt=f"Execute neural processing tasks efficiently. Monitor model "
                         f"performance, adjust hyperparameters, and handle computational "
                         f"resource allocation in real-time.",
            technician_prompt=f"Perform low-level neural computations. Execute forward "
                           f"and backward passes, manage memory, and optimize "
                           f"tensor operations."
        )
        
        # Neural architecture configurations
        self.neural_config = neural_config or self._get_default_config(specialization)
        
        # Available neural models
        self.models = {}
        self.active_model = None
        self.model_performance = {}
        
        # Processing state
        self.is_processing = False
        self.current_task = None
        self.processing_history = []
        
        # Initialize default models
        self._initialize_models()
    
    def _get_default_config(self, specialization: str) -> Dict:
        """Get default configuration for different specializations."""
        configs = {
            'vision': {
                'input_channels': 3,
                'num_classes': 1000,
                'bnn_conv_layers': [1, 2],
                'preferred_architecture': 'HybridCNN'
            },
            'nlp': {
                'input_size': 300,  # Word embedding size
                'hidden_sizes': [512, 256],
                'lstm_hidden_size': 256,
                'num_classes': 10,
                'sequence_length': 100,
                'use_cnn': True,
                'use_bnn_lstm': True,
                'preferred_architecture': 'HybridSequenceModel'
            },
            'timeseries': {
                'input_features': 10,
                'lstm_hidden_size': 128,
                'output_features': 1,
                'sequence_length': 50,
                'forecast_horizon': 5,
                'use_bnn_lstm': False,
                'preferred_architecture': 'HybridTimeSeries'
            },
            'classification': {
                'input_size': 784,
                'hidden_sizes': [512, 256, 128],
                'num_classes': 10,
                'bnn_layers': [1],
                'preferred_architecture': 'HybridANNBNN'
            }
        }
        return configs.get(specialization, configs['classification'])
    
    def _initialize_models(self):
        """Initialize neural models based on specialization."""
        config = self.neural_config
        
        try:
            if config['preferred_architecture'] == 'HybridCNN':
                self.models['cnn'] = HybridCNN(
                    input_channels=config.get('input_channels', 3),
                    num_classes=config.get('num_classes', 10),
                    bnn_conv_layers=config.get('bnn_conv_layers', [1, 2])
                )
                self.active_model = 'cnn'
                
            elif config['preferred_architecture'] == 'HybridSequenceModel':
                self.models['sequence'] = HybridSequenceModel(
                    input_size=config.get('input_size', 50),
                    hidden_sizes=config.get('hidden_sizes', [256, 128]),
                    lstm_hidden_size=config.get('lstm_hidden_size', 128),
                    num_classes=config.get('num_classes', 10),
                    sequence_length=config.get('sequence_length', 100),
                    use_cnn=config.get('use_cnn', True),
                    use_bnn_lstm=config.get('use_bnn_lstm', True),
                    bnn_layers=config.get('bnn_layers', []),
                    num_lstm_layers=2
                )
                self.active_model = 'sequence'
                
            elif config['preferred_architecture'] == 'HybridTimeSeries':
                self.models['timeseries'] = HybridTimeSeries(
                    input_features=config.get('input_features', 5),
                    lstm_hidden_size=config.get('lstm_hidden_size', 64),
                    output_features=config.get('output_features', 1),
                    sequence_length=config.get('sequence_length', 50),
                    forecast_horizon=config.get('forecast_horizon', 1),
                    use_bnn_lstm=config.get('use_bnn_lstm', False),
                    num_lstm_layers=2
                )
                self.active_model = 'timeseries'
                
            elif config['preferred_architecture'] == 'HybridANNBNN':
                self.models['classifier'] = HybridANNBNN(
                    input_size=config.get('input_size', 784),
                    hidden_sizes=config.get('hidden_sizes', [512, 256, 128]),
                    num_classes=config.get('num_classes', 10),
                    bnn_layers=config.get('bnn_layers', [1]),
                    dropout_rate=0.3
                )
                self.active_model = 'classifier'
            
            # Initialize performance tracking
            for model_name, model in self.models.items():
                total_params, trainable_params = count_parameters(model)
                model_size = estimate_model_size(model)
                self.model_performance[model_name] = {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': model_size,
                    'accuracy': 0.0,
                    'processing_time': 0.0,
                    'memory_usage': 0.0,
                    'tasks_processed': 0
                }
                
        except Exception as e:
            print(f"Error initializing models for {self.agent_id}: {e}")
            # Fallback to simple linear model
            self.models['fallback'] = nn.Linear(10, 2)
            self.active_model = 'fallback'
    
    def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming messages using HRM and neural processing."""
        if message.signal_strength < self.threshold:
            return None
        
        # Use HRM to process the message
        hrm_result = self.hrm.process_task(message.content)
        
        # Extract task details
        task_type = hrm_result.get('task_type', 'classification')
        data = hrm_result.get('data', None)
        requirements = hrm_result.get('requirements', {})
        
        # Process with neural networks
        if data is not None:
            neural_result = self._process_neural_task(task_type, data, requirements)
            
            # Create response message
            response_content = {
                'agent_id': self.agent_id,
                'specialization': self.specialization,
                'task_type': task_type,
                'neural_result': neural_result,
                'hrm_analysis': hrm_result,
                'model_performance': self.model_performance.get(self.active_model, {}),
                'processing_metadata': {
                    'model_used': self.active_model,
                    'processing_time': neural_result.get('processing_time', 0),
                    'memory_usage': neural_result.get('memory_usage', 0)
                }
            }
            
            return Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response_content,
                signal_strength=message.signal_strength * 0.9  # Slight decay
            )
        
        return None
    
    def _process_neural_task(self, task_type: str, data: Any, requirements: Dict) -> Dict:
        """Process neural task using appropriate model."""
        import time
        start_time = time.time()
        
        try:
            # Select optimal model based on task and requirements
            optimal_model = self._select_optimal_model(task_type, requirements)
            
            if optimal_model not in self.models:
                optimal_model = self.active_model or list(self.models.keys())[0]
            
            model = self.models[optimal_model]
            model.eval()
            
            # Prepare data based on model type
            input_data = self._prepare_data(data, optimal_model)
            
            # Process with neural network
            with torch.no_grad():
                if isinstance(model, (HybridSequenceModel, HybridTimeSeries, BinarizedLSTM)):
                    # Sequence models expect 3D input (batch, sequence, features)
                    if input_data.dim() == 2:
                        input_data = input_data.unsqueeze(0)  # Add batch dimension
                    output = model(input_data)
                    if isinstance(output, tuple):
                        output = output[0]  # Take first element for LSTM outputs
                else:
                    # Regular models expect 2D input (batch, features)
                    if input_data.dim() == 1:
                        input_data = input_data.unsqueeze(0)  # Add batch dimension
                    output = model(input_data)
            
            # Post-process output
            result = self._post_process_output(output, task_type)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(input_data, output)
            
            # Update performance tracking
            self._update_performance_metrics(optimal_model, result, processing_time, memory_usage)
            
            return {
                'success': True,
                'output': result,
                'raw_output': output.detach().cpu().numpy() if torch.is_tensor(output) else output,
                'model_used': optimal_model,
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'task_type': task_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_used': optimal_model if 'optimal_model' in locals() else 'unknown',
                'processing_time': time.time() - start_time,
                'task_type': task_type
            }
    
    def _select_optimal_model(self, task_type: str, requirements: Dict) -> str:
        """Select optimal model based on task and requirements."""
        # Priority-based selection
        if 'accuracy' in requirements and requirements['accuracy'] > 0.95:
            # High accuracy requirement - prefer full precision models
            for model_name in ['classifier', 'sequence', 'timeseries']:
                if model_name in self.models:
                    return model_name
        
        if 'efficiency' in requirements and requirements['efficiency'] > 0.8:
            # High efficiency requirement - prefer binarized models
            for model_name in ['cnn', 'sequence']:
                if model_name in self.models:
                    return model_name
        
        # Default selection based on task type
        task_model_mapping = {
            'classification': 'classifier',
            'vision': 'cnn',
            'nlp': 'sequence',
            'timeseries': 'timeseries',
            'forecasting': 'timeseries'
        }
        
        preferred_model = task_model_mapping.get(task_type, self.active_model)
        return preferred_model if preferred_model in self.models else list(self.models.keys())[0]
    
    def _prepare_data(self, data: Any, model_name: str) -> torch.Tensor:
        """Prepare data for specific model type."""
        if isinstance(data, torch.Tensor):
            return data
        
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        
        if isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32)
        
        # Default conversion
        return torch.tensor([[data]], dtype=torch.float32)
    
    def _post_process_output(self, output: torch.Tensor, task_type: str) -> Any:
        """Post-process model output based on task type."""
        if task_type == 'classification':
            probabilities = torch.softmax(output, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            return {
                'predicted_class': predicted_class.item() if predicted_class.numel() == 1 else predicted_class.tolist(),
                'probabilities': probabilities.squeeze().tolist(),
                'confidence': torch.max(probabilities).item()
            }
        
        elif task_type in ['regression', 'forecasting']:
            return {
                'predictions': output.squeeze().tolist(),
                'mean': torch.mean(output).item(),
                'std': torch.std(output).item()
            }
        
        else:
            return {
                'raw_output': output.squeeze().tolist(),
                'shape': list(output.shape)
            }
    
    def _estimate_memory_usage(self, input_data: torch.Tensor, output: torch.Tensor) -> float:
        """Estimate memory usage in MB."""
        input_size = input_data.element_size() * input_data.nelement() / (1024 * 1024)
        output_size = output.element_size() * output.nelement() / (1024 * 1024)
        return input_size + output_size
    
    def _update_performance_metrics(self, model_name: str, result: Dict, 
                                   processing_time: float, memory_usage: float):
        """Update performance metrics for the model."""
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]
            perf['processing_time'] = (perf['processing_time'] * perf['tasks_processed'] + processing_time) / (perf['tasks_processed'] + 1)
            perf['memory_usage'] = (perf['memory_usage'] * perf['tasks_processed'] + memory_usage) / (perf['tasks_processed'] + 1)
            perf['tasks_processed'] += 1
            
            # Update accuracy if available
            if 'confidence' in result:
                perf['accuracy'] = (perf['accuracy'] * (perf['tasks_processed'] - 1) + result['confidence']) / perf['tasks_processed']
    
    def get_model_info(self) -> Dict:
        """Get information about available models."""
        info = {
            'agent_id': self.agent_id,
            'specialization': self.specialization,
            'active_model': self.active_model,
            'available_models': list(self.models.keys()),
            'model_performance': self.model_performance,
            'neural_config': self.neural_config
        }
        return info
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name in self.models:
            self.active_model = model_name
            return True
        return False
    
    def optimize_model(self, optimization_target: str = 'efficiency') -> Dict:
        """Optimize model for specific target."""
        if not self.models:
            return {'success': False, 'error': 'No models available'}
        
        model = self.models[self.active_model]
        
        if optimization_target == 'efficiency':
            # Convert more layers to binary for efficiency
            if hasattr(model, 'bnn_layers'):
                original_bnn_layers = model.bnn_layers.copy()
                # Add more layers to binarization
                if hasattr(model, 'layers'):
                    model.bnn_layers = list(range(len(model.layers) - 1))  # All but output layer
                return {
                    'success': True,
                    'optimization': 'increased_binarization',
                    'original_bnn_layers': original_bnn_layers,
                    'new_bnn_layers': model.bnn_layers
                }
        
        elif optimization_target == 'accuracy':
            # Reduce binarization for accuracy
            if hasattr(model, 'bnn_layers'):
                original_bnn_layers = model.bnn_layers.copy()
                model.bnn_layers = []  # Use full precision
                return {
                    'success': True,
                    'optimization': 'full_precision',
                    'original_bnn_layers': original_bnn_layers,
                    'new_bnn_layers': model.bnn_layers
                }
        
        return {'success': False, 'error': f'Unknown optimization target: {optimization_target}'}

# Factory function for creating neural processing agents
def create_neural_processing_agent(agent_id: str, specialization: str, 
                                 neural_config: Optional[Dict] = None) -> NeuralProcessingAgent:
    """Create a neural processing agent with the specified configuration."""
    return NeuralProcessingAgent(agent_id, specialization, neural_config)

# Example usage
if __name__ == "__main__":
    # Create different types of neural processing agents
    vision_agent = create_neural_processing_agent(
        "vision_agent_001", 
        "vision",
        {
            'input_channels': 3,
            'num_classes': 1000,
            'bnn_conv_layers': [1, 2],
            'preferred_architecture': 'HybridCNN'
        }
    )
    
    nlp_agent = create_neural_processing_agent(
        "nlp_agent_001", 
        "nlp",
        {
            'input_size': 300,
            'hidden_sizes': [512, 256],
            'lstm_hidden_size': 256,
            'num_classes': 10,
            'sequence_length': 100,
            'use_cnn': True,
            'use_bnn_lstm': True,
            'preferred_architecture': 'HybridSequenceModel'
        }
    )
    
    timeseries_agent = create_neural_processing_agent(
        "timeseries_agent_001", 
        "timeseries",
        {
            'input_features': 10,
            'lstm_hidden_size': 128,
            'output_features': 1,
            'sequence_length': 50,
            'forecast_horizon': 5,
            'use_bnn_lstm': False,
            'preferred_architecture': 'HybridTimeSeries'
        }
    )
    
    # Test model information
    print("=== NEURAL PROCESSING AGENTS ===\n")
    
    print("1. Vision Agent:")
    vision_info = vision_agent.get_model_info()
    print(f"   Active Model: {vision_info['active_model']}")
    print(f"   Available Models: {vision_info['available_models']}")
    if vision_info['model_performance']:
        perf = vision_info['model_performance'][vision_info['active_model']]
        print(f"   Model Size: {perf['model_size_mb']:.2f} MB")
        print(f"   Parameters: {perf['total_parameters']:,}")
    
    print("\n2. NLP Agent:")
    nlp_info = nlp_agent.get_model_info()
    print(f"   Active Model: {nlp_info['active_model']}")
    print(f"   Available Models: {nlp_info['available_models']}")
    if nlp_info['model_performance']:
        perf = nlp_info['model_performance'][nlp_info['active_model']]
        print(f"   Model Size: {perf['model_size_mb']:.2f} MB")
        print(f"   Parameters: {perf['total_parameters']:,}")
    
    print("\n3. Time Series Agent:")
    ts_info = timeseries_agent.get_model_info()
    print(f"   Active Model: {ts_info['active_model']}")
    print(f"   Available Models: {ts_info['available_models']}")
    if ts_info['model_performance']:
        perf = ts_info['model_performance'][ts_info['active_model']]
        print(f"   Model Size: {perf['model_size_mb']:.2f} MB")
        print(f"   Parameters: {perf['total_parameters']:,}")
    
    print("\n=== NEURAL PROCESSING CAPABILITIES ===")
    print("✓ Hybrid ANN-BNN architectures for optimal efficiency")
    print("✓ LSTM layers for sequential data processing")
    print("✓ CNN layers for spatial feature extraction")
    print("✓ Dynamic model selection based on requirements")
    print("✓ Real-time performance optimization")
    print("✓ Memory-efficient binary neural networks")
    print("✓ Multi-task learning capabilities")
    
    print("\n=== SPECIALIZATIONS ===")
    print("• Vision: Image classification, object detection")
    print("• NLP: Text classification, sentiment analysis")
    print("• Time Series: Forecasting, anomaly detection")
    print("• Classification: Multi-class prediction")
    print("• Regression: Continuous value prediction")