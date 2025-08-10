"""
Demonstration of Neural Processing Agents integrated with the Brain System.
Shows how specialized neural agents work together in a collaborative network.
"""

import time
import random
from typing import Dict, List, Any
from brain_system import Brain
from neural_processing_agent import create_neural_processing_agent
from communication_system import Message

# Mock tensor class for demonstration when PyTorch is not available
class MockTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = [random.random() for _ in range(self._size_from_shape(shape))]
    
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
        import numpy as np
        return np.array(self.data)

# Mock PyTorch functions
def mock_randn(*shape):
    return MockTensor(shape)

def mock_softmax(tensor, dim=-1):
    # Simple softmax approximation
    data = tensor.data.copy()
    max_val = max(data)
    exp_data = [x - max_val for x in data]
    exp_data = [2.71828 ** x for x in exp_data]  # e^x
    sum_exp = sum(exp_data)
    softmax_data = [x / sum_exp for x in exp_data]
    result = MockTensor(tensor.shape)
    result.data = softmax_data
    return result

def mock_argmax(tensor, dim=-1):
    max_idx = tensor.data.index(max(tensor.data))
    return MockTensor((1,))

def mock_max(tensor):
    return max(tensor.data)

def mock_mean(tensor):
    return sum(tensor.data) / len(tensor.data)

def mock_std(tensor):
    mean = mock_mean(tensor)
    variance = sum((x - mean) ** 2 for x in tensor.data) / len(tensor.data)
    return variance ** 0.5

# Mock torch module
class MockTorch:
    @staticmethod
    def randn(*args):
        return mock_randn(*args)
    
    @staticmethod
    def softmax(tensor, dim=-1):
        return mock_softmax(tensor, dim)
    
    @staticmethod
    def argmax(tensor, dim=-1):
        return mock_argmax(tensor, dim)
    
    @staticmethod
    def max(tensor):
        return mock_max(tensor)
    
    @staticmethod
    def mean(tensor):
        return mock_mean(tensor)
    
    @staticmethod
    def std(tensor):
        return mock_std(tensor)

# Use mock torch if real torch is not available
try:
    import torch
except ImportError:
    torch = MockTorch()

class NeuralBrainDemo:
    """Demonstration of neural processing capabilities in the brain system."""
    
    def __init__(self):
        self.brain = Brain("neural_brain_demo")
        self.setup_neural_agents()
        self.demo_results = {}
    
    def setup_neural_agents(self):
        """Create and register specialized neural processing agents."""
        
        # Vision processing agent
        vision_agent = create_neural_processing_agent(
            "vision_neural_agent",
            "vision",
            {
                'input_channels': 3,
                'num_classes': 10,
                'bnn_conv_layers': [1, 2],
                'preferred_architecture': 'HybridCNN'
            }
        )
        
        # NLP processing agent
        nlp_agent = create_neural_processing_agent(
            "nlp_neural_agent", 
            "nlp",
            {
                'input_size': 100,
                'hidden_sizes': [256, 128],
                'lstm_hidden_size': 128,
                'num_classes': 5,
                'sequence_length': 50,
                'use_cnn': True,
                'use_bnn_lstm': True,
                'preferred_architecture': 'HybridSequenceModel'
            }
        )
        
        # Time series processing agent
        timeseries_agent = create_neural_processing_agent(
            "timeseries_neural_agent",
            "timeseries",
            {
                'input_features': 5,
                'lstm_hidden_size': 64,
                'output_features': 1,
                'sequence_length': 30,
                'forecast_horizon': 10,
                'use_bnn_lstm': False,
                'preferred_architecture': 'HybridTimeSeries'
            }
        )
        
        # Classification agent
        classification_agent = create_neural_processing_agent(
            "classification_neural_agent",
            "classification",
            {
                'input_size': 784,
                'hidden_sizes': [512, 256, 128],
                'num_classes': 10,
                'bnn_layers': [1],
                'preferred_architecture': 'HybridANNBNN'
            }
        )
        
        # Register agents with brain system
        self.brain.communication_network.add_agent(vision_agent)
        self.brain.communication_network.add_agent(nlp_agent)
        self.brain.communication_network.add_agent(timeseries_agent)
        self.brain.communication_network.add_agent(classification_agent)
        
        # Create neural connections between agents
        self.brain.communication_network.connect_agents("vision_neural_agent", "nlp_neural_agent", 0.8)
        self.brain.communication_network.connect_agents("nlp_neural_agent", "classification_neural_agent", 0.7)
        self.brain.communication_network.connect_agents("timeseries_neural_agent", "classification_neural_agent", 0.9)
        self.brain.communication_network.connect_agents("vision_neural_agent", "classification_neural_agent", 0.6)
        
        print("✓ Neural processing agents registered and connected")
    
    def demo_vision_processing(self):
        """Demonstrate vision processing capabilities."""
        print("\n=== VISION PROCESSING DEMO ===")
        
        # Create sample image data (simulated 32x32 RGB image)
        image_data = torch.randn(1, 3, 32, 32)  # Batch=1, Channels=3, Height=32, Width=32
        
        # Create task message
        task_message = Message(
            sender_id="demo_system",
            receiver_id="vision_neural_agent",
            content={
                'task_type': 'classification',
                'data': image_data,
                'requirements': {
                    'accuracy': 0.85,
                    'efficiency': 0.7
                }
            },
            signal_strength=0.9
        )
        
        # Process through brain system
        start_time = time.time()
        result = self.brain.process_input(task_message, "vision_processing")
        processing_time = time.time() - start_time
        
        if result and result.get("success"):
            print(f"✓ Vision processing completed in {processing_time:.3f}s")
            
            # Extract agent results
            agent_results = result.get("agent_results", [])
            if agent_results:
                for agent_result in agent_results:
                    agent_id = agent_result.get("agent_id", "unknown")
                    specialization = agent_result.get("specialization", "unknown")
                    print(f"  Agent: {agent_id} ({specialization})")
                    
                    agent_result_data = agent_result.get("result", {})
                    if agent_result_data.get("success"):
                        print(f"  Processing time: {agent_result.get('processing_time', 0):.3f}s")
            
            self.demo_results['vision'] = {
                'success': True,
                'processing_time': processing_time,
                'result': result
            }
        else:
            print("✗ Vision processing failed")
            self.demo_results['vision'] = {'success': False}
    
    def demo_nlp_processing(self):
        """Demonstrate NLP processing capabilities."""
        print("\n=== NLP PROCESSING DEMO ===")
        
        # Create sample text sequence data
        text_sequence = torch.randn(1, 50, 100)  # Batch=1, Sequence=50, Features=100
        
        # Create task message
        task_message = Message(
            sender_id="demo_system",
            receiver_id="nlp_neural_agent",
            content={
                'task_type': 'classification',
                'data': text_sequence,
                'requirements': {
                    'accuracy': 0.8,
                    'efficiency': 0.8
                }
            },
            signal_strength=0.85
        )
        
        # Process through brain system
        start_time = time.time()
        result = self.brain.process_input(task_message, "vision_processing")
        processing_time = time.time() - start_time
        
        if result:
            print(f"✓ NLP processing completed in {processing_time:.3f}s")
            print(f"  Agent: {result.content.get('agent_id', 'unknown')}")
            print(f"  Model: {result.content.get('processing_metadata', {}).get('model_used', 'unknown')}")
            
            neural_result = result.content.get('neural_result', {})
            if neural_result.get('success'):
                output = neural_result.get('output', {})
                print(f"  Predicted class: {output.get('predicted_class', 'N/A')}")
                print(f"  Confidence: {output.get('confidence', 0):.3f}")
                print(f"  Processing time: {neural_result.get('processing_time', 0):.3f}s")
                print(f"  Memory usage: {neural_result.get('memory_usage', 0):.3f} MB")
            
            self.demo_results['nlp'] = {
                'success': True,
                'processing_time': processing_time,
                'result': result
            }
        else:
            print("✗ NLP processing failed")
            self.demo_results['nlp'] = {'success': False}
    
    def demo_timeseries_processing(self):
        """Demonstrate time series processing capabilities."""
        print("\n=== TIME SERIES PROCESSING DEMO ===")
        
        # Create sample time series data
        timeseries_data = torch.randn(1, 30, 5)  # Batch=1, Sequence=30, Features=5
        
        # Create task message
        task_message = Message(
            sender_id="demo_system",
            receiver_id="timeseries_neural_agent",
            content={
                'task_type': 'forecasting',
                'data': timeseries_data,
                'requirements': {
                    'accuracy': 0.75,
                    'efficiency': 0.9
                }
            },
            signal_strength=0.8
        )
        
        # Process through brain system
        start_time = time.time()
        result = self.brain.process_input(task_message, "vision_processing")
        processing_time = time.time() - start_time
        
        if result:
            print(f"✓ Time series processing completed in {processing_time:.3f}s")
            print(f"  Agent: {result.content.get('agent_id', 'unknown')}")
            print(f"  Model: {result.content.get('processing_metadata', {}).get('model_used', 'unknown')}")
            
            neural_result = result.content.get('neural_result', {})
            if neural_result.get('success'):
                output = neural_result.get('output', {})
                predictions = output.get('predictions', [])
                print(f"  Forecast horizon: {len(predictions)} steps")
                print(f"  Prediction mean: {output.get('mean', 0):.3f}")
                print(f"  Prediction std: {output.get('std', 0):.3f}")
                print(f"  Processing time: {neural_result.get('processing_time', 0):.3f}s")
                print(f"  Memory usage: {neural_result.get('memory_usage', 0):.3f} MB")
            
            self.demo_results['timeseries'] = {
                'success': True,
                'processing_time': processing_time,
                'result': result
            }
        else:
            print("✗ Time series processing failed")
            self.demo_results['timeseries'] = {'success': False}
    
    def demo_collaborative_processing(self):
        """Demonstrate collaborative processing between neural agents."""
        print("\n=== COLLABORATIVE NEURAL PROCESSING DEMO ===")
        
        # Create a complex task that requires multiple agents
        complex_task = Message(
            sender_id="demo_system",
            receiver_id="vision_neural_agent",  # Start with vision
            content={
                'task_type': 'multimodal_analysis',
                'data': {
                    'image': torch.randn(1, 3, 32, 32),
                    'text': torch.randn(1, 50, 100),
                    'timeseries': torch.randn(1, 30, 5)
                },
                'requirements': {
                    'accuracy': 0.85,
                    'efficiency': 0.75,
                    'collaboration': True
                }
            },
            signal_strength=0.95
        )
        
        # Process through brain system (should trigger agent collaboration)
        start_time = time.time()
        results = []
        
        # Send to multiple agents
        for agent_id in ["vision_neural_agent", "nlp_neural_agent", "timeseries_neural_agent"]:
            task_message = Message(
                sender_id="demo_system",
                receiver_id=agent_id,
                content=complex_task.content,
                signal_strength=0.9
            )
            result = self.brain.process_message(task_message)
            if result:
                results.append(result)
        
        processing_time = time.time() - start_time
        
        print(f"✓ Collaborative processing completed in {processing_time:.3f}s")
        print(f"  Agents involved: {len(results)}")
        
        for i, result in enumerate(results):
            agent_id = result.content.get('agent_id', f'agent_{i}')
            model_used = result.content.get('processing_metadata', {}).get('model_used', 'unknown')
            print(f"  {agent_id}: {model_used}")
            
            neural_result = result.content.get('neural_result', {})
            if neural_result.get('success'):
                proc_time = neural_result.get('processing_time', 0)
                memory = neural_result.get('memory_usage', 0)
                print(f"    Processing time: {proc_time:.3f}s, Memory: {memory:.3f} MB")
        
        self.demo_results['collaborative'] = {
            'success': True,
            'processing_time': processing_time,
            'results': results
        }
    
    def demo_model_optimization(self):
        """Demonstrate model optimization capabilities."""
        print("\n=== MODEL OPTIMIZATION DEMO ===")
        
        # Get vision agent
        vision_agent = self.brain.agents.get("vision_neural_agent")
        if not vision_agent:
            print("✗ Vision agent not found")
            return
        
        # Get initial model info
        initial_info = vision_agent.get_model_info()
        print(f"Initial model: {initial_info['active_model']}")
        
        if initial_info['model_performance']:
            perf = initial_info['model_performance'][initial_info['active_model']]
            print(f"  Model size: {perf['model_size_mb']:.2f} MB")
            print(f"  Parameters: {perf['total_parameters']:,}")
        
        # Optimize for efficiency
        print("\nOptimizing for efficiency...")
        optimization_result = vision_agent.optimize_model('efficiency')
        
        if optimization_result['success']:
            print(f"✓ Optimization successful: {optimization_result['optimization']}")
            print(f"  BNN layers changed from {optimization_result['original_bnn_layers']} to {optimization_result['new_bnn_layers']}")
            
            # Get updated model info
            updated_info = vision_agent.get_model_info()
            if updated_info['model_performance']:
                perf = updated_info['model_performance'][updated_info['active_model']]
                print(f"  Updated model size: {perf['model_size_mb']:.2f} MB")
        
        # Test processing with optimized model
        test_data = torch.randn(1, 3, 32, 32)
        task_message = Message(
            sender_id="demo_system",
            receiver_id="vision_neural_agent",
            content={
                'task_type': 'classification',
                'data': test_data,
                'requirements': {'efficiency': 0.9}
            },
            signal_strength=0.8
        )
        
        start_time = time.time()
        result = self.brain.process_message(task_message)
        processing_time = time.time() - start_time
        
        if result:
            neural_result = result.content.get('neural_result', {})
            if neural_result.get('success'):
                print(f"✓ Optimized model processing time: {neural_result.get('processing_time', 0):.3f}s")
                print(f"  Memory usage: {neural_result.get('memory_usage', 0):.3f} MB")
        
        self.demo_results['optimization'] = {
            'success': True,
            'optimization_result': optimization_result,
            'processing_time': processing_time
        }
    
    def run_all_demos(self):
        """Run all neural processing demonstrations."""
        print("🧠 NEURAL BRAIN SYSTEM DEMONSTRATION")
        print("=" * 50)
        
        # Run individual demos
        self.demo_vision_processing()
        self.demo_nlp_processing()
        self.demo_timeseries_processing()
        self.demo_collaborative_processing()
        self.demo_model_optimization()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print demonstration summary."""
        print("\n" + "=" * 50)
        print("📊 DEMONSTRATION SUMMARY")
        print("=" * 50)
        
        successful_demos = sum(1 for result in self.demo_results.values() if result.get('success', False))
        total_demos = len(self.demo_results)
        
        print(f"✓ Successful demos: {successful_demos}/{total_demos}")
        
        for demo_name, result in self.demo_results.items():
            status = "✓" if result.get('success', False) else "✗"
            proc_time = result.get('processing_time', 0)
            print(f"{status} {demo_name.capitalize()}: {proc_time:.3f}s")
        
        # Brain system stats
        print(f"\n🧠 Brain System Stats:")
        print(f"  Total agents: {len(self.brain.agents)}")
        print(f"  Total connections: {len(self.brain.connections)}")
        print(f"  Messages processed: {self.brain.get_system_stats().get('total_messages_processed', 0)}")
        
        # Neural agent stats
        neural_agents = [agent for agent in self.brain.agents.values() 
                        if hasattr(agent, 'models')]
        print(f"  Neural agents: {len(neural_agents)}")
        
        total_params = 0
        total_memory = 0
        for agent in neural_agents:
            if hasattr(agent, 'model_performance'):
                for perf in agent.model_performance.values():
                    total_params += perf.get('total_parameters', 0)
                    total_memory += perf.get('model_size_mb', 0)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total memory: {total_memory:.2f} MB")
        
        print("\n🎯 Key Capabilities Demonstrated:")
        print("  • Multi-modal neural processing")
        print("  • Hybrid ANN-BNN-LSTM architectures")
        print("  • Real-time model optimization")
        print("  • Collaborative agent processing")
        print("  • Memory-efficient binary networks")
        print("  • Dynamic model selection")

def main():
    """Main demonstration function."""
    demo = NeuralBrainDemo()
    demo.run_all_demos()

if __name__ == "__main__":
    main()