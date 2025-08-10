"""
Example Usage and Testing of the Brain System

This script demonstrates the capabilities of the integrated brain system
that combines the Four-Level HRM with the Neuron-style Communication System.
"""

import time
import json
from brain_system import Brain, BrainMode, create_enhanced_brain
from communication_system import SignalGenerator

def test_basic_brain_functionality():
    """Test basic brain functionality"""
    print("=" * 60)
    print("TESTING BASIC BRAIN FUNCTIONALITY")
    print("=" * 60)
    
    # Create a basic brain
    brain = Brain("test_brain_001")
    
    # Test 1: Process simple input
    print("\n1. Testing simple input processing...")
    test_input = {
        "task": "analyze_data",
        "data": [1, 2, 3, 4, 5],
        "requirements": ["calculate_mean", "find_max"]
    }
    
    result = brain.process_input(test_input, "data_analysis")
    print(f"Result: {json.dumps(result, indent=2, default=str)}")
    
    # Test 2: Get brain status
    print("\n2. Getting brain status...")
    status = brain.get_brain_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Test 3: Add specialized agent
    print("\n3. Adding specialized agent...")
    success = brain.add_specialized_agent("test_python_fixer", "PythonSyntaxFixer", 0.6)
    print(f"Agent added successfully: {success}")
    
    # Test 4: Process with specialized agent
    print("\n4. Testing with specialized agent...")
    python_code = """
    def example_function()
        print("Hello World")
        return True
    """
    
    result = brain.process_input(python_code, "python_syntax_fix")
    print(f"Python processing result: {json.dumps(result, indent=2, default=str)}")
    
    brain.shutdown()
    print("\nBasic brain functionality test completed!")

def test_enhanced_brain():
    """Test enhanced brain with multiple specialized agents"""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED BRAIN WITH MULTIPLE AGENTS")
    print("=" * 60)
    
    # Create enhanced brain
    brain = create_enhanced_brain("enhanced_brain_001")
    
    # Test 1: Process different types of inputs
    print("\n1. Testing different input types...")
    
    # Data analysis input
    data_input = {
        "type": "sales_data",
        "values": [100, 150, 200, 175, 225, 300],
        "period": "Q1-Q2 2024"
    }
    
    result = brain.process_input(data_input, "data_analysis")
    print(f"Data analysis result: {result['processing_summary']}")
    
    # Creative writing input
    creative_input = {
        "topic": "artificial intelligence",
        "style": "informative",
        "length": "medium"
    }
    
    result = brain.process_input(creative_input, "creative_writing")
    print(f"Creative writing result: {result['processing_summary']}")
    
    # SQL query input
    sql_input = {
        "tables": ["users", "orders"],
        "query_type": "join",
        "conditions": ["users.id = orders.user_id", "orders.total > 100"]
    }
    
    result = brain.process_input(sql_input, "sql_query")
    print(f"SQL query result: {result['processing_summary']}")
    
    # Test 2: Continuous operation
    print("\n2. Testing continuous operation...")
    brain.start_continuous_operation()
    
    # Send multiple signals
    for i in range(5):
        signal = SignalGenerator.create_data_signal(
            {"test_id": i, "data": f"test_data_{i}"},
            source_id="test_generator",
            strength=0.8
        )
        brain.communication_network.broadcast_signal(signal)
        time.sleep(0.5)
    
    # Let it process for a few seconds
    time.sleep(3)
    
    # Stop continuous operation
    brain.stop_continuous_operation()
    
    # Get final metrics
    print("\n3. Getting comprehensive metrics...")
    metrics = brain.get_comprehensive_metrics()
    print(f"Total signals processed: {metrics['metrics']['total_signals_processed']}")
    print(f"Success rate: {metrics['metrics']['success_rate']:.2%}")
    print(f"Agent count: {metrics['agent_count']}")
    
    brain.shutdown()
    print("\nEnhanced brain test completed!")

def test_brain_modes():
    """Test different brain operational modes"""
    print("\n" + "=" * 60)
    print("TESTING BRAIN OPERATIONAL MODES")
    print("=" * 60)
    
    brain = create_enhanced_brain("mode_test_brain")
    
    # Test each mode
    modes = [BrainMode.REACTIVE, BrainMode.PROACTIVE, BrainMode.LEARNING, BrainMode.CREATIVE]
    
    for mode in modes:
        print(f"\nTesting mode: {mode.value}")
        brain.set_mode(mode)
        
        # Process input in this mode
        test_input = {
            "mode_test": True,
            "mode": mode.value,
            "data": f"test_data_for_{mode.value}"
        }
        
        result = brain.process_input(test_input, "mode_test")
        print(f"Mode {mode.value} result: {result['processing_summary']['status']}")
        
        # Get status
        status = brain.get_brain_status()
        print(f"Current mode: {status['mode']}")
        
        time.sleep(1)
    
    brain.shutdown()
    print("\nBrain modes test completed!")

def test_brain_learning_and_optimization():
    """Test brain learning and optimization capabilities"""
    print("\n" + "=" * 60)
    print("TESTING BRAIN LEARNING AND OPTIMIZATION")
    print("=" * 60)
    
    brain = create_enhanced_brain("learning_test_brain")
    
    # Set to learning mode
    brain.set_mode(BrainMode.LEARNING)
    
    # Start continuous operation
    brain.start_continuous_operation()
    
    # Send a series of inputs for learning
    print("\n1. Sending learning inputs...")
    learning_inputs = [
        {"type": "pattern_recognition", "data": [1, 2, 4, 8, 16]},
        {"type": "pattern_recognition", "data": [2, 4, 8, 16, 32]},
        {"type": "pattern_recognition", "data": [3, 6, 12, 24, 48]},
        {"type": "anomaly_detection", "data": [1, 2, 3, 100, 5]},
        {"type": "anomaly_detection", "data": [10, 20, 30, 200, 50]}
    ]
    
    for i, input_data in enumerate(learning_inputs):
        result = brain.process_input(input_data, "learning_input")
        print(f"Learning input {i+1}: {result['processing_summary']['status']}")
        time.sleep(0.5)
    
    # Let learning process
    print("\n2. Processing learning cycles...")
    time.sleep(5)
    
    # Check learning history
    print(f"\n3. Learning history size: {len(brain.learning_history)}")
    
    # Check optimization history
    print(f"Optimization history size: {len(brain.optimization_history)}")
    
    # Stop continuous operation
    brain.stop_continuous_operation()
    
    # Get final metrics
    metrics = brain.get_comprehensive_metrics()
    print(f"\n4. Final metrics:")
    print(f"Learning rate: {metrics['metrics']['learning_rate']}")
    print(f"Adaptation score: {metrics['metrics']['adaptation_score']}")
    print(f"System health: {metrics['metrics']['system_health']}")
    
    brain.shutdown()
    print("\nLearning and optimization test completed!")

def test_brain_persistence():
    """Test brain state persistence"""
    print("\n" + "=" * 60)
    print("TESTING BRAIN STATE PERSISTENCE")
    print("=" * 60)
    
    # Create and configure a brain
    brain = create_enhanced_brain("persistence_test_brain")
    
    # Process some inputs
    print("\n1. Processing inputs before saving...")
    for i in range(3):
        test_input = {"test_id": i, "data": f"persistence_test_{i}"}
        result = brain.process_input(test_input, "persistence_test")
        print(f"Input {i+1}: {result['processing_summary']['status']}")
    
    # Save brain state
    print("\n2. Saving brain state...")
    brain.save_brain_state("brain_state_test.json")
    
    # Get metrics before shutdown
    metrics_before = brain.get_comprehensive_metrics()
    print(f"Signals processed before save: {metrics_before['metrics']['total_signals_processed']}")
    
    # Shutdown
    brain.shutdown()
    
    # Create new brain and load state
    print("\n3. Loading brain state...")
    new_brain = Brain("loaded_brain")
    new_brain.load_brain_state("brain_state_test.json")
    
    # Get metrics after load
    metrics_after = new_brain.get_comprehensive_metrics()
    print(f"Signals processed after load: {metrics_after['metrics']['total_signals_processed']}")
    
    # Test that loaded brain works
    print("\n4. Testing loaded brain functionality...")
    test_input = {"test": "loaded_brain_test", "data": "post_load_test"}
    result = new_brain.process_input(test_input, "post_load_test")
    print(f"Post-load test result: {result['processing_summary']['status']}")
    
    new_brain.shutdown()
    print("\nPersistence test completed!")

def test_error_handling():
    """Test brain error handling"""
    print("\n" + "=" * 60)
    print("TESTING BRAIN ERROR HANDLING")
    print("=" * 60)
    
    brain = create_enhanced_brain("error_test_brain")
    
    # Test 1: Invalid input
    print("\n1. Testing invalid input...")
    try:
        result = brain.process_input(None, "invalid_test")
        print(f"Invalid input result: {result}")
    except Exception as e:
        print(f"Caught exception: {e}")
    
    # Test 2: Malformed input
    print("\n2. Testing malformed input...")
    malformed_input = {"malformed": True, "data": [1, 2, "invalid", 4]}
    result = brain.process_input(malformed_input, "malformed_test")
    print(f"Malformed input result: {result.get('success', False)}")
    
    # Test 3: Agent communication failure
    print("\n3. Testing agent communication failure...")
    # Remove an agent to simulate failure
    if "coordinator" in brain.communication_network.agents:
        brain.communication_network.remove_agent("coordinator")
    
    # Try to process input
    result = brain.process_input({"test": "no_coordinator"}, "failure_test")
    print(f"Agent failure result: {result.get('success', False)}")
    
    # Check error handling in metrics
    metrics = brain.get_comprehensive_metrics()
    print(f"Success rate after errors: {metrics['metrics']['success_rate']:.2%}")
    
    brain.shutdown()
    print("\nError handling test completed!")

def run_performance_test():
    """Run performance tests on the brain system"""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE TESTS")
    print("=" * 60)
    
    brain = create_enhanced_brain("performance_test_brain")
    
    # Test 1: Single input performance
    print("\n1. Testing single input performance...")
    start_time = time.time()
    
    test_input = {"performance_test": True, "data": list(range(100))}
    result = brain.process_input(test_input, "performance_test")
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Single input processing time: {processing_time:.4f} seconds")
    
    # Test 2: Multiple inputs performance
    print("\n2. Testing multiple inputs performance...")
    start_time = time.time()
    
    inputs = []
    for i in range(10):
        test_input = {"batch_test": True, "batch_id": i, "data": list(range(50))}
        inputs.append(test_input)
    
    results = []
    for input_data in inputs:
        result = brain.process_input(input_data, "batch_test")
        results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / len(inputs)
    
    print(f"Total time for {len(inputs)} inputs: {total_time:.4f} seconds")
    print(f"Average time per input: {average_time:.4f} seconds")
    
    # Test 3: Continuous operation performance
    print("\n3. Testing continuous operation performance...")
    brain.start_continuous_operation()
    
    start_time = time.time()
    
    # Send signals during continuous operation
    for i in range(20):
        signal = SignalGenerator.create_data_signal(
            {"continuous_test": True, "signal_id": i},
            source_id="performance_tester",
            strength=0.7
        )
        brain.communication_network.broadcast_signal(signal)
        time.sleep(0.1)
    
    # Let it process
    time.sleep(2)
    
    end_time = time.time()
    continuous_time = end_time - start_time
    
    brain.stop_continuous_operation()
    
    print(f"Continuous operation time: {continuous_time:.4f} seconds")
    
    # Get performance metrics
    metrics = brain.get_comprehensive_metrics()
    print(f"\n4. Performance metrics:")
    print(f"Total signals processed: {metrics['metrics']['total_signals_processed']}")
    print(f"Average processing time: {metrics['metrics']['average_processing_time']:.4f} seconds")
    print(f"Success rate: {metrics['metrics']['success_rate']:.2%}")
    print(f"Network efficiency: {metrics['metrics']['network_efficiency']:.2%}")
    
    brain.shutdown()
    print("\nPerformance test completed!")

def main():
    """Main function to run all tests"""
    print("BRAIN SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    print("This script tests the integrated brain system that combines")
    print("the Four-Level HRM with the Neuron-style Communication System.")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_brain_functionality()
        test_enhanced_brain()
        test_brain_modes()
        test_brain_learning_and_optimization()
        test_brain_persistence()
        test_error_handling()
        run_performance_test()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()