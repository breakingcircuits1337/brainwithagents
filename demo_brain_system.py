"""
Brain System Demo - Practical Example

This script demonstrates practical usage of the brain system
with real-world examples and scenarios.
"""

import time
import json
from brain_system import create_enhanced_brain, BrainMode
from communication_system import SignalGenerator

def demo_data_analysis_scenario():
    """Demonstrate data analysis scenario"""
    print("=" * 60)
    print("DEMO: DATA ANALYSIS SCENARIO")
    print("=" * 60)
    
    # Create brain
    brain = create_enhanced_brain("data_analysis_brain")
    
    # Scenario: Analyze sales data
    print("\nScenario: Analyzing quarterly sales data...")
    
    sales_data = {
        "quarter": "Q4 2024",
        "products": {
            "Product A": [12000, 15000, 18000, 22000],
            "Product B": [8000, 9500, 11000, 13000],
            "Product C": [15000, 16000, 17500, 19000]
        },
        "regions": ["North", "South", "East", "West"],
        "analysis_requirements": [
            "calculate_growth_rates",
            "identify_top_performers",
            "detect_seasonal_patterns",
            "generate_recommendations"
        ]
    }
    
    print(f"Input data: {json.dumps(sales_data, indent=2)}")
    
    # Process through brain
    result = brain.process_input(sales_data, "sales_analysis")
    
    print(f"\nAnalysis Result:")
    print(f"Status: {result['processing_summary']['status']}")
    print(f"Agents processed: {result['processing_summary']['agents_processed']}")
    print(f"Success rate: {result['processing_summary']['success_rate']:.2%}")
    print(f"Specializations involved: {result['processing_summary']['specializations_involved']}")
    
    # Show detailed results
    if result['agent_results']:
        print(f"\nDetailed Results:")
        for agent_result in result['agent_results']:
            print(f"- {agent_result['agent_id']} ({agent_result['specialization']}): "
                  f"Processing time: {agent_result['processing_time']:.4f}s")
    
    brain.shutdown()
    print("\nData analysis demo completed!")

def demo_code_processing_scenario():
    """Demonstrate code processing scenario"""
    print("\n" + "=" * 60)
    print("DEMO: CODE PROCESSING SCENARIO")
    print("=" * 60)
    
    # Create brain
    brain = create_enhanced_brain("code_processing_brain")
    
    # Scenario: Fix Python code with syntax errors
    print("\nScenario: Fixing Python code with syntax errors...")
    
    problematic_code = """
def calculate_average(numbers)
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers
    
def find_maximum(numbers)
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
    
# Test the functions
test_data = [1, 2, 3, 4, 5]
print("Average:", calculate_average(test_data))
print("Maximum:", find_maximum(test_data))
"""
    
    print("Input code:")
    print(problematic_code)
    
    # Process through brain
    result = brain.process_input(problematic_code, "python_syntax_fix")
    
    print(f"\nCode Processing Result:")
    print(f"Status: {result['processing_summary']['status']}")
    print(f"Success rate: {result['processing_summary']['success_rate']:.2%}")
    
    # Show if Python fixer was involved
    if result['agent_results']:
        python_fixer_result = next(
            (r for r in result['agent_results'] if r['specialization'] == 'PythonSyntaxFixer'),
            None
        )
        if python_fixer_result:
            print(f"Python fixer processed: {python_fixer_result['result']['success']}")
    
    brain.shutdown()
    print("\nCode processing demo completed!")

def demo_creative_writing_scenario():
    """Demonstrate creative writing scenario"""
    print("\n" + "=" * 60)
    print("DEMO: CREATIVE WRITING SCENARIO")
    print("=" * 60)
    
    # Create brain
    brain = create_enhanced_brain("creative_writing_brain")
    
    # Set to creative mode
    brain.set_mode(BrainMode.CREATIVE)
    
    # Scenario: Generate creative content
    print("\nScenario: Generating creative content about AI...")
    
    creative_request = {
        "topic": "The Future of Artificial Intelligence",
        "style": "inspirational",
        "length": "medium",
        "key_points": [
            "AI's impact on society",
            "Ethical considerations",
            "Technological advancements",
            "Human-AI collaboration"
        ],
        "target_audience": "general_public"
    }
    
    print(f"Creative request: {json.dumps(creative_request, indent=2)}")
    
    # Process through brain
    result = brain.process_input(creative_request, "creative_writing")
    
    print(f"\nCreative Writing Result:")
    print(f"Status: {result['processing_summary']['status']}")
    print(f"Agents processed: {result['processing_summary']['agents_processed']}")
    print(f"Specializations involved: {result['processing_summary']['specializations_involved']}")
    
    # Show creative writer involvement
    if result['agent_results']:
        creative_writer_result = next(
            (r for r in result['agent_results'] if r['specialization'] == 'CreativeWriter'),
            None
        )
        if creative_writer_result:
            print(f"Creative writer processed: {creative_writer_result['result']['success']}")
    
    brain.shutdown()
    print("\nCreative writing demo completed!")

def demo_continuous_monitoring_scenario():
    """Demonstrate continuous monitoring scenario"""
    print("\n" + "=" * 60)
    print("DEMO: CONTINUOUS MONITORING SCENARIO")
    print("=" * 60)
    
    # Create brain
    brain = create_enhanced_brain("monitoring_brain")
    
    # Set to reactive mode for monitoring
    brain.set_mode(BrainMode.REACTIVE)
    
    # Scenario: Monitor system metrics continuously
    print("\nScenario: Continuous system monitoring...")
    
    # Start continuous operation
    brain.start_continuous_operation()
    
    # Simulate monitoring data
    monitoring_data = [
        {"cpu_usage": 45.2, "memory_usage": 62.8, "disk_usage": 78.3},
        {"cpu_usage": 52.1, "memory_usage": 68.5, "disk_usage": 79.1},
        {"cpu_usage": 78.9, "memory_usage": 82.3, "disk_usage": 85.7},  # High usage
        {"cpu_usage": 91.2, "memory_usage": 94.1, "disk_usage": 92.8},  # Critical
        {"cpu_usage": 67.3, "memory_usage": 71.2, "disk_usage": 81.5},  # Recovering
    ]
    
    print("Sending monitoring data...")
    
    for i, data in enumerate(monitoring_data):
        # Create monitoring signal
        signal = SignalGenerator.create_data_signal(
            {
                "type": "system_metrics",
                "timestamp": time.time(),
                "metrics": data,
                "alert_thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "disk_usage": 90.0
                }
            },
            source_id="system_monitor",
            strength=0.8 if any(data[k] > 80 for k in data) else 0.5
        )
        
        # Broadcast to brain
        brain.communication_network.broadcast_signal(signal)
        
        print(f"Data point {i+1}: CPU={data['cpu_usage']}%, "
              f"Memory={data['memory_usage']}%, Disk={data['disk_usage']}%")
        
        # Check for alerts
        if any(data[k] > 80 for k in data):
            print("  ⚠️  High usage detected!")
        if any(data[k] > 90 for k in data):
            print("  🚨 Critical usage detected!")
        
        time.sleep(1)
    
    # Let it process
    print("\nProcessing monitoring data...")
    time.sleep(3)
    
    # Stop continuous operation
    brain.stop_continuous_operation()
    
    # Get monitoring results
    metrics = brain.get_comprehensive_metrics()
    
    print(f"\nMonitoring Results:")
    print(f"Total signals processed: {metrics['metrics']['total_signals_processed']}")
    print(f"Success rate: {metrics['metrics']['success_rate']:.2%}")
    print(f"Network efficiency: {metrics['metrics']['network_efficiency']:.2%}")
    
    # Show agent activity
    if 'network_status' in metrics:
        agent_activity = metrics['network_status']['network_metrics']['agent_activity']
        print(f"Most active agent: {max(agent_activity.items(), key=lambda x: x[1])}")
    
    brain.shutdown()
    print("\nContinuous monitoring demo completed!")

def demo_learning_scenario():
    """Demonstrate learning scenario"""
    print("\n" + "=" * 60)
    print("DEMO: LEARNING SCENARIO")
    print("=" * 60)
    
    # Create brain
    brain = create_enhanced_brain("learning_brain")
    
    # Set to learning mode
    brain.set_mode(BrainMode.LEARNING)
    
    # Scenario: Learn from patterns
    print("\nScenario: Learning from data patterns...")
    
    # Start continuous operation for learning
    brain.start_continuous_operation()
    
    # Training data with patterns
    training_data = [
        {"pattern": "linear", "data": [1, 2, 3, 4, 5], "expected": "arithmetic_sequence"},
        {"pattern": "geometric", "data": [2, 4, 8, 16, 32], "expected": "geometric_sequence"},
        {"pattern": "fibonacci", "data": [1, 1, 2, 3, 5], "expected": "fibonacci_sequence"},
        {"pattern": "squares", "data": [1, 4, 9, 16, 25], "expected": "perfect_squares"},
        {"pattern": "primes", "data": [2, 3, 5, 7, 11], "expected": "prime_numbers"},
    ]
    
    print("Training with pattern data...")
    
    for i, data in enumerate(training_data):
        # Create learning signal
        signal = SignalGenerator.create_data_signal(
            {
                "type": "pattern_learning",
                "training_data": data,
                "learning_phase": "training"
            },
            source_id="pattern_trainer",
            strength=0.9
        )
        
        # Send to brain
        brain.communication_network.broadcast_signal(signal)
        
        print(f"Training sample {i+1}: {data['pattern']} pattern")
        
        time.sleep(0.5)
    
    # Let learning process
    print("\nProcessing learning cycles...")
    time.sleep(5)
    
    # Test learning with new data
    print("\nTesting learning with new patterns...")
    
    test_data = [
        {"pattern": "test_linear", "data": [10, 20, 30, 40, 50]},
        {"pattern": "test_geometric", "data": [3, 9, 27, 81, 243]},
        {"pattern": "test_unknown", "data": [1, 8, 27, 64, 125]},  # Cubes
    ]
    
    for i, data in enumerate(test_data):
        # Create test signal
        signal = SignalGenerator.create_data_signal(
            {
                "type": "pattern_recognition",
                "test_data": data,
                "learning_phase": "testing"
            },
            source_id="pattern_tester",
            strength=0.8
        )
        
        # Send to brain
        brain.communication_network.broadcast_signal(signal)
        
        print(f"Test sample {i+1}: {data['pattern']}")
        
        time.sleep(0.5)
    
    # Stop continuous operation
    brain.stop_continuous_operation()
    
    # Get learning results
    metrics = brain.get_comprehensive_metrics()
    
    print(f"\nLearning Results:")
    print(f"Learning rate: {metrics['metrics']['learning_rate']}")
    print(f"Adaptation score: {metrics['metrics']['adaptation_score']}")
    print(f"Learning history size: {len(brain.learning_history)}")
    
    brain.shutdown()
    print("\nLearning demo completed!")

def main():
    """Main function to run all demos"""
    print("BRAIN SYSTEM PRACTICAL DEMOS")
    print("=" * 60)
    print("This script demonstrates practical usage scenarios")
    print("for the integrated brain system.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_data_analysis_scenario()
        demo_code_processing_scenario()
        demo_creative_writing_scenario()
        demo_continuous_monitoring_scenario()
        demo_learning_scenario()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe brain system has demonstrated its capabilities in:")
        print("✓ Data analysis and pattern recognition")
        print("✓ Code processing and syntax fixing")
        print("✓ Creative content generation")
        print("✓ Continuous monitoring and alerting")
        print("✓ Learning and adaptation")
        
    except Exception as e:
        print(f"\nERROR: Demo execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()