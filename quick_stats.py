from analysis import ARCParser, ARCAnalyzer

def main():
    parser = ARCParser()
    parser.load_data()
    
    analyzer = ARCAnalyzer(parser)
    
    print("=== ARC-AGI Dataset Statistics ===")
    training_stats = analyzer.get_dataset_statistics('training')
    print(f"Training tasks: {training_stats['total_tasks']}")
    print(f"Unique grid sizes: {len(training_stats['unique_grid_sizes'])}")
    print(f"Most common colors: {training_stats['most_common_colors']}")
    
    evaluation_stats = analyzer.get_dataset_statistics('evaluation')
    print(f"\nEvaluation tasks: {evaluation_stats['total_tasks']}")
    
    test_stats = analyzer.get_dataset_statistics('test')
    print(f"Test tasks: {test_stats['total_tasks']}")
    
    print("\n=== Sample Task Analysis ===")
    sample_task_id = list(parser.training_data.keys())[0]
    task_analysis = analyzer.analyze_task(sample_task_id)
    
    print(f"Task ID: {sample_task_id}")
    print(f"Training examples: {task_analysis['num_train_examples']}")
    print(f"Test examples: {task_analysis['num_test_examples']}")
    
    first_example = task_analysis['train_analyses'][0]
    print(f"\nFirst training example:")
    print(f"Input shape: {first_example['input']['shape']}")
    print(f"Output shape: {first_example['output']['shape']}")
    print(f"Shape changed: {first_example['transformation']['shape_change']}")
    
    transformations = first_example['transformation']
    detected_transforms = [k for k, v in transformations.items() 
                          if isinstance(v, bool) and v and k.startswith('is_')]
    if detected_transforms:
        print(f"Detected transformations: {detected_transforms}")

if __name__ == "__main__":
    main()
