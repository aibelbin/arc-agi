from analysis import ARCUtils

def main():
    utils = ARCUtils()
    
    print("=== ARC-AGI Task Explorer ===")
    
    print("\n1. Color Statistics:")
    color_stats = utils.get_color_stats()
    print(f"Colors used: {color_stats['unique_colors']}")
    print("Most frequent colors:")
    for color, count in list(color_stats['color_frequency'].items())[:5]:
        print(f"  Color {color}: {count} occurrences")
    
    print("\n2. Simple Tasks (≤5x5, same input/output size):")
    simple_tasks = utils.find_simple_tasks()
    print(f"Found {len(simple_tasks)} simple tasks")
    print(f"Examples: {simple_tasks[:5]}")
    
    print("\n3. Tasks by Transformation Type:")
    rotation_tasks = utils.find_transformation_type('rotation')
    scaling_tasks = utils.find_transformation_type('scaling')
    flip_tasks = utils.find_transformation_type('flip')
    tiling_tasks = utils.find_transformation_type('tiling')
    
    print(f"Rotation: {len(rotation_tasks)} tasks")
    print(f"Scaling: {len(scaling_tasks)} tasks") 
    print(f"Flip: {len(flip_tasks)} tasks")
    print(f"Tiling: {len(tiling_tasks)} tasks")
    
    print("\n4. Sample Task Analysis:")
    sample_task = simple_tasks[0] if simple_tasks else list(utils.parser.training_data.keys())[0]
    
    summary = utils.get_task_summary(sample_task)
    print(f"Task {sample_task}:")
    print(f"  Input: {summary['input_shape']}, Output: {summary['output_shape']}")
    print(f"  Shape changes: {summary['shape_changes']}")
    print(f"  Transformations: {summary['transformations']}")
    
    print(f"\n5. Detailed view of task {sample_task}:")
    utils.show_task(sample_task, detailed=True)

if __name__ == "__main__":
    main()
