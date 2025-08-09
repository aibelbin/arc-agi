from analysis import ARCParser, ARCAnalyzer
import argparse

def main():
    print("ARC-AGI Analysis")
    print("=" * 30)

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["training", "test", "evaluation"], default="training",
                    help="Dataset split to analyze")
    args = ap.parse_args()

    parser = ARCParser()
    parser.load_data(args.split)
    analyzer = ARCAnalyzer(parser)
    
    summary = analyzer.get_dataset_summary()
    
    print(f"Total tasks: {summary['total_tasks']}")
    
    print(f"\nComplexity levels:")
    for level, count in summary['complexity_levels'].items():
        print(f"  {level}: {count} examples")
    
    print(f"\nTop transformation types:")
    for transform, count in summary['transformation_types'].most_common(5):
        print(f"  {transform}: {count} examples")
    
    print(f"\nMost common grid sizes:")
    for size, count in summary['grid_sizes'].most_common(5):
        print(f"  {size}: {count} examples")

if __name__ == "__main__":
    main()
