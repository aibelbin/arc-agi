# ARC-AGI Competition Analysis Tools

This repository contains a comprehensive analysis framework for the ARC-AGI (Abstraction and Reasoning Corpus) competition.

## Project Structure

```
aib/
├── analysis/                 # Core analysis framework
│   ├── __init__.py          # Package initialization
│   ├── parser.py            # ARCParser - loads and manages datasets
│   ├── analyzer.py          # ARCAnalyzer - analyzes grids and transformations
│   ├── utils.py             # ARCUtils - high-level utilities and visualization
│   └── pattern_analysis.py  # Pattern analysis functions
├── default/                 # ARC-AGI dataset files
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   └── arc-agi_test_challenges.json
├── main.py                  # Main analysis script with pattern examples
├── quick_stats.py           # Quick dataset statistics
├── explore_tasks.py         # Task exploration utilities
└── README.md               # This file
```

## Quick Start

### 1. Basic Dataset Statistics
```bash
python3 quick_stats.py
```
Shows overview of datasets, grid sizes, and common transformations.

### 2. Task Exploration
```bash
python3 explore_tasks.py
```
Explores simple tasks, transformation types, and shows detailed task examples.

### 3. Full Pattern Analysis
```bash
python3 main.py
```
Runs comprehensive pattern analysis including color mapping examples.

## Core Components

### ARCParser
- Loads all ARC-AGI datasets (training, evaluation, test)
- Provides access to tasks and solutions
- Handles data path configuration

### ARCAnalyzer
- Analyzes individual grids (patterns, objects, statistics)
- Detects transformations (rotation, scaling, tiling, etc.)
- Analyzes complete tasks and finds patterns

### ARCUtils
- High-level interface for exploration
- Grid visualization
- Task filtering and searching
- Export functionality

## Usage Examples

### Load and Analyze a Specific Task
```python
from analysis import ARCUtils

utils = ARCUtils()
utils.show_task('0d3d703e', detailed=True)
```

### Find Simple Tasks for Development
```python
from analysis import ARCUtils

utils = ARCUtils()
simple_tasks = utils.find_simple_tasks(max_size=5, same_size_only=True)
print(f"Found {len(simple_tasks)} simple tasks")
```

### Analyze Transformations
```python
from analysis import ARCParser, ARCAnalyzer

parser = ARCParser()
parser.load_data()
analyzer = ARCAnalyzer(parser)

analysis = analyzer.analyze_task('task_id')
print(analysis['train_analyses'][0]['transformation'])
```

## Key Features

1. **Grid Analysis**: Shape, colors, symmetry, patterns, objects
2. **Transformation Detection**: Rotation, scaling, tiling, flipping
3. **Pattern Recognition**: Rectangular patterns, borders, symmetries
4. **Object Detection**: Connected component analysis with flood fill
5. **Task Comparison**: Similarity scoring and pattern matching
6. **Visualization**: ASCII grid display for debugging
7. **Export**: JSON export of analysis results

## Competition Strategy

Based on the analysis, recommended approach:

1. **Start Simple**: Focus on 27 small tasks (≤5x5, same input/output size)
2. **Color Mapping**: 65.5% of tasks preserve colors - implement color transformation detection
3. **Pattern Libraries**: Build libraries of common transformation patterns
4. **Geometric Transformations**: Handle rotation (6 tasks), scaling (12 tasks), etc.
5. **Hybrid Approach**: Combine rule-based and learning approaches

## Next Steps

1. Implement basic color mapping solver
2. Add geometric transformation detection
3. Build pattern library from analysis
4. Create validation framework
5. Develop ensemble of different approaches

## Requirements

- Python 3.7+
- numpy
- json (built-in)
- collections (built-in)

## Installation

```bash
pip3 install numpy
```

The framework is now organized for easy extension and development of ARC-AGI solving algorithms.
