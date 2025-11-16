# TGCN_PLUS
AST-based Subgraph Construction with Noise Reduction for Cross-Project Defect Prediction

## Requirements

TGCN_PLUS is executed on Linux (Ubuntu 22.04.3).

## Quick Start

### Preprocessing

Generate a global dictionary
```bash
python GenerateGlobalDictionary.py
```
Generate a list of key node types
```bash
python preprocess.py
```

### Running

```bash
python run_cross-project.py
```

