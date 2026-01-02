# Python Tutorial

This directory contains the example code for the [Bazel Python Tutorial](https://bazel.build/start/python).

## Overview

The tutorial is organized into three stages, each demonstrating increasingly complex Bazel concepts:

- **Stage 1**: A single target in a single package (basic `py_binary`)
- **Stage 2**: Multiple targets in a single package (`py_binary` + `py_library`)
- **Stage 3**: Multiple packages with visibility rules

## Prerequisites

- [Bazel](https://bazel.build/install) installed
- Python 3.11 or later (Bazel will download the toolchain automatically)

## Running the Examples

### Stage 1: Single Target

```bash
cd stage1
bazel build //main:hello_world
bazel-bin/main/hello_world
```

Or run directly:

```bash
bazel run //main:hello_world
```

### Stage 2: Multiple Targets

```bash
cd stage2
bazel build //main:hello_world
bazel run //main:hello_world
```

### Stage 3: Multiple Packages

```bash
cd stage3
bazel build //main:hello_world
bazel run //main:hello_world
```

## Directory Structure

```
python-tutorial/
├── README.md
├── stage1/
│   ├── MODULE.bazel
│   └── main/
│       ├── BUILD
│       └── hello_world.py
├── stage2/
│   ├── MODULE.bazel
│   └── main/
│       ├── BUILD
│       ├── greeting.py
│       └── hello_world.py
└── stage3/
    ├── MODULE.bazel
    ├── lib/
    │   ├── BUILD
    │   └── time_utils.py
    └── main/
        ├── BUILD
        ├── greeting.py
        └── hello_world.py
```

## Learn More

- [Bazel Python Tutorial](https://bazel.build/start/python)
- [rules_python Documentation](https://rules-python.readthedocs.io/)
- [Python Rules Reference](https://bazel.build/reference/be/python)
