# C++20 Modules Tools

## Overview

This folder contains two tools: `aggregate-ddi` and `generate-modmap`. These tools are designed to facilitate the processing of C++20 modules information and direct dependent information (DDI). They can aggregate module information, process dependencies, and generate module maps for use in C++20 modular projects.

## The format of DDI

The format of DDI content is [p1689](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1689r5.html).
for example,

```
{
  "revision": 0,
  "rules": [
    {
      "primary-output": "path/to/a.pcm",
      "provides": [
        {
          "is-interface": true,
          "logical-name": "a",
          "source-path": "path/to/a.cppm"
        }
      ],
      "requires": [
        {
          "logical-name": "b"
        }
      ]
    }
  ],
  "version": 1
}
```

## Tools

### `aggregate-ddi`

#### Description

`aggregate-ddi` is a tool that aggregates C++20 module information from multiple sources and processes DDI files to generate a consolidated output containing module paths and their dependencies.

#### Usage

```sh
aggregate-ddi -m <cpp20modules-info-file1> -m <cpp20modules-info-file2> ... -d <ddi-file1> <path/to/pcm1> -d <ddi-file2> <path/to/pcm2> ... -o <output-file>
```

#### Command Line Arguments

- `-m <cpp20modules-info-file>`: Path to a JSON file containing C++20 module information.
- `-d <ddi-file> <pcm-path>`: Path to a DDI file and its associated PCM path.
- `-o <output-file>`: Path to the output file where the aggregated information will be stored.

#### Example

```sh
aggregate-ddi -m module-info1.json -m module-info2.json -d ddi1.json /path/to/pcm1 -d ddi2.json /path/to/pcm2 -o output.json
```

### `generate-modmap`

#### Description

`generate-modmap` is a tool that generates a module map from a DDI file and C++20 modules information file. It creates two output files: one for the module map and one for the input module paths.

#### Usage

```sh
generate-modmap <ddi-file> <cpp20modules-info-file> <output-file> <compiler>
```

#### Command Line Arguments

- `<ddi-file>`: Path to the DDI file containing module dependencies.
- `<cpp20modules-info-file>`: Path to the JSON file containing C++20 modules information.
- `<output-file>`: Path to the output file where the module map will be stored.
- `<compiler>`: Compiler type the modmap to use. Only `clang`, `gcc`, `msvc-cl` supported.

#### Example

```sh
generate-modmap ddi.json cpp20modules-info.json modmap clang
```

This command will generate two files:
- `modmap`: containing the module map.
- `modmap.input`: containing the module paths.
