// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.benchmark.codegenerator;

import com.google.common.annotations.VisibleForTesting;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Abstract base class for creating 4 types of project, or modify existing ones.
 * Subclasses are for different languages.
 */
public abstract class CodeGenerator {

  @VisibleForTesting static final String TARGET_A_FEW_FILES = "AFewFiles";
  @VisibleForTesting static final int SIZE_A_FEW_FILES = 10;

  @VisibleForTesting static final String TARGET_MANY_FILES = "ManyFiles";
  @VisibleForTesting static final int SIZE_MANY_FILES = 1000;

  @VisibleForTesting static final String TARGET_LONG_CHAINED_DEPS = "LongChainedDeps";
  @VisibleForTesting static final int SIZE_LONG_CHAINED_DEPS = 20;

  @VisibleForTesting static final String TARGET_PARALLEL_DEPS = "ParallelDeps";
  @VisibleForTesting static final int SIZE_PARALLEL_DEPS = 20;

  public void generateNewProject(
      String outputDir,
      boolean aFewFiles,
      boolean manyFiles,
      boolean longChainedDeps,
      boolean parallelDeps) {
    Path dir = Paths.get(outputDir);
    if (aFewFiles) {
      createTargetWithSomeFiles(dir.resolve(TARGET_A_FEW_FILES), SIZE_A_FEW_FILES);
    }
    if (manyFiles) {
      createTargetWithSomeFiles(dir.resolve(TARGET_MANY_FILES), SIZE_MANY_FILES);
    }
    if (longChainedDeps) {
      createTargetWithLongChainedDeps(dir.resolve(TARGET_LONG_CHAINED_DEPS));
    }
    if (parallelDeps) {
      createTargetWithParallelDeps(dir.resolve(TARGET_PARALLEL_DEPS));
    }
  }

  public void modifyExistingProject(
      String outputDir,
      boolean aFewFiles,
      boolean manyFiles,
      boolean longChainedDeps,
      boolean parallelDeps) {
    Path dir = Paths.get(outputDir);
    if (aFewFiles) {
      modifyTargetWithSomeFiles(dir.resolve(TARGET_A_FEW_FILES));
    }
    if (manyFiles) {
      modifyTargetWithSomeFiles(dir.resolve(TARGET_MANY_FILES));
    }
    if (longChainedDeps) {
      modifyTargetWithLongChainedDeps(dir.resolve(TARGET_LONG_CHAINED_DEPS));
    }
    if (parallelDeps) {
      modifyTargetWithParallelDeps(dir.resolve(TARGET_PARALLEL_DEPS));
    }
  }

  abstract void createTargetWithSomeFiles(Path projectPath, int numberOfFiles);
  abstract void modifyTargetWithSomeFiles(Path projectPath);

  abstract void createTargetWithLongChainedDeps(Path projectPath);
  abstract void modifyTargetWithLongChainedDeps(Path projectPath);

  abstract void createTargetWithParallelDeps(Path projectPath);
  abstract void modifyTargetWithParallelDeps(Path projectPath);

  public abstract String getDirSuffix();
}