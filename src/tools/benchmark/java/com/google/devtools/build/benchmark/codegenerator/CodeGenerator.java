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
import com.google.common.collect.ImmutableSet;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Abstract base class for creating 4 types of project, or modify existing ones.
 * Subclasses are for different languages.
 */
public abstract class CodeGenerator {

  @VisibleForTesting static final String TARGET_A_FEW_FILES = "AFewFiles";
  @VisibleForTesting static final String TARGET_MANY_FILES = "ManyFiles";
  @VisibleForTesting static final String TARGET_LONG_CHAINED_DEPS = "LongChainedDeps";
  @VisibleForTesting static final String TARGET_PARALLEL_DEPS = "ParallelDeps";

  public void generateNewProject(String outputDir, ImmutableSet<String> projectNames) {
    Path dir = Paths.get(outputDir);
    for (String projectName : projectNames) {
      switch (projectName) {
        case TARGET_A_FEW_FILES:
          createTargetWithSomeFiles(dir.resolve(TARGET_A_FEW_FILES), getSizeAFewFiles());
          break;
        case TARGET_MANY_FILES:
          createTargetWithSomeFiles(dir.resolve(TARGET_MANY_FILES), getSizeManyFiles());
          break;
        case TARGET_LONG_CHAINED_DEPS:
          createTargetWithLongChainedDeps(dir.resolve(TARGET_LONG_CHAINED_DEPS));
          break;
        case TARGET_PARALLEL_DEPS:
          createTargetWithParallelDeps(dir.resolve(TARGET_PARALLEL_DEPS));
          break;
        default:
          // Do nothing
      }
    }
  }

  public void modifyExistingProject(String outputDir, ImmutableSet<String> projectNames) {
    Path dir = Paths.get(outputDir);
    for (String projectName : projectNames) {
      switch (projectName) {
        case TARGET_A_FEW_FILES:
          modifyTargetWithSomeFiles(dir.resolve(TARGET_A_FEW_FILES));
          break;
        case TARGET_MANY_FILES:
          modifyTargetWithSomeFiles(dir.resolve(TARGET_MANY_FILES));
          break;
        case TARGET_LONG_CHAINED_DEPS:
          modifyTargetWithLongChainedDeps(dir.resolve(TARGET_LONG_CHAINED_DEPS));
          break;
        case TARGET_PARALLEL_DEPS:
          modifyTargetWithParallelDeps(dir.resolve(TARGET_PARALLEL_DEPS));
          break;
        default:
          // Do nothing
      }
    }
  }

  abstract void createTargetWithSomeFiles(Path projectPath, int numberOfFiles);
  abstract void modifyTargetWithSomeFiles(Path projectPath);

  abstract void createTargetWithLongChainedDeps(Path projectPath);
  abstract void modifyTargetWithLongChainedDeps(Path projectPath);

  abstract void createTargetWithParallelDeps(Path projectPath);
  abstract void modifyTargetWithParallelDeps(Path projectPath);

  public abstract String getDirSuffix();
  public abstract int getSizeAFewFiles();
  public abstract int getSizeManyFiles();
  public abstract int getSizeLongChainedDeps();
  public abstract int getSizeParallelDeps();
}