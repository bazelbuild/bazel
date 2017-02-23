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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Create 4 types of Java project, or modify existing ones. */
public class JavaCodeGenerator {

  @VisibleForTesting static final String TARGET_A_FEW_FILES = "AFewFiles";
  @VisibleForTesting static final int SIZE_A_FEW_FILES = 10;

  @VisibleForTesting static final String TARGET_MANY_FILES = "ManyFiles";
  @VisibleForTesting static final int SIZE_MANY_FILES = 1000;

  @VisibleForTesting static final String TARGET_LONG_CHAINED_DEPS = "LongChainedDeps";
  @VisibleForTesting static final int SIZE_LONG_CHAINED_DEPS = 20;

  @VisibleForTesting static final String TARGET_PARALLEL_DEPS = "ParallelDeps";
  @VisibleForTesting static final int SIZE_PARALLEL_DEPS = 20;

  public static void generateNewProject(
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

  public static void modifyExistingProject(
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

  /** Target type 1/2: Create targets with some files */
  private static void createTargetWithSomeFiles(Path projectPath, int numberOfFiles) {
    if (pathExists(projectPath)) {
      return;
    }

    try {
      Files.createDirectories(projectPath);

      for (int i = 0; i < numberOfFiles; ++i) {
        JavaCodeGeneratorHelper.writeRandomClassToDir(
            /* addExtraMethod = */ false, "RandomClass" + i, "com.example.generated", projectPath);
      }

      JavaCodeGeneratorHelper.writeMainClassToDir("com.example.generated", projectPath);
      JavaCodeGeneratorHelper.buildFileWithMainClass(projectPath.getFileName().toString(), "", projectPath);
    } catch (IOException e) {
      System.err.println("Error creating target with some files: " + e.getMessage());
    }
  }

  /** Target type 1/2: Modify targets with some files */
  private static void modifyTargetWithSomeFiles(Path projectPath) {
    File dir = projectPath.toFile();
    if (directoryExists(dir)) {
      System.err.format(
          "Project dir (%s) does not contain code for modification.\n", projectPath.toString());
      return;
    }
    try {
      JavaCodeGeneratorHelper.writeRandomClassToDir(
          /* addExtraMethod = */ true, "RandomClass0", "com.example.generated", projectPath);
    } catch (IOException e) {
      System.err.println("Error modifying targets some files: " + e.getMessage());
    }
  }

  /** Target type 3: Create targets with a few long chained dependencies (A -> B -> C -> … -> Z) */
  private static void createTargetWithLongChainedDeps(Path projectPath) {
    if (pathExists(projectPath)) {
      return;
    }

    try {
      Files.createDirectories(projectPath);

      int count = SIZE_LONG_CHAINED_DEPS;

      // Call next one for 0..(count-2)
      for (int i = 0; i < count - 1; ++i) {
        JavaCodeGeneratorHelper.targetWithNextHelper(i, true, projectPath);
        JavaCodeGeneratorHelper.buildFileWithNextDeps(
            i, "    deps=[ \":Deps" + (i + 1) + "\" ],\n", projectPath);
      }
      // Don't call next one for (count-1)
      JavaCodeGeneratorHelper.targetWithNextHelper(count - 1, false, projectPath);
      JavaCodeGeneratorHelper.buildFileWithNextDeps(count - 1, "", projectPath);

      JavaCodeGeneratorHelper.writeMainClassToDir("com.example.generated", projectPath);

      String deps = "    deps=[ \":Deps0\" ],\n";
      JavaCodeGeneratorHelper.buildFileWithMainClass(TARGET_LONG_CHAINED_DEPS, deps, projectPath);
    } catch (IOException e) {
      System.err.println(
          "Error creating targets with a few long chained dependencies: " + e.getMessage());
    }
  }

  /** Target type 3: Modify targets with a few long chained dependencies (A -> B -> C -> … -> Z) */
  private static void modifyTargetWithLongChainedDeps(Path projectPath) {
    File dir = projectPath.toFile();
    if (directoryExists(dir)) {
      System.err.format(
          "Project dir (%s) does not contain code for modification.\n", projectPath.toString());
      return;
    }
    try {
      JavaCodeGeneratorHelper.targetWithNextExtraHelper(
          (SIZE_LONG_CHAINED_DEPS + 1) >> 1, true, projectPath);
    } catch (IOException e) {
      System.err.println(
          "Error modifying targets with a few long chained dependencies: " + e.getMessage());
    }
  }

  /** Target type 4: Create targets with lots of parallel dependencies (A -> B, C, D, E, F, G, H) */
  private static void createTargetWithParallelDeps(Path projectPath) {
    if (pathExists(projectPath)) {
      return;
    }

    try {
      Files.createDirectories(projectPath);

      int count = SIZE_PARALLEL_DEPS;

      // parallel dependencies B~Z
      for (int i = 1; i < count; ++i) {
        JavaCodeGeneratorHelper.writeRandomClassToDir(
            false, "Deps" + i, "com.example.deps" + i, projectPath);
        JavaCodeGeneratorHelper.buildFileWithNextDeps(i, "", projectPath);
      }

      // A(Main)
      JavaCodeGeneratorHelper.parallelDepsMainClassHelper(count, projectPath);

      String deps = "    deps=[ ";
      for (int i = 1; i < count; ++i) {
        deps += "\":Deps" + i + "\", ";
      }
      deps += "], \n";
      JavaCodeGeneratorHelper.buildFileWithMainClass(TARGET_PARALLEL_DEPS, deps, projectPath);
    } catch (IOException e) {
      System.err.println(
          "Error creating targets with lots of parallel dependencies: " + e.getMessage());
    }
  }

  /** Target type 4: Modify targets with lots of parallel dependencies (A -> B, C, D, E, F, G, H) */
  private static void modifyTargetWithParallelDeps(Path projectPath) {
    File dir = projectPath.toFile();
    if (directoryExists(dir)) {
      System.err.format(
          "Project dir (%s) does not contain code for modification.\n", projectPath.toString());
      return;
    }
    try {
      JavaCodeGeneratorHelper.writeRandomClassToDir(
          true, "Deps1", "com.example.deps1", projectPath);
    } catch (IOException e) {
      System.err.println(
          "Error creating targets with lots of parallel dependencies: " + e.getMessage());
    }
  }

  private static boolean pathExists(Path path) {
    File dir = path.toFile();
    if (dir.exists()) {
      System.err.println("File or directory exists, not rewriting it: " + path);
      return true;
    }

    return false;
  }

  private static boolean directoryExists(File file) {
    return !(file.exists() && file.isDirectory());
  }
}
