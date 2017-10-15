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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link CppCodeGenerator}. */
@RunWith(JUnit4.class)
public class CppCodeGeneratorTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testGenerateNewProject() throws IOException {
    File createdFolder = folder.newFolder("GenerateNewProject");
    Path dir = createdFolder.toPath();
    CppCodeGenerator cppCodeGenerator = new CppCodeGenerator();
    cppCodeGenerator.generateNewProject(dir.toString(), ImmutableSet.of(
        CppCodeGenerator.TARGET_A_FEW_FILES,
        CppCodeGenerator.TARGET_LONG_CHAINED_DEPS,
        CppCodeGenerator.TARGET_MANY_FILES,
        CppCodeGenerator.TARGET_PARALLEL_DEPS
    ));

    // Check dir contains 4 project directories
    File[] filesList = dir.toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    assertThat(filenames).containsExactly(
        CppCodeGenerator.TARGET_A_FEW_FILES,
        CppCodeGenerator.TARGET_LONG_CHAINED_DEPS,
        CppCodeGenerator.TARGET_MANY_FILES,
        CppCodeGenerator.TARGET_PARALLEL_DEPS);

    // Target 1: a few files
    checkSimpleTarget(
        dir, CppCodeGenerator.TARGET_A_FEW_FILES, cppCodeGenerator.getSizeAFewFiles());

    // Target 2: many files
    checkSimpleTarget(dir, CppCodeGenerator.TARGET_MANY_FILES, cppCodeGenerator.getSizeManyFiles());

    // Target 3: long chained deps
    checkDepsTarget(
        dir, CppCodeGenerator.TARGET_LONG_CHAINED_DEPS, cppCodeGenerator.getSizeLongChainedDeps());

    // Target 4: parallel deps
    checkDepsTarget(
        dir, CppCodeGenerator.TARGET_PARALLEL_DEPS, cppCodeGenerator.getSizeParallelDeps());
  }

  private static ImmutableSet<String> fileArrayToImmutableSet(File[] files) {
    ImmutableSet.Builder<String> builder = ImmutableSet.builder();
    for (File file : files) {
      builder.add(file.getName());
    }
    return builder.build();
  }

  private static void checkSimpleTarget(Path root, String targetName, int targetSize) {
    // Check all files including BUILD, .cc, .h
    File[] filesList =
        root.resolve(targetName).toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    ImmutableSet.Builder<String> randomClassNames = ImmutableSet.builder();
    randomClassNames.add("BUILD");
    for (int i = 0; i < targetSize; ++i) {
      randomClassNames.add("RandomClass" + i + ".h");
      randomClassNames.add("RandomClass" + i + ".cc");
    }
    assertThat(filenames).containsExactlyElementsIn(randomClassNames.build());
  }

  private static void checkDepsTarget(Path root, String targetName, int targetSize) {
    // Check all files including BUILD, .cc, .h
    File[] filesList =
        root.resolve(targetName).toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    ImmutableSet.Builder<String> randomClassNames = ImmutableSet.builder();
    randomClassNames.add("BUILD");
    randomClassNames.add("Main.cc");
    for (int i = 1; i < targetSize; ++i) {
      randomClassNames.add("Deps" + i + ".h");
      randomClassNames.add("Deps" + i + ".cc");
    }
    assertThat(filenames).containsExactlyElementsIn(randomClassNames.build());
  }
}