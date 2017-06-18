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

/** Test for {@link JavaCodeGenerator}. */
@RunWith(JUnit4.class)
public class JavaCodeGeneratorTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testGenerateNewProject() throws IOException {
    File createdFolder = folder.newFolder("GenerateNewProject");
    Path dir = createdFolder.toPath();
    JavaCodeGenerator javaCodeGenerator = new JavaCodeGenerator();
    javaCodeGenerator.generateNewProject(dir.toString(), ImmutableSet.of(
        JavaCodeGenerator.TARGET_A_FEW_FILES,
        JavaCodeGenerator.TARGET_LONG_CHAINED_DEPS,
        JavaCodeGenerator.TARGET_MANY_FILES,
        JavaCodeGenerator.TARGET_PARALLEL_DEPS));

    // Check dir contains 4 project directories
    File[] filesList = dir.toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    assertThat(filenames).containsExactly(
        JavaCodeGenerator.TARGET_A_FEW_FILES,
        JavaCodeGenerator.TARGET_LONG_CHAINED_DEPS,
        JavaCodeGenerator.TARGET_MANY_FILES,
        JavaCodeGenerator.TARGET_PARALLEL_DEPS);

    // Target 1: a few files
    checkProjectPathContains(dir, JavaCodeGenerator.TARGET_A_FEW_FILES);
    checkSimpleTarget(
        dir, JavaCodeGenerator.TARGET_A_FEW_FILES, javaCodeGenerator.getSizeAFewFiles());

    // Target 2: many files
    checkProjectPathContains(dir, JavaCodeGenerator.TARGET_MANY_FILES);
    checkSimpleTarget(
        dir, JavaCodeGenerator.TARGET_MANY_FILES, javaCodeGenerator.getSizeManyFiles());

    // Target 3: long chained deps
    checkProjectPathContains(dir, JavaCodeGenerator.TARGET_LONG_CHAINED_DEPS);
    checkDepsTarget(
        dir, JavaCodeGenerator.TARGET_LONG_CHAINED_DEPS,
        javaCodeGenerator.getSizeLongChainedDeps());

    // Target 4: parallel deps
    checkProjectPathContains(dir, JavaCodeGenerator.TARGET_PARALLEL_DEPS);
    checkDepsTarget(
        dir, JavaCodeGenerator.TARGET_PARALLEL_DEPS, javaCodeGenerator.getSizeParallelDeps());
  }

  private static ImmutableSet<String> fileArrayToImmutableSet(File[] files) {
    ImmutableSet.Builder<String> builder = ImmutableSet.builder();
    for (File file : files) {
      builder.add(file.getName());
    }
    return builder.build();
  }

  private static void checkProjectPathContains(Path root, String targetName) {
    // Check project dir contains BUILD and com
    File[] filesList = root.resolve(targetName).toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    assertThat(filenames).containsExactly("BUILD", "com");

    // Check project dir contains com/example
    filesList = root.resolve(targetName).resolve("com").toFile().listFiles();
    assertThat(filesList).isNotNull();
    filenames = fileArrayToImmutableSet(filesList);
    assertThat(filenames).containsExactly("example");
  }

  private static void checkSimpleTarget(Path root, String targetName, int targetSize) {
    // Check Java files
    File[] filesList =
        root.resolve(targetName).resolve("com/example/generated").toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    ImmutableSet.Builder<String> randomClassNames = ImmutableSet.builder();
    randomClassNames.add("Main.java");
    for (int i = 0; i < targetSize; ++i) {
      randomClassNames.add("RandomClass" + i + ".java");
    }
    assertThat(filenames).containsExactlyElementsIn(randomClassNames.build());
  }

  private static void checkDepsTarget(Path root, String targetName, int targetSize) {
    // Check Java files
    for (int i = 1; i < targetSize; ++i) {
      File[] filesList =
          root.resolve(targetName).resolve("com/example/deps" + i).toFile().listFiles();
      assertThat(filesList).isNotNull();
      ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
      assertThat(filenames).containsExactly("Deps" + i + ".java");
    }
    File[] filesList =
        root.resolve(targetName).resolve("com/example/generated").toFile().listFiles();
    assertThat(filesList).isNotNull();
    ImmutableSet<String> filenames = fileArrayToImmutableSet(filesList);
    assertThat(filenames).containsExactly("Main.java");
  }
}
