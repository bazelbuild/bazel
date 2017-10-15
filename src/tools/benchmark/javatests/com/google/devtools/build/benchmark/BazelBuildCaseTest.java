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

package com.google.devtools.build.benchmark;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableSet;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.Scanner;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BazelBuildCaseTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testPrepareGeneratedCode_Copy() throws IOException {
    Path root = folder.newFolder("PrepareGeneratedCodeCopy").toPath();
    // Prepare source
    Path source = root.resolve("source");
    source.toFile().mkdir();
    try (PrintWriter writer = new PrintWriter(source.resolve("file").toFile(), UTF_8.name())) {
      writer.println("content");
    }
    // Prepare destination
    Path destination = root.resolve("destination");
    destination.toFile().mkdir();

    new BazelBuildCase().prepareGeneratedCode(source, destination);

    ImmutableSet<String> filenames = fileArrayToImmutableSet(destination.toFile().listFiles());
    assertThat(filenames).containsExactly("WORKSPACE", "file");
    assertThat(
        new Scanner(destination.resolve("file")).useDelimiter("\\Z").next()).isEqualTo("content");
  }

  @Test
  public void testPrepareGeneratedCode_Generate() throws IOException {
    Path root = folder.newFolder("PrepareGeneratedCodeGenerate").toPath();
    // Prepare source, don't mkdir
    Path source = root.resolve("source");
    // Prepare destination
    Path destination = root.resolve("destination");
    destination.toFile().mkdir();

    new BazelBuildCase().prepareGeneratedCode(source, destination);

    // Check both source and destination directory include generated code
    ImmutableSet<String> sourceList = fileArrayToImmutableSet(source.toFile().listFiles());
    ImmutableSet<String> destinationList =
        fileArrayToImmutableSet(destination.toFile().listFiles());
    assertThat(sourceList)
        .containsExactly("cpp", "java");
    assertThat(destinationList)
        .containsExactly("cpp", "java", "WORKSPACE");

    ImmutableSet<String> targets = ImmutableSet.of(
        "AFewFiles", "LongChainedDeps", "ManyFiles", "ParallelDeps");
    ImmutableSet<String> sourceCppList =
        fileArrayToImmutableSet(source.resolve("cpp").toFile().listFiles());
    ImmutableSet<String> sourceJavaList =
        fileArrayToImmutableSet(source.resolve("java").toFile().listFiles());
    ImmutableSet<String> destinationCppList =
        fileArrayToImmutableSet(destination.resolve("cpp").toFile().listFiles());
    ImmutableSet<String> destinationJavaList =
        fileArrayToImmutableSet(destination.resolve("java").toFile().listFiles());
    assertThat(sourceCppList).containsExactlyElementsIn(targets);
    assertThat(sourceJavaList).containsExactlyElementsIn(targets);
    assertThat(destinationCppList).containsExactlyElementsIn(targets);
    assertThat(destinationJavaList).containsExactlyElementsIn(targets);
  }

  private static ImmutableSet<String> fileArrayToImmutableSet(File[] files) {
    ImmutableSet.Builder<String> fileNames = ImmutableSet.builder();
    for (File file : files) {
      fileNames.add(file.getName());
    }
    return fileNames.build();
  }
}
