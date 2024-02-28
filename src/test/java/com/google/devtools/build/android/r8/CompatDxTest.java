// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for CompatDx. */
@RunWith(JUnit4.class)
public class CompatDxTest {
  private static final String EXAMPLE_JAR_FILE_1 = System.getProperty("CompatDxTests.arithmetic");
  private static final String EXAMPLE_JAR_FILE_2 = System.getProperty("CompatDxTests.barray");

  private static final String NO_LOCALS = "--no-locals";
  private static final String NO_POSITIONS = "--positions=none";
  private static final String MULTIDEX = "--multi-dex";
  private static final String NUM_THREADS_5 = "--num-threads=5";

  @Rule public TemporaryFolder temp = new TemporaryFolder();

  @Test
  public void noFilesTest() throws IOException {
    assertThat(runDexer("--no-files")).isEmpty();
  }

  @Test
  public void noOutputTest() throws IOException {
    runDexerWithoutOutput(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void singleJarInputFile() throws IOException {
    assertThat(
            runDexer(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void multipleJarInputFiles() throws IOException {
    assertThat(
            runDexer(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1, EXAMPLE_JAR_FILE_2)
                .stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void outputZipFile() throws IOException {
    List<Path> out =
        runDexerWithOutput("foo.dex.zip", NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
    assertThat(out.stream().map(Path::getFileName).map(Path::toString))
        .containsExactly("foo.dex.zip");
    assertThat(archiveFiles(out.get(0))).containsExactly("classes.dex");
  }

  @Test
  public void useMultipleThreads() throws IOException {
    assertThat(
            runDexer(NUM_THREADS_5, NO_POSITIONS, NO_LOCALS, EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void withPositions() throws IOException {
    assertThat(
            runDexer(NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void withLocals() throws IOException {
    assertThat(
            runDexer(NO_POSITIONS, MULTIDEX, EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void withoutMultidex() throws IOException {
    assertThat(
            runDexer(NO_POSITIONS, NO_LOCALS, EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("classes.dex");
  }

  @Test
  public void writeToNamedDexFile() throws IOException {
    assertThat(
            runDexerWithOutput("named-output.dex", EXAMPLE_JAR_FILE_1).stream()
                .map(Path::getFileName)
                .map(Path::toString))
        .containsExactly("named-output.dex");
  }

  @Test
  public void keepClassesSingleDexTest() throws IOException {
    List<Path> out = runDexerWithOutput("out.zip", "--keep-classes", EXAMPLE_JAR_FILE_1);
    assertThat(out.stream().map(Path::getFileName).map(Path::toString)).containsExactly("out.zip");
    assertThat(archiveFiles(out.get(0)))
        .containsExactly(
            "classes.dex",
            "com/google/devtools/build/android/r8/testdata/arithmetic/Arithmetic.class");
  }

  @Test
  public void keepClassesMultiDexTest() throws IOException {
    List<Path> out =
        runDexerWithOutput("out.zip", "--keep-classes", "--multi-dex", EXAMPLE_JAR_FILE_1);
    assertThat(out.stream().map(Path::getFileName).map(Path::toString)).containsExactly("out.zip");
    assertThat(archiveFiles(out.get(0)))
        .containsExactly(
            "classes.dex",
            "com/google/devtools/build/android/r8/testdata/arithmetic/Arithmetic.class");
  }

  @Test
  public void ignoreDexInArchiveTest() throws IOException {
    // Create a JAR with both a .class and a .dex file (the .dex file is just empty).
    Path jarWithClassesAndDex = temp.newFile("test.jar").toPath();
    Files.copy(
        Paths.get(EXAMPLE_JAR_FILE_1), jarWithClassesAndDex, StandardCopyOption.REPLACE_EXISTING);
    jarWithClassesAndDex.toFile().setWritable(true);
    URI uri = URI.create("jar:" + jarWithClassesAndDex.toUri());
    try (FileSystem fileSystem =
        FileSystems.newFileSystem(uri, ImmutableMap.of("create", "true"))) {
      Path dexFile = fileSystem.getPath("classes.dex");
      Files.newOutputStream(dexFile, StandardOpenOption.CREATE).close();
    }

    // Only test this with CompatDx, as dx does not like the empty .dex file.
    List<String> d8Args =
        ImmutableList.of("--output=" + temp.newFolder("out"), jarWithClassesAndDex.toString());
    CompatDx.main(d8Args.toArray(new String[0]));
  }

  private List<Path> runDexer(String... args) throws IOException {
    return runDexerWithOutput("", args);
  }

  private void runDexerWithoutOutput(String... args) throws IOException {
    runDexerWithOutput(null, args);
  }

  private Path getOutputDir() throws IOException {
    Path dir = temp.getRoot().toPath().resolve("out");
    Files.createDirectory(dir);
    return dir;
  }

  private List<Path> runDexerWithOutput(String out, String... args) throws IOException {
    Path d8Out = null;
    if (out != null) {
      Path baseD8 = getOutputDir();
      d8Out = baseD8.resolve(out);
    }

    List<String> d8Args = new ArrayList<>();
    d8Args.add("--dex");
    if (d8Out != null) {
      d8Args.add("--output=" + d8Out);
    }
    Collections.addAll(d8Args, args);
    CompatDx.main(d8Args.toArray(new String[0]));

    if (out == null) {
      // Can't check output if explicitly not writing any.
      return ImmutableList.of();
    }

    List<Path> d8Files;
    try (Stream<Path> d8FilesStream =
        Files.list(Files.isDirectory(d8Out) ? d8Out : d8Out.getParent())) {
      d8Files = d8FilesStream.sorted().collect(toList());
    }
    return d8Files;
  }

  private static List<String> archiveFiles(Path path) throws IOException {
    try (ZipFile zipFile = new ZipFile(path.toFile(), UTF_8)) {
      return zipFile.stream().map(ZipEntry::getName).collect(toList());
    }
  }
}
