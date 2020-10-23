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
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toSet;

import com.google.common.base.Joiner;
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
import java.util.Set;
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
    runDexer("--no-files");
  }

  @Test
  public void noOutputTest() throws IOException {
    runDexerWithoutOutput(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void singleJarInputFile() throws IOException {
    runDexer(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void multipleJarInputFiles() throws IOException {
    runDexer(NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1, EXAMPLE_JAR_FILE_2);
  }

  @Test
  public void outputZipFile() throws IOException {
    runDexerWithOutput("foo.dex.zip", NO_POSITIONS, NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void useMultipleThreads() throws IOException {
    runDexer(NUM_THREADS_5, NO_POSITIONS, NO_LOCALS, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void withPositions() throws IOException {
    runDexer(NO_LOCALS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void withLocals() throws IOException {
    runDexer(NO_POSITIONS, MULTIDEX, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void withoutMultidex() throws IOException {
    runDexer(NO_POSITIONS, NO_LOCALS, EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void writeToNamedDexFile() throws IOException {
    runDexerWithOutput("named-output.dex", EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void keepClassesSingleDexTest() throws IOException {
    runDexerWithOutput("out.zip", "--keep-classes", EXAMPLE_JAR_FILE_1);
  }

  @Test
  public void keepClassesMultiDexTest() throws IOException {
    runDexerWithOutput("out.zip", "--keep-classes", "--multi-dex", EXAMPLE_JAR_FILE_1);
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

  private void runDexer(String... args) throws IOException {
    runDexerWithOutput("", args);
  }

  private void runDexerWithoutOutput(String... args) throws IOException {
    runDexerWithOutput(null, args);
  }

  private Path getOutputD8() {
    return temp.getRoot().toPath().resolve("d8-out");
  }

  private Path getOutputDX() {
    return temp.getRoot().toPath().resolve("dx-out");
  }

  private void runDexerWithOutput(String out, String... args) throws IOException {
    Path d8Out = null;
    Path dxOut = null;
    if (out != null) {
      Path baseD8 = getOutputD8();
      Path baseDX = getOutputDX();
      Files.createDirectory(baseD8);
      Files.createDirectory(baseDX);
      d8Out = baseD8.resolve(out);
      dxOut = baseDX.resolve(out);
      assertThat(dxOut.toString()).isNotEqualTo(d8Out.toString());
    }

    List<String> d8Args = new ArrayList<>();
    d8Args.add("--dex");
    if (d8Out != null) {
      d8Args.add("--output=" + d8Out);
    }
    Collections.addAll(d8Args, args);
    System.out.println("running: d8 " + Joiner.on(" ").join(d8Args));
    CompatDx.main(d8Args.toArray(new String[0]));

    List<String> dxArgs = new ArrayList<>();
    dxArgs.add("--dex");
    if (dxOut != null) {
      dxArgs.add("--output=" + dxOut);
    }
    Collections.addAll(dxArgs, args);
    System.out.println("running: dx " + Joiner.on(" ").join(dxArgs));
    com.android.dx.command.Main.main(dxArgs.toArray(new String[0]));

    if (out == null) {
      // Can't check output if explicitly not writing any.
      return;
    }

    List<Path> d8Files;
    try (Stream<Path> d8FilesStream =
        Files.list(Files.isDirectory(d8Out) ? d8Out : d8Out.getParent())) {
      d8Files = d8FilesStream.sorted().collect(toList());
    }
    List<Path> dxFiles;
    try (Stream<Path> dxFilesStream =
        Files.list(Files.isDirectory(dxOut) ? dxOut : dxOut.getParent())) {
      dxFiles = dxFilesStream.sorted().collect(toList());
    }
    assertWithMessage("Out file names differ")
        .that(
            Joiner.on(System.lineSeparator())
                .join(d8Files.stream().map(Path::getFileName).iterator()))
        .isEqualTo(
            Joiner.on(System.lineSeparator())
                .join(dxFiles.stream().map(Path::getFileName).iterator()));
    for (int i = 0; i < d8Files.size(); i++) {
      if (FileUtils.isArchive(d8Files.get(i))) {
        compareArchiveFiles(d8Files.get(i), dxFiles.get(i));
      }
    }
  }

  private static void compareArchiveFiles(Path d8File, Path dxFile) throws IOException {
    ZipFile d8Zip = new ZipFile(d8File.toFile(), UTF_8);
    ZipFile dxZip = new ZipFile(dxFile.toFile(), UTF_8);
    // TODO(zerny): This should test resource containment too once supported.
    Set<String> d8Content = d8Zip.stream().map(ZipEntry::getName).collect(toSet());
    Set<String> dxContent =
        dxZip.stream()
            .map(ZipEntry::getName)
            .filter(
                name ->
                    name.endsWith(FileUtils.DEX_EXTENSION)
                        || name.endsWith(FileUtils.CLASS_EXTENSION))
            .collect(toSet());
    assertWithMessage("Expected dx and d8 output to contain same DEX anf class file entries")
        .that(d8Content)
        .containsExactlyElementsIn(dxContent);
  }
}
