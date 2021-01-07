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
package com.google.devtools.build.android.dexer;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.code.PositionList;
import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexFileSplitter}. */
@RunWith(JUnit4.class)
public class DexFileSplitterTest {

  private static final Path INPUT_JAR;
  private static final Path INPUT_JAR2;
  private static final Path MIXED_JAR;
  private static final Path MAIN_DEX_LIST_FILE;
  static final String DEX_PREFIX = "classes";

  static {
    try {
      Runfiles runfiles = Runfiles.create();

      INPUT_JAR = Paths.get(runfiles.rlocation(System.getProperty("testinputjar")));
      INPUT_JAR2 = Paths.get(runfiles.rlocation(System.getProperty("testinputjar2")));
      MIXED_JAR = Paths.get(runfiles.rlocation(System.getProperty("mixedinputjar")));
      MAIN_DEX_LIST_FILE = Paths.get(runfiles.rlocation(System.getProperty("testmaindexlist")));
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Test
  public void testSingleInputSingleOutput() throws Exception {
    Path dexArchive = buildDexArchive();
    ImmutableList<Path> outputArchives = runDexSplitter(256 * 256, "from_single", dexArchive);
    assertThat(outputArchives).hasSize(1);

    ImmutableSet<String> expectedFiles = dexEntries(dexArchive);
    assertThat(dexEntries(outputArchives.get(0))).containsExactlyElementsIn(expectedFiles);
  }

  @Test
  public void testDuplicateInputIgnored() throws Exception {
    Path dexArchive = buildDexArchive();
    ImmutableList<Path> outputArchives =
        runDexSplitter(256 * 256, "from_duplicate", dexArchive, dexArchive);
    assertThat(outputArchives).hasSize(1);

    ImmutableSet<String> expectedFiles = dexEntries(dexArchive);
    assertThat(dexEntries(outputArchives.get(0))).containsExactlyElementsIn(expectedFiles);
  }

  @Test
  public void testSingleInputMultidexOutput() throws Exception {
    Path dexArchive = buildDexArchive();
    ImmutableList<Path> outputArchives = runDexSplitter(200, "multidex_from_single", dexArchive);
    assertThat(outputArchives.size()).isGreaterThan(1);

    ImmutableSet<String> expectedEntries = dexEntries(dexArchive);
    assertExpectedEntries(outputArchives, expectedEntries);
  }

  @Test
  public void testMultipleInputsMultidexOutput() throws Exception {
    Path dexArchive = buildDexArchive();
    Path dexArchive2 = buildDexArchive(INPUT_JAR2, "jar2.dex.zip");
    ImmutableList<Path> outputArchives = runDexSplitter(200, "multidex", dexArchive, dexArchive2);
    assertThat(outputArchives.size()).isGreaterThan(1);

    HashSet<String> expectedEntries = new HashSet<>();
    expectedEntries.addAll(dexEntries(dexArchive));
    expectedEntries.addAll(dexEntries(dexArchive2));
    assertExpectedEntries(outputArchives, expectedEntries);
  }

  /**
   * Tests that the same input creates identical output in 2 runs.  Flakiness here would indicate
   * race conditions or other concurrency issues.
   */
  @Test
  public void testDeterminism() throws Exception {
    Path dexArchive = buildDexArchive();
    Path dexArchive2 = buildDexArchive(INPUT_JAR2, "jar2.dex.zip");
    ImmutableList<Path> outputArchives = runDexSplitter(200, "det1", dexArchive, dexArchive2);
    assertThat(outputArchives.size()).isGreaterThan(1);
    ImmutableList<Path> outputArchives2 = runDexSplitter(200, "det2", dexArchive, dexArchive2);
    assertThat(outputArchives2).hasSize(outputArchives.size()); // paths differ though

    Path outputRoot2 = outputArchives2.get(0).getParent();
    for (Path outputArchive : outputArchives) {
      ImmutableList<ZipEntry> expectedEntries;
      try (ZipFile zip = new ZipFile(outputArchive.toFile())) {
        expectedEntries = zip.stream().collect(ImmutableList.<ZipEntry>toImmutableList());
      }
      ImmutableList<ZipEntry> actualEntries;
      try (ZipFile zip2 = new ZipFile(outputRoot2.resolve(outputArchive.getFileName()).toFile())) {
        actualEntries = zip2.stream().collect(ImmutableList.<ZipEntry>toImmutableList());
      }
      int len = expectedEntries.size();
      assertThat(actualEntries).hasSize(len);
      for (int i = 0; i < len; ++i) {
        ZipEntry expected = expectedEntries.get(i);
        ZipEntry actual = actualEntries.get(i);
        assertWithMessage(actual.getName()).that(actual.getName()).isEqualTo(expected.getName());
        assertWithMessage(actual.getName()).that(actual.getSize()).isEqualTo(expected.getSize());
        assertWithMessage(actual.getName()).that(actual.getCrc()).isEqualTo(expected.getCrc());
      }
    }
  }

  @Test
  public void testMainDexList() throws Exception {
    Path dexArchive = buildDexArchive();
    ImmutableList<Path> outputArchives =
        runDexSplitter(
            200,
            /*inclusionFilterJar=*/ null,
            "main_dex_list",
            MAIN_DEX_LIST_FILE,
            /*minimalMainDex=*/ false,
            dexArchive);

    ImmutableSet<String> expectedEntries = dexEntries(dexArchive);
    assertThat(outputArchives.size()).isGreaterThan(1);
    assertThat(dexEntries(outputArchives.get(0)))
        .containsAtLeastElementsIn(expectedMainDexEntries());
    assertExpectedEntries(outputArchives, expectedEntries);
  }

  @Test
  public void testMainDexList_containsForbidden() throws Exception {
    Path dexArchive = buildDexArchive();
    Path mainDexFile = Files.createTempFile("main_dex_list", ".txt");
    Files.write(mainDexFile, ImmutableList.of("com/google/Ok.class", "j$/my/Bad.class"), UTF_8);
    try {
      runDexSplitter(
          256 * 256,
          /*inclusionFilterJar=*/ null,
          "invalid_main_dex_list",
          mainDexFile,
          /*minimalMainDex=*/ false,
          dexArchive);
      fail("IllegalArgumentException expected");
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessageThat().contains("j$");
    }
  }

  @Test
  public void testMinimalMainDex() throws Exception {
    Path dexArchive = buildDexArchive();
    ImmutableList<Path> outputArchives =
        runDexSplitter(
            256 * 256,
            /*inclusionFilterJar=*/ null,
            "minimal_main_dex",
            MAIN_DEX_LIST_FILE,
            /*minimalMainDex=*/ true,
            dexArchive);

    ImmutableSet<String> expectedEntries = dexEntries(dexArchive);
    assertThat(outputArchives.size()).isGreaterThan(1);
    assertThat(dexEntries(outputArchives.get(0)))
        .containsExactlyElementsIn(expectedMainDexEntries());
    assertExpectedEntries(outputArchives, expectedEntries);
  }

  @Test
  public void testInclusionFilterJar() throws Exception {
    Path dexArchive = buildDexArchive();
    Path dexArchive2 = buildDexArchive(INPUT_JAR2, "jar2.dex.zip");
    ImmutableList<Path> outputArchives =
        runDexSplitter(
            256 * 256,
            INPUT_JAR2,
            "filtered",
            /*mainDexList=*/ null,
            /*minimalMainDex=*/ false,
            dexArchive,
            dexArchive2);

    // Only expect entries from the Jar we filtered by
    assertExpectedEntries(outputArchives, dexEntries(dexArchive2));
  }

  @Test
  public void testMixedInput_keptSeparate() throws Exception {
    Path dexArchive = buildDexArchive();
    Path mixedArchive = buildDexArchive(MIXED_JAR, "mixed.jar.dex.zip");
    ImmutableList<Path> outputArchives =
        runDexSplitter(256 * 256, "mixed_input", dexArchive, mixedArchive);
    assertThat(outputArchives).hasSize(3);
    assertThat(dexEntries(outputArchives.get(0))).contains("aaa/Baz.class.dex");
    assertThat(dexEntries(outputArchives.get(1))).containsExactly("j$/test/Foo.class.dex");
    assertThat(dexEntries(outputArchives.get(2))).containsExactly("zzz/Bar.class.dex");
  }

  private static Iterable<String> expectedMainDexEntries() throws IOException {
    return Iterables.transform(
        Files.readAllLines(MAIN_DEX_LIST_FILE),
        new Function<String, String>() {
          @Override
          public String apply(String input) {
            return input + ".dex";
          }
        });
  }

  @Test
  public void testMultidexOffWithMultidexFlags() throws Exception {
    Path dexArchive = buildDexArchive();
    try {
      runDexSplitter(
          200,
          /*inclusionFilterJar=*/ null,
          "should_fail",
          /*mainDexList=*/ null,
          /*minimalMainDex=*/ true,
          dexArchive);
      fail("Expected IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("--minimal-main-dex not allowed without --main-dex-list");
    }
  }

  private void assertExpectedEntries(
      ImmutableList<Path> outputArchives, Set<String> expectedEntries) throws IOException {
    ImmutableSet.Builder<String> actualFiles = ImmutableSet.builder();
    for (Path outputArchive : outputArchives) {
      actualFiles.addAll(dexEntries(outputArchive));
    }
    // ImmutableSet.Builder.build would fail if there were duplicates.  Additionally we make sure
    // all expected files are here
    assertThat(actualFiles.build()).containsExactlyElementsIn(expectedEntries);
  }

  private ImmutableSet<String> dexEntries(Path dexArchive) throws IOException {
    try (ZipFile input = new ZipFile(dexArchive.toFile())) {
      ImmutableSet<String> result =
          input.stream()
              .map(ZipEntryName.INSTANCE)
              .filter(Predicates.containsPattern(".*\\.class.dex$"))
              .collect(ImmutableSet.<String>toImmutableSet());
      assertThat(result).isNotEmpty();
      return result;
    }
  }

  private ImmutableList<Path> runDexSplitter(int maxNumberOfIdxPerDex, String outputRoot,
      Path... dexArchives) throws IOException {
    return runDexSplitter(
        maxNumberOfIdxPerDex,
        /*inclusionFilterJar=*/ null,
        outputRoot,
        /*mainDexList=*/ null,
        /*minimalMainDex=*/ false,
        dexArchives);
  }

  private ImmutableList<Path> runDexSplitter(
      int maxNumberOfIdxPerDex,
      @Nullable Path inclusionFilterJar,
      String outputRoot,
      @Nullable Path mainDexList,
      boolean minimalMainDex,
      Path... dexArchives)
      throws IOException {
    DexFileSplitter.Options options = new DexFileSplitter.Options();
    options.inputArchives = ImmutableList.copyOf(dexArchives);
    options.outputDirectory =
        FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR"), outputRoot);
    options.maxNumberOfIdxPerDex = maxNumberOfIdxPerDex;
    options.mainDexListFile = mainDexList;
    options.minimalMainDex = minimalMainDex;
    options.inclusionFilterJar = inclusionFilterJar;
    DexFileSplitter.splitIntoShards(options);
    assertThat(options.outputDirectory.toFile().exists()).isTrue();
    ImmutableSet<Path> files = readFiles(options.outputDirectory, "*.zip");

    ImmutableList.Builder<Path> result = ImmutableList.builder();
    for (int i = 1; i <= files.size(); ++i) {
      Path path = options.outputDirectory.resolve(i + ".shard.zip");
      assertThat(files).contains(path);
      result.add(path);
    }
    return result.build(); // return expected files in sorted order
  }

  private static ImmutableSet<Path> readFiles(Path directory, String glob) throws IOException {
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(directory, glob)) {
      return ImmutableSet.copyOf(stream);
    }
  }

  private Path buildDexArchive() throws Exception {
    return buildDexArchive(INPUT_JAR, "libtests.dex.zip");
  }

  private Path buildDexArchive(Path inputJar, String outputZip) throws Exception {
    DexBuilder.Options options = new DexBuilder.Options();
    // Use Jar file that has this test in it as the input Jar
    options.inputJar = inputJar;
    options.outputZip =
        FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR"), outputZip);
    options.maxThreads = 1;
    Dexing.DexingOptions dexingOptions = new Dexing.DexingOptions();
    dexingOptions.optimize = true;
    dexingOptions.positionInfo = PositionList.LINES;
    DexBuilder.buildDexArchive(options, new Dexing(new DxContext(), dexingOptions));
    return options.outputZip;
  }

  // Can't use lambda for Java 7 compatibility so we can run this Jar through dx without desugaring.
  private enum ZipEntryName implements Function<ZipEntry, String> {
    INSTANCE;
    @Override
    public String apply(ZipEntry input) {
      return input.getName();
    }
  }
}
