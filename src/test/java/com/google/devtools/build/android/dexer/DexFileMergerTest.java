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

import com.android.dex.ClassDef;
import com.android.dex.Dex;
import com.android.dx.command.dexer.DxContext;
import com.android.dx.dex.code.PositionList;
import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexFileMerger}. */
@RunWith(JUnit4.class)
public class DexFileMergerTest {

  private static final Path INPUT_JAR;
  private static final Path INPUT_JAR2;
  private static final Path MAIN_DEX_LIST_FILE;
  static final String DEX_PREFIX = "classes";

  static {
    try {
      Runfiles runfiles = Runfiles.create();

      INPUT_JAR = Paths.get(runfiles.rlocation(System.getProperty("testinputjar")));
      INPUT_JAR2 = Paths.get(runfiles.rlocation(System.getProperty("testinputjar2")));
      MAIN_DEX_LIST_FILE = Paths.get(runfiles.rlocation(System.getProperty("testmaindexlist")));
    } catch (Exception e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /** Exercises DexFileMerger to write a single .dex file. */
  @Test
  public void testMergeDexArchive_singleOutputDex() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive = runDexFileMerger(dexArchive, 256 * 256, "from_dex_archive.dex.zip");

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "classes.dex");
  }

  @Test
  public void testMergeDexArchive_duplicateInputDeduped() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive = runDexFileMerger(
        256 * 256,
        /*forceJumbo=*/ false,
        "duplicate.dex.zip",
        MultidexStrategy.MINIMAL,
        /*mainDexList=*/ null,
        /*minimalMainDex=*/ false,
        DEX_PREFIX,
        dexArchive,
        dexArchive);  // input Jar twice to induce duplicates

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "classes.dex");
  }

  /** Similar to {@link #testMergeDexArchive_singleOutputDex} but uses --multidex=given_shard. */
  @Test
  public void testMergeDexArchive_givenShard() throws Exception {
    Path dexArchive = buildDexArchive(INPUT_JAR, "3.classes.jar");
    Path outputArchive =
        runDexFileMerger(
            256 * 256,
            /*forceJumbo=*/ false,
            "given_shard.dex.zip",
            MultidexStrategy.GIVEN_SHARD,
            /*mainDexList=*/ null,
            /*minimalMainDex=*/ false,
            DEX_PREFIX,
            dexArchive);

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "classes3.dex");
  }

  /**
   * Similar to {@link #testMergeDexArchive_singleOutputDex} but different name for output dex file.
   */
  @Test
  public void testMergeDexArchive_singleOutputPrefixDex() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive =
        runDexFileMerger(
            256 * 256,
            /*forceJumbo=*/ false,
            "prefix.dex.zip",
            MultidexStrategy.MINIMAL,
            /*mainDexList=*/ null,
            /*minimalMainDex=*/ false,
            "noname",
            dexArchive);

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "noname.dex");
  }

  /** Exercises DexFileMerger with two input archives. */
  @Test
  public void testMergeDexArchive_multipleInputs() throws Exception {
    Path dexArchive = buildDexArchive();
    Path dexArchive2 = buildDexArchive(INPUT_JAR2, "libtestdata.jar.dex.zip");
    Path outputArchive =
        runDexFileMerger(
            256 * 256,
            /*forceJumbo=*/ false,
            "multiple_inputs.dex.zip",
            MultidexStrategy.MINIMAL,
            /*mainDexList=*/ null,
            /*minimalMainDex=*/ false,
            DEX_PREFIX,
            dexArchive,
            dexArchive2);

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    expectedClassCount += matchingFileCount(dexArchive2, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "classes.dex");
  }

  /**
   * Similar to {@link #testMergeDexArchive_singleOutputDex} but forces multiple output dex files.
   */
  @Test
  public void testMergeDexArchive_multidex() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive = runDexFileMerger(dexArchive, 200, "multidex_from_dex_archive.dex.zip");

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertMultidexOutput(expectedClassCount, outputArchive, ImmutableSet.<String>of());
  }

  @Test
  public void testMergeDexArchive_mainDexList() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive =
        runDexFileMerger(
            200,
            /*forceJumbo=*/ false,
            "main_dex_list.dex.zip",
            MultidexStrategy.MINIMAL,
            MAIN_DEX_LIST_FILE,
            /*minimalMainDex=*/ false,
            DEX_PREFIX,
            dexArchive);

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertMainDexOutput(expectedClassCount, outputArchive, false);
  }

  @Test
  public void testMergeDexArchive_minimalMainDex() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive =
        runDexFileMerger(
            256 * 256,
            /*forceJumbo=*/ false,
            "minimal_main_dex.dex.zip",
            MultidexStrategy.MINIMAL,
            MAIN_DEX_LIST_FILE,
            /*minimalMainDex=*/ true,
            DEX_PREFIX,
            dexArchive);

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertMainDexOutput(expectedClassCount, outputArchive, true);
  }

  @Test
  public void testMultidexOffWithMultidexFlags() throws Exception {
    Path dexArchive = buildDexArchive();
    try {
      runDexFileMerger(
          200,
          /*forceJumbo=*/ false,
          "classes.dex.zip",
          MultidexStrategy.OFF,
          /*mainDexList=*/ null,
          /*minimalMainDex=*/ true,
          DEX_PREFIX,
          dexArchive);
      fail("Expected DexFileMerger to fail");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "--minimal-main-dex is only supported with multidex enabled, but mode is: OFF");
    }
    try {
      runDexFileMerger(
          200,
          /*forceJumbo=*/ false,
          "classes.dex.zip",
          MultidexStrategy.OFF,
          MAIN_DEX_LIST_FILE,
          /*minimalMainDex=*/ false,
          DEX_PREFIX,
          dexArchive);
      fail("Expected DexFileMerger to fail");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("--main-dex-list is only supported with multidex enabled, but mode is: OFF");
    }
  }

  /** Exercises --forceJumbo support. */
  @Test
  public void testMergeDexArchive_forceJumbo() throws Exception {
    Path dexArchive = buildDexArchive();
    Path outputArchive;
    try {
      outputArchive = runDexFileMerger(256 * 256, /*forceJumbo=*/ true, "from_dex_archive.dex.zip",
          MultidexStrategy.OFF, /*mainDexList=*/ null, /*minimalMainDex=*/ false, DEX_PREFIX,
          dexArchive);
    } catch (IllegalStateException e) {
      assertThat(e).hasMessageThat().isEqualTo("--forceJumbo flag not supported");
      System.err.println("Skipping this test due to missing --forceJumbo support in Android SDK.");
      e.printStackTrace();
      return;
    }

    int expectedClassCount = matchingFileCount(dexArchive, ".*\\.class.dex$");
    assertSingleDexOutput(expectedClassCount, outputArchive, "classes.dex");
  }

  private void assertSingleDexOutput(int expectedClassCount, Path outputArchive, String dexFileName)
      throws IOException {
    try (ZipFile output = new ZipFile(outputArchive.toFile())) {
      ZipEntry entry = Iterators.getOnlyElement(Iterators.forEnumeration(output.entries()));
      assertThat(entry.getName()).isEqualTo(dexFileName);
      Dex dex = new Dex(output.getInputStream(entry));
      assertThat(dex.classDefs()).hasSize(expectedClassCount);
    }
  }

  private Multimap<String, String> assertMultidexOutput(int expectedClassCount,
      Path outputArchive, Set<String> mainDexList) throws IOException {
    SetMultimap<String, String> dexFiles = HashMultimap.create();
    try (ZipFile output = new ZipFile(outputArchive.toFile())) {
      Enumeration<? extends ZipEntry> entries = output.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        assertThat(entry.getName()).containsMatch("classes[2-9]?.dex");
        Dex dex = new Dex(output.getInputStream(entry));
        for (ClassDef clazz : dex.classDefs()) {
          dexFiles.put(entry.getName(),
              toSlashedClassName(dex.typeNames().get(clazz.getTypeIndex())));
        }
      }
    }
    assertThat(dexFiles.keySet().size()).isAtLeast(2);
    assertThat(dexFiles.size()).isAtLeast(1);
    assertThat(dexFiles).hasSize(expectedClassCount);
    for (int i = 0; i < dexFiles.keySet().size(); ++i) {
      assertThat(dexFiles).containsKey(expectedDexFileName(i));
    }
    for (int i = 1; i < dexFiles.keySet().size(); ++i) {
      Set<String> prev = dexFiles.get(expectedDexFileName(i - 1));
      if (i == 1) {
        prev = Sets.difference(prev, mainDexList);
      }
      Set<String> shard = dexFiles.get(expectedDexFileName(i));
      for (String c1 : prev) {
        for (String c2 : shard) {
          assertWithMessage(
                  c2
                      + " in shard "
                      + i
                      + " should compare as larger than "
                      + c1
                      + "; list of all shards for reference: "
                      + dexFiles)
              .that(ZipEntryComparator.compareClassNames(c2, c1))
              .isGreaterThan(0);
        }
      }
    }
    return dexFiles;
  }

  private static String expectedDexFileName(int i) {
    return DEX_PREFIX + (i == 0 ? "" : i + 1) + ".dex";
  }

  private void assertMainDexOutput(int expectedClassCount, Path outputArchive,
      boolean minimalMainDex) throws IOException {
    HashSet<String> mainDexList = new HashSet<>();
    for (String filename : Files.readAllLines(MAIN_DEX_LIST_FILE, UTF_8)) {
      mainDexList.add(
          filename.endsWith(".class") ? filename.substring(0, filename.length() - 6) : filename);
    }
    Multimap<String, String> dexFiles =
        assertMultidexOutput(expectedClassCount, outputArchive, mainDexList);
    assertThat(dexFiles.keySet().size()).isAtLeast(2);
    if (minimalMainDex) {
      assertThat(dexFiles.get("classes.dex")).containsExactlyElementsIn(mainDexList);
    } else {
      assertThat(dexFiles.get("classes.dex")).containsAtLeastElementsIn(mainDexList);
    }
  }

  /** Converts signature classes, eg., "Lpath/to/Class;", to regular names like "path/to/Class". */
  private static String toSlashedClassName(String signatureClassname) {
    return signatureClassname.substring(1, signatureClassname.length() - 1);
  }

  private int matchingFileCount(Path dexArchive, String filenameFilter) throws IOException {
    try (ZipFile input = new ZipFile(dexArchive.toFile())) {
      return Iterators.size(Iterators.filter(
          Iterators.transform(Iterators.forEnumeration(input.entries()), ZipEntryName.INSTANCE),
          Predicates.containsPattern(filenameFilter)));
    }
  }

  private Path runDexFileMerger(Path dexArchive, int maxNumberOfIdxPerDex, String outputBasename)
      throws IOException {
    return runDexFileMerger(
        maxNumberOfIdxPerDex,
        /*forceJumbo=*/ false,
        outputBasename,
        MultidexStrategy.MINIMAL,
        /*mainDexList=*/ null,
        /*minimalMainDex=*/ false,
        DEX_PREFIX,
        dexArchive);
  }

  private Path runDexFileMerger(
      int maxNumberOfIdxPerDex,
      boolean forceJumbo,
      String outputBasename,
      MultidexStrategy multidexMode,
      @Nullable Path mainDexList,
      boolean minimalMainDex,
      String dexPrefix,
      Path... dexArchives)
      throws IOException {
    DexFileMerger.Options options = new DexFileMerger.Options();
    options.inputArchives = ImmutableList.copyOf(dexArchives);
    options.outputArchive =
        FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR"), outputBasename);
    options.multidexMode = multidexMode;
    options.maxNumberOfIdxPerDex = maxNumberOfIdxPerDex;
    options.forceJumbo = forceJumbo;
    options.mainDexListFile = mainDexList;
    options.minimalMainDex = minimalMainDex;
    options.dexPrefix = dexPrefix;
    DexFileMerger.buildMergedDexFiles(options);
    assertThat(options.outputArchive.toFile().exists()).isTrue();
    return options.outputArchive;
  }

  private Path buildDexArchive() throws Exception {
    return buildDexArchive(INPUT_JAR, "libtests.dex.zip");
  }

  private Path buildDexArchive(Path inputJar, String outputZip) throws Exception {
    DexBuilder.Options options = new DexBuilder.Options();
    options.inputJar = inputJar;
    options.outputZip = FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR"), outputZip);
    options.maxThreads = 1;
    Dexing.DexingOptions dexingOptions = new Dexing.DexingOptions();
    dexingOptions.optimize = true;
    dexingOptions.positionInfo = PositionList.LINES;
    DexBuilder.buildDexArchive(options, new Dexing(new DxContext(), dexingOptions));
    return options.outputZip;
  }

  private enum ZipEntryName implements Function<ZipEntry, String> {
    INSTANCE;
    @Override
    public String apply(ZipEntry input) {
      return input.getName();
    }
  }
}
