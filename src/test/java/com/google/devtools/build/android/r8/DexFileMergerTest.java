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

import com.android.tools.r8.CompilationFailedException;
import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.DexFileMergerHelper;
import com.android.tools.r8.ExtractMarker;
import com.android.tools.r8.OutputMode;
import com.android.tools.r8.errors.CompilationError;
import com.google.common.io.ByteStreams;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

// import com.android.tools.r8.maindexlist.MainDexListTests;

/** Test for DexFileMerger. */
@RunWith(JUnit4.class)
public class DexFileMergerTest {

  @Rule public TemporaryFolder temp = new TemporaryFolder();

  private static final String CLASSES_JAR = System.getProperty("DexFileMergerTest.dexmergersample");
  private String class1Class;
  private String class2Class;

  @Before
  public void setUp() throws Exception {
    File jarUnzipFolder = temp.newFolder();
    unzip(Paths.get(CLASSES_JAR).toString(), jarUnzipFolder, entry -> true);
    class1Class =
        jarUnzipFolder
            + "/com/google/devtools/build/android/r8/testdata/dexmergersample/Class1.class";
    class2Class =
        jarUnzipFolder
            + "/com/google/devtools/build/android/r8/testdata/dexmergersample/Class2.class";
  }

  private Path compileTwoClasses(OutputMode outputMode, boolean addMarker)
      throws CompilationFailedException, IOException {
    // Compile Class1 and Class2.
    Path output = temp.newFolder().toPath().resolve("compiled.zip");
    D8Command command =
        D8Command.builder()
            .setOutput(output, outputMode)
            .addProgramFiles(Paths.get(class2Class))
            .addProgramFiles(Paths.get(class1Class))
            .build();

    DexFileMergerHelper.runD8ForTesting(command, !addMarker);

    return output;
  }

  private Path mergeWithDexMerger(Path input) throws Exception {
    Path output = temp.newFolder().toPath().resolve("merged-with-dexmerger.zip");
    DexFileMerger.main(new String[] {"--input", input.toString(), "--output", output.toString()});
    return output;
  }

  private Path mergeWithD8(Path input) throws Exception {
    Path output = temp.newFolder().toPath().resolve("merged-with-d8.zip");
    D8.main(new String[] {input.toString(), "--output", output.toString()});
    return output;
  }

  @Test
  public void markerPreserved() throws Exception {
    Path input = compileTwoClasses(OutputMode.DexIndexed, true);
    assertThat(ExtractMarker.extractMarkerFromDexFile(input)).hasSize(1);
    assertThat(ExtractMarker.extractMarkerFromDexFile(mergeWithDexMerger(input))).hasSize(1);
    assertThat(ExtractMarker.extractMarkerFromDexFile(mergeWithD8(input))).hasSize(1);
  }

  @Test
  public void markerNotAdded() throws Exception {
    Path input = compileTwoClasses(OutputMode.DexIndexed, false);
    assertThat(ExtractMarker.extractMarkerFromDexFile(input)).isEmpty();
    assertThat(ExtractMarker.extractMarkerFromDexFile(mergeWithDexMerger(input))).isEmpty();
    assertThat(ExtractMarker.extractMarkerFromDexFile(mergeWithD8(input))).isEmpty();
  }

  @Test
  public void mergeTwoFiles() throws CompilationFailedException, IOException {
    Path mergerInputZip = compileTwoClasses(OutputMode.DexFilePerClassFile, false);

    Path mergerOutputZip = temp.getRoot().toPath().resolve("merger-out.zip");
    DexFileMerger.main(
        new String[] {
          "--input", mergerInputZip.toString(), "--output", mergerOutputZip.toString()
        });

    // TODO(sgjesse): Validate by running methods of Class1 and Class2.
    // https://r8.googlesource.com/r8/+/5ee92486c896b918efb62e69bff5dfa79f30e7c2/src/test/java/com/android/tools/r8/dexfilemerger/DexFileMergerTests.java#103
  }

  // TODO(sgjesse): Port tests for merge overflow.
  // https://r8.googlesource.com/r8/+/5ee92486c896b918efb62e69bff5dfa79f30e7c2/src/test/java/com/android/tools/r8/dexfilemerger/DexFileMergerTests.java#131

  // Copied from R8 class com.android.tools.r8.utils.ZipUtils.
  private interface OnEntryHandler {
    void onEntry(ZipEntry entry, InputStream input) throws IOException;
  }

  // Copied from R8 class com.android.tools.r8.utils.ZipUtils.
  private static void iter(String zipFileStr, OnEntryHandler handler) throws IOException {
    try (ZipFile zipFile = new ZipFile(zipFileStr, UTF_8)) {
      zipFile.stream()
          .forEach(
              entry -> {
                try (InputStream entryStream = zipFile.getInputStream(entry)) {
                  handler.onEntry(entry, entryStream);
                } catch (IOException e) {
                  throw new AssertionError(e);
                }
              });
    }
  }

  // Copied from R8 class com.android.tools.r8.utils.ZipUtils.
  private static List<File> unzip(String zipFile, File outDirectory, Predicate<ZipEntry> filter)
      throws IOException {
    final Path outDirectoryPath = outDirectory.toPath();
    final List<File> outFiles = new ArrayList<>();
    iter(
        zipFile,
        (entry, input) -> {
          String name = entry.getName();
          if (!entry.isDirectory() && filter.test(entry)) {
            if (name.contains("..")) {
              // Protect against malicious archives.
              throw new CompilationError("Invalid entry name \"" + name + "\"");
            }
            Path outPath = outDirectoryPath.resolve(name);
            File outFile = outPath.toFile();
            outFile.getParentFile().mkdirs();
            System.out.println(outFile);
            try (OutputStream output = new FileOutputStream(outFile)) {
              ByteStreams.copy(input, output);
            }
            outFiles.add(outFile);
          }
        });
    return outFiles;
  }
}
