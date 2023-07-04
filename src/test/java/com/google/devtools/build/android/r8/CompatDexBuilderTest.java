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

import com.android.tools.r8.D8;
import com.android.tools.r8.D8Command;
import com.android.tools.r8.OutputMode;
import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for CompatDexBuilder. */
@RunWith(JUnit4.class)
public class CompatDexBuilderTest {
  @Rule public TemporaryFolder temp = new TemporaryFolder();

  @Test
  public void compileManyClasses()
      throws IOException, InterruptedException, ExecutionException, OptionsParsingException {
    // Random set of classes from the R8 example test directory naming001.
    final String inputJar = System.getProperty("CompatDexBuilderTests.naming001");
    final ImmutableList<String> classNames =
        ImmutableList.of(
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "Reflect2$A",
            "Reflect2$B",
            "Reflect2",
            "Reflect");

    // Run CompatDexBuilder on naming001.jar
    Path outputZip = temp.getRoot().toPath().resolve("out.zip");
    CompatDexBuilder.main(
        new String[] {"--input_jar", inputJar, "--output_zip", outputZip.toString()});
    assertThat(Files.exists(outputZip)).isTrue();

    // Verify if all the classes have their corresponding ".class.dex" files in the zip.
    Set<String> expectedNames = new HashSet<>();
    for (String className : classNames) {
      expectedNames.add(
          "com/google/devtools/build/android/r8/testdata/naming001/" + className + ".class.dex");
    }
    try (ZipFile zipFile = new ZipFile(outputZip.toFile(), UTF_8)) {
      zipFile.stream()
          .forEach(
              ze -> {
                expectedNames.remove(ze.getName());
              });
    }
    assertThat(expectedNames).isEmpty();
  }

  @Test
  public void compileWithSyntheticLambdas() throws Exception {
    final String contextName = "com/google/devtools/build/android/r8/testdata/lambda/Lambda";
    final String inputJar = System.getProperty("CompatDexBuilderTests.lambda");
    final Path outputZip = temp.getRoot().toPath().resolve("out.zip");
    CompatDexBuilder.main(
        new String[] {"--input_jar", inputJar, "--output_zip", outputZip.toString()});
    assertThat(Files.exists(outputZip)).isTrue();

    try (ZipFile zipFile = new ZipFile(outputZip.toFile(), UTF_8)) {
      assertThat(zipFile.getEntry(contextName + ".class.dex")).isNotNull();
      ZipEntry entry = zipFile.getEntry("META-INF/synthetic-contexts.map");
      assertThat(entry).isNotNull();
      try (BufferedReader reader =
          new BufferedReader(new InputStreamReader(zipFile.getInputStream(entry), UTF_8))) {
        String line = reader.readLine();
        assertThat(line).isNotNull();
        // Format of mapping is: <synthetic-binary-name>;<context-binary-name>\n
        int sep = line.indexOf(';');
        String syntheticNameInMap = line.substring(0, sep);
        String contextNameInMap = line.substring(sep + 1);
        // The synthetic will be prefixed by the context type. This checks the synthetic name
        // is larger than the context to avoid hardcoding the synthetic names, which may change.
        assertThat(syntheticNameInMap).startsWith(contextName);
        assertThat(syntheticNameInMap).isNotEqualTo(contextName);
        // Check expected context.
        assertThat(contextNameInMap).isEqualTo(contextName);
        // Only one synthetic and its context should be present.
        line = reader.readLine();
        assertThat(line).isNull();
      }
    }
  }

  @Test
  public void compileTwoClassesAndRun() throws Exception {
    // Run CompatDexBuilder on dexMergeSample.jar
    final String inputJar = System.getProperty("CompatDexBuilderTests.twosimpleclasses");
    Path outputZip = temp.getRoot().toPath().resolve("out.zip");
    CompatDexBuilder.main(
        new String[] {"--input_jar", inputJar, "--output_zip", outputZip.toString()});

    // Merge zip content into a single dex file.
    Path d8OutDir = temp.newFolder().toPath();
    D8.run(
        D8Command.builder()
            .setOutput(d8OutDir, OutputMode.DexIndexed)
            .addProgramFiles(outputZip)
            .build());

    // TODO(sgjesse): Validate by running methods of Class1 and Class2.
    // https://r8.googlesource.com/r8/+/5ee92486c896b918efb62e69bff5dfa79f30e7c2/src/test/java/com/android/tools/r8/compatdexbuilder/CompatDexBuilderTests.java#95
  }
}
