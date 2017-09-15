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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.PatternFilenameFilter;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ManifestMergerAction}. */
@RunWith(JUnit4.class)
public class ManifestMergerActionTest {

  private Path working;

  /**
   * Returns a runfile's path.
   *
   * <p>The `path` must specify a valid runfile, meaning on Windows (where $RUNFILES_MANIFEST_ONLY
   * is 1) the `path` must exactly match a runfiles manifest entry, and on Linux/MacOS `path` must
   * point to a valid file in the runfiles directory.
   */
  @Nullable
  private static Path rlocation(String path) throws IOException {
    FileSystem fs = FileSystems.getDefault();
    if (fs.getPath(path).isAbsolute()) {
      return fs.getPath(path);
    }
    if ("1".equals(System.getenv("RUNFILES_MANIFEST_ONLY"))) {
      String manifest = System.getenv("RUNFILES_MANIFEST_FILE");
      assertThat(manifest).isNotNull();
      try (BufferedReader r =
          Files.newBufferedReader(Paths.get(manifest), Charset.defaultCharset())) {
        Splitter splitter = Splitter.on(' ').limit(2);
        String line = null;
        while ((line = r.readLine()) != null) {
          List<String> tokens = splitter.splitToList(line);
          if (tokens.size() == 2) {
            if (tokens.get(0).equals(path)) {
              return fs.getPath(tokens.get(1));
            }
          }
        }
      }
      return null;
    } else {
      String runfiles = System.getenv("RUNFILES_DIR");
      if (runfiles == null) {
        runfiles = System.getenv("JAVA_RUNFILES");
        assertThat(runfiles).isNotNull();
      }
      Path result = fs.getPath(runfiles).resolve(path);
      assertThat(result.toFile().exists()).isTrue(); // comply with function's contract
      return result;
    }
  }

  @Before public void setup() throws Exception {
    working = Files.createTempDirectory(toString());
    working.toFile().deleteOnExit();
  }

  @Test public void testMerge() throws Exception {
    final Path workingDir = Paths.get(System.getProperty("user.dir"));
    assertThat(workingDir.toFile().exists()).isTrue();
    assertThat(workingDir.toFile().isDirectory()).isTrue();

    String dataDir = System.getProperty("testdatadir");
    if (dataDir.charAt(dataDir.length() - 1) != '/') {
      dataDir = dataDir + '/';
    }

    final Path mergerManifest = rlocation(dataDir + "merger/AndroidManifest.xml");
    final Path mergeeManifestOne = rlocation(dataDir + "mergeeOne/AndroidManifest.xml");
    final Path mergeeManifestTwo = rlocation(dataDir + "mergeeTwo/AndroidManifest.xml");
    assertThat(mergerManifest.toFile().exists()).isTrue();
    assertThat(mergeeManifestOne.toFile().exists()).isTrue();
    assertThat(mergeeManifestTwo.toFile().exists()).isTrue();

    // The following code retrieves the path of the only AndroidManifest.xml in the expected/
    // manifests directory. Unfortunately, this test runs internally and externally and the files
    // have different names.
    final File expectedManifestDirectory =
        mergerManifest.getParent().resolveSibling("expected").toFile();
    final String[] debug =
        expectedManifestDirectory.list(new PatternFilenameFilter(".*AndroidManifest\\.xml$"));
    assertThat(debug).isNotNull();
    final File[] expectedManifestDirectoryManifests =
        expectedManifestDirectory.listFiles((File dir, String name) -> true);
    assertThat(expectedManifestDirectoryManifests).isNotNull();
    assertThat(expectedManifestDirectoryManifests).hasLength(1);
    final Path expectedManifest = expectedManifestDirectoryManifests[0].toPath();

    Files.createDirectories(working.resolve("output"));
    final Path mergedManifest = working.resolve("output/mergedManifest.xml");

    List<String> args =
        generateArgs(
            mergerManifest,
            ImmutableMap.of(mergeeManifestOne, "mergeeOne", mergeeManifestTwo, "mergeeTwo"),
            false, /* isLibrary */
            ImmutableMap.of("applicationId", "com.google.android.apps.testapp"),
            "", /* custom_package */
            mergedManifest);
    ManifestMergerAction.main(args.toArray(new String[0]));

    assertThat(
        Joiner.on(" ")
            .join(Files.readAllLines(mergedManifest, UTF_8))
            .replaceAll("\\s+", " ")
            .trim())
        .isEqualTo(
            Joiner.on(" ")
                .join(Files.readAllLines(expectedManifest, UTF_8))
                .replaceAll("\\s+", " ")
                .trim());
  }

  @Test public void fullIntegration() throws Exception {
    Files.createDirectories(working.resolve("output"));
    final Path binaryOutput = working.resolve("output/binaryManifest.xml");
    final Path libFooOutput = working.resolve("output/libFooManifest.xml");
    final Path libBarOutput = working.resolve("output/libBarManifest.xml");

    final Path binaryManifest = AndroidDataBuilder.of(working.resolve("binary"))
        .createManifest("AndroidManifest.xml", "com.google.app", "")
        .buildUnvalidated()
        .getManifest();
    final Path libFooManifest = AndroidDataBuilder.of(working.resolve("libFoo"))
        .createManifest("AndroidManifest.xml", "com.google.foo",
            " <application android:name=\"${applicationId}\" />")
        .buildUnvalidated()
        .getManifest();
    final Path libBarManifest = AndroidDataBuilder.of(working.resolve("libBar"))
        .createManifest("AndroidManifest.xml", "com.google.bar",
            "<application android:name=\"${applicationId}\">",
            "<activity android:name=\".activityFoo\" />",
            "</application>")
        .buildUnvalidated()
        .getManifest();

    // libFoo manifest merging
    List<String> args = generateArgs(libFooManifest, ImmutableMap.<Path, String>of(), true,
        ImmutableMap.<String, String>of(), "", libFooOutput);
    ManifestMergerAction.main(args.toArray(new String[0]));
    assertThat(Joiner.on(" ")
        .join(Files.readAllLines(libFooOutput, UTF_8))
        .replaceAll("\\s+", " ").trim()).contains(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            + "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\""
            + " package=\"com.google.foo\">"
            + " <application android:name=\"${applicationId}\" />"
            + "</manifest>");

    // libBar manifest merging
    args = generateArgs(libBarManifest, ImmutableMap.<Path, String>of(), true,
        ImmutableMap.<String, String>of(), "com.google.libbar", libBarOutput);
    ManifestMergerAction.main(args.toArray(new String[0]));
    assertThat(Joiner.on(" ")
        .join(Files.readAllLines(libBarOutput, UTF_8))
        .replaceAll("\\s+", " ").trim()).contains(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            + " <manifest xmlns:android=\"http://schemas.android.com/apk/res/android\""
            + " package=\"com.google.libbar\" >"
            + " <application android:name=\"${applicationId}\" >"
            + " <activity android:name=\"com.google.bar.activityFoo\" />"
            + " </application>"
            + " </manifest>");

    // binary manifest merging
    args = generateArgs(
        binaryManifest,
        ImmutableMap.of(libFooOutput, "libFoo", libBarOutput, "libBar"),
        false, /* library */
        ImmutableMap.of(
            "applicationId", "com.google.android.app",
            "foo", "this \\\\: is \"a, \"bad string"),
        "", /* customPackage */
        binaryOutput);
    ManifestMergerAction.main(args.toArray(new String[0]));
    assertThat(Joiner.on(" ")
        .join(Files.readAllLines(binaryOutput, UTF_8))
        .replaceAll("\\s+", " ").trim()).contains(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
            + " <manifest xmlns:android=\"http://schemas.android.com/apk/res/android\""
            + " package=\"com.google.android.app\" >"
            + " <application android:name=\"com.google.android.app\" >"
            + " <activity android:name=\"com.google.bar.activityFoo\" />"
            + " </application>"
            + " </manifest>");
  }

  private List<String> generateArgs(
      Path manifest,
      Map<Path, String> mergeeManifests,
      boolean library,
      Map<String, String> manifestValues,
      String customPackage,
      Path manifestOutput) {
    return ImmutableList.of(
        "--manifest", manifest.toString(),
        "--mergeeManifests", mapToDictionaryString(mergeeManifests),
        "--mergeType", library ? "LIBRARY" : "APPLICATION",
        "--manifestValues", mapToDictionaryString(manifestValues),
        "--customPackage", customPackage,
        "--manifestOutput", manifestOutput.toString());
  }

  private <K, V> String mapToDictionaryString(Map<K, V> map) {
    StringBuilder sb = new StringBuilder();
    Iterator<Entry<K, V>> iter = map.entrySet().iterator();
    while (iter.hasNext()) {
      Entry<K, V> entry = iter.next();
      sb.append(entry.getKey().toString().replace(":", "\\:").replace(",", "\\,"));
      sb.append(':');
      sb.append(entry.getValue().toString().replace(":", "\\:").replace(",", "\\,"));
      if (iter.hasNext()) {
        sb.append(',');
      }
    }
    return sb.toString();
  }
}
