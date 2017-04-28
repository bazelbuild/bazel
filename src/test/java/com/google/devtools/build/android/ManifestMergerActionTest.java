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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.PatternFilenameFilter;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ManifestMergerAction}. */
@RunWith(JUnit4.class)
public class ManifestMergerActionTest {

  private Path working;

  @Before public void setup() throws Exception {
    working = Files.createTempDirectory(toString());
    working.toFile().deleteOnExit();
  }

  @Test public void testMerge() throws Exception {
    final Path workingDir = Paths.get(System.getProperty("user.dir"));
    final Path manifestData = workingDir.resolve(System.getProperty("testdatadir"));

    final Path mergerMannifest = manifestData.resolve("merger/AndroidManifest.xml");
    final Path mergeeManifestOne = manifestData.resolve("mergeeOne/AndroidManifest.xml");
    final Path mergeeManifestTwo = manifestData.resolve("mergeeTwo/AndroidManifest.xml");

    // The following code retrieves the path of the only AndroidManifest.xml in the expected/
    // manifests directory. Unfortunately, this test runs internally and externally and the files
    // have different names.
    final File expectedManifestDirectory = manifestData.resolve("expected").toFile();
    final File[] expectedManifestDirectoryManifests =
        expectedManifestDirectory.listFiles(new PatternFilenameFilter(".*AndroidManifest\\.xml$"));
    assertThat(expectedManifestDirectoryManifests.length).isEqualTo(1);
    final Path expectedManifest = expectedManifestDirectoryManifests[0].toPath();

    Files.createDirectories(working.resolve("output"));
    final Path mergedManifest = working.resolve("output/mergedManifest.xml");

    List<String> args = generateArgs(
        mergerMannifest,
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
            "<activity android:name=\"com.google.bar.activityFoo\" />",
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
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\""
            + " package=\"com.google.libbar\">"
            + "<application android:name=\"${applicationId}\">"
            + " <activity android:name=\"com.google.bar.activityFoo\">"
            + "</activity>"
            + " </application>"
            + "</manifest>");

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
