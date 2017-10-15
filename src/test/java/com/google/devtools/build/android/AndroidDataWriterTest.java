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

import static com.google.common.truth.Truth.assertAbout;
import static com.google.common.truth.Truth.assertThat;

import com.android.ide.common.internal.PngCruncher;
import com.android.ide.common.internal.PngException;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Jimfs;
import com.google.common.truth.FailureStrategy;
import com.google.common.truth.SubjectFactory;
import com.google.common.util.concurrent.MoreExecutors;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the AndroidDataWriter. */
@RunWith(JUnit4.class)
public class AndroidDataWriterTest {

  private static final String START_RESOURCES =
      new String(AndroidDataWriter.PRELUDE) + "<resources";
  private static final String END_RESOURCES = new String(AndroidDataWriter.END_RESOURCES);

  private FileSystem fs;

  @Before
  public void createCleanEnvironment() {
    fs = Jimfs.newFileSystem();
  }

  @Test
  public void writeResourceFile() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    String drawable = "drawable/menu.gif";
    String layout = "layout/foo.xml";
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(layout, AndroidDataBuilder.ResourceType.LAYOUT, "")
            .addResourceBinary(drawable, Files.createFile(fs.getPath("menu.gif")))
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(drawable)).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(layout)).exists();
  }

  @Test
  public void writePngInRawAndNotInRaw() throws Exception {
    Path tmpDir = Files.createTempDirectory(this.toString());
    final Path target = tmpDir.resolve("target");
    final Path source = tmpDir.resolve("source");
    final String drawable = "drawable/crunch.png";
    final String raw = "raw/nocrunch.png";
    final String layout = "layout/foo.xml";
    AndroidDataWriter mergedDataWriter =
        AndroidDataWriter.createWith(
            target,
            target.resolve("res"),
            target.resolve("assets"),
            new PngCruncher() {
              @Override
              public int start() {
                return 0;
              }

              @Override
              public void end(int key) throws InterruptedException {
              }

              @Override
              public void crunchPng(int key, File from, File to)
                  throws PngException {
                assertThat(from.toString()).doesNotContain(raw);
                try {
                  Files.copy(from.toPath(), to.toPath());
                } catch (IOException e) {
                  throw new PngException(e);
                }
              }
            },
            MoreExecutors.newDirectExecutorService());
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(layout, AndroidDataBuilder.ResourceType.LAYOUT, "")
            .addResourceBinary(drawable, Files.createFile(tmpDir.resolve("crunch.png")))
            .addResourceBinary(raw, Files.createFile(tmpDir.resolve("nocrunch.png")))
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(drawable)).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(raw)).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(layout)).exists();
  }

  @Test
  public void writeNinepatchResourceFile() throws Exception {
    Path tmpDir = Files.createTempDirectory(this.toString());
    Path target = tmpDir.resolve("target");
    Path source = tmpDir.resolve("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    String drawable = "drawable-hdpi-v4/seven_eight.9.png";
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResourceBinary(drawable, Files.createFile(fs.getPath("seven_eight.9.png")))
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve(drawable)).exists();
  }

  @Test
  public void writeResourceXml() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/ids.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<item name=\"id1\" type=\"id\"/>",
                "<item name=\"id\" type=\"id\"/>")
            .addResource(
                "values/stubs.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<item name=\"walrus\" type=\"drawable\"/>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("values/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + ">",
            "<!-- " + fs.getPath("source/res/values/stubs.xml") + " --><eat-comment/>",
            "<item name='walrus' type='drawable'/>",
            "<!-- " + fs.getPath("source/res/values/ids.xml") + " --><eat-comment/>",
            "<item name='id' type='id'/>",
            "<item name='id1' type='id'/>",
            END_RESOURCES);
  }

  @Test
  public void writeResourceXmlWithQualfiers() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/ids.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<item name=\"id1\" type=\"id\"/>")
            .addResource(
                "values-en/ids.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<item name=\"id1\" type=\"id\"/>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("values/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + ">",
            "<!-- "
                + fs.getPath("source/res/values/ids.xml")
                + " --><eat-comment/><item name='id1' type='id'/>",
            END_RESOURCES);
    assertAbout(paths).that(actual.getResourceDir().resolve("values-en/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values-en/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + ">",
            "<!-- "
                + fs.getPath("source/res/values-en/ids.xml")
                + " --><eat-comment/><item name='id1' type='id'/>",
            END_RESOURCES);
  }

  @Test
  public void writePublicResourceSameNameDifferentType() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "values/integers.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<integer name=\"foo\">12345</integer>",
                "<public name=\"foo\" type=\"integer\" id=\"0x7f040000\"/>",
                "<integer name=\"zoo\">54321</integer>",
                "<public name=\"zoo\" type=\"integer\" />")
            .addResource(
                "values/strings.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<string name=\"foo\">meow</string>",
                "<public name=\"foo\" type=\"string\" id=\"0x7f050000\"/>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("values/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + ">",
            "<!-- " + fs.getPath("source/res/values/integers.xml") + " --><eat-comment/>",
            "<integer name='foo'>12345</integer>",
            "<integer name='zoo'>54321</integer>",
            "<!-- " + fs.getPath("source/res/values/strings.xml") + " --><eat-comment/>",
            "<string name='foo'>meow</string>",
            "<!-- " + fs.getPath("source/res/values/integers.xml") + " --><eat-comment/>",
            "<public name='foo' type='integer' id='0x7f040000'/>",
            "<public name='foo' type='string' id='0x7f050000'/>",
            "<public name='zoo' type='integer' />",
            END_RESOURCES);
  }

  @Test
  public void writeWithIDDuplicates() throws Exception {
    // We parse IDs from layout, etc. XML. We can include it in the merged values.xml redundantly
    // (redundant because we also give aapt the original layout xml, which it can parse for IDs
    // too), but make sure we don't accidentally put multiple copies in the merged values.xml file.
    // Otherwise, aapt will throw an error if there are duplicates in the same values.xml file.
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addResource(
                "layout/some_layout.xml",
                AndroidDataBuilder.ResourceType.LAYOUT,
                "<TextView android:id=\"@+id/MyTextView\"",
                "          android:text=\"@string/walrus\"",
                "          android:layout_above=\"@+id/AnotherTextView\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />",
                // Test redundantly having a "+id/MyTextView" in a different attribute.
                "<TextView android:id=\"@id/AnotherTextView\"",
                "          android:text=\"@string/walrus\"",
                "          android:layout_below=\"@+id/MyTextView\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />")
            // Test what happens if a user accidentally uses the same ID in multiple layouts too.
            .addResource(
                "layout/another_layout.xml",
                AndroidDataBuilder.ResourceType.LAYOUT,
                "<TextView android:id=\"@+id/MyTextView\"",
                "          android:text=\"@string/walrus\"",
                "          android:layout_width=\"wrap_content\"",
                "          android:layout_height=\"wrap_content\" />")
            // Also check what happens if a value XML file also contains the same ID.
            .addResource(
                "values/ids.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<item name=\"MyTextView\" type=\"id\"/>",
                "<item name=\"OtherId\" type=\"id\"/>")
            .addResource(
                "values/strings.xml",
                AndroidDataBuilder.ResourceType.VALUE,
                "<string name=\"walrus\">I has a bucket</string>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("layout/some_layout.xml")).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("values/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + ">",
            "<!-- " + fs.getPath("source/res/values/ids.xml") + " --><eat-comment/>",
            "<item name='MyTextView' type='id'/>",
            "<item name='OtherId' type='id'/>",
            "<!-- " + fs.getPath("source/res/values/strings.xml") + " --><eat-comment/>",
            "<string name='walrus'>I has a bucket</string>",
            END_RESOURCES);
  }

  @Test
  public void writeResourceXmlWithAttributes() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addValuesWithAttributes(
                "values/ids.xml",
                ImmutableMap.of("foo", "fooVal", "bar", "barVal"),
                "<item name=\"id1\" type=\"id\"/>",
                "<item name=\"id\" type=\"id\"/>")
            .addValuesWithAttributes(
                "values/stubs.xml",
                ImmutableMap.of("baz", "bazVal"),
                "<item name=\"walrus\" type=\"drawable\"/>")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getResourceDir().resolve("values/values.xml")).exists();
    assertAbout(paths)
        .that(actual.getResourceDir().resolve("values/values.xml"))
        .xmlContentsIsEqualTo(
            START_RESOURCES + " foo=\"fooVal\" bar=\"barVal\" baz=\"bazVal\">",
            "<!-- " + fs.getPath("source/res/values/stubs.xml") + " --><eat-comment/>",
            "<item name='walrus' type='drawable'/>",
            "<!-- " + fs.getPath("source/res/values/ids.xml") + " --><eat-comment/>",
            "<item name='id' type='id'/>",
            "<item name='id1' type='id'/>",
            END_RESOURCES);
  }

  @Test
  public void writeAssetFile() throws Exception {
    Path target = fs.getPath("target");
    Path source = fs.getPath("source");
    AndroidDataWriter mergedDataWriter = AndroidDataWriter.createWithDefaults(target);
    String asset = "hunting/of/the/boojum";
    ParsedAndroidData direct =
        AndroidDataBuilder.of(source)
            .addAsset(asset, "not a snark!")
            .createManifest("AndroidManifest.xml", "com.carroll.lewis", "")
            .buildParsed();
    MergedAndroidData actual =
        UnwrittenMergedAndroidData.of(
                source.resolve("AndroidManifest.xml"), direct, ParsedAndroidDataBuilder.empty())
            .write(mergedDataWriter);

    assertAbout(paths).that(actual.getManifest()).exists();
    assertAbout(paths).that(actual.getAssetDir().resolve(asset)).exists();
  }

  private static final SubjectFactory<PathsSubject, Path> paths =
      new SubjectFactory<PathsSubject, Path>() {
        @Override
        public PathsSubject getSubject(FailureStrategy failureStrategy, Path path) {
          return new PathsSubject(failureStrategy, path);
        }
      };
}
