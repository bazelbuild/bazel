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
import static com.google.devtools.build.android.DensitySpecificManifestProcessor.PLAY_STORE_SUPPORTED_DENSITIES;
import static com.google.devtools.build.android.DensitySpecificManifestProcessor.SCREEN_SIZES;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.jimfs.Jimfs;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/** Tests for {@link DensitySpecificManifestProcessor}. */
@RunWith(JUnit4.class)
public class DensitySpecificManifestProcessorTest {

  private FileSystem fs;
  private Path tmp;

  @Test public void testNoDensities() throws Exception {
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(ImmutableList.<String>of(),
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isEqualTo(manifest);
  }

  @Test public void testSingleDensity() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("xhdpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkModification(modified, densities);
  }
  
  @Test public void test280Density() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("280dpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkModification(modified, densities);
  }

  @Test public void testMultipleDensities() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("xhdpi", "xxhdpi", "560dpi", "xxxhdpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkModification(modified, densities);
  }

  @Test public void omitCompatibleScreensIfDensityUnsupported() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("xhdpi", "340dpi", "xxhdpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkCompatibleScreensOmitted(modified);
  }

  @Test public void testExistingCompatibleScreens() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("xhdpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "<compatible-screens>",
        "</compatible-screens>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkModification(modified, densities);
  }

  @Test public void testExistingSupersetCompatibleScreens() throws Exception {
    ImmutableList<String> densities = ImmutableList.of("ldpi");
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "<compatible-screens>",
        "  <screen android:screenSize='small' android:screenDensity='ldpi' />",
        "  <screen android:screenSize='normal' android:screenDensity='ldpi' />",
        "  <screen android:screenSize='large' android:screenDensity='ldpi' />",
        "  <screen android:screenSize='xlarge' android:screenDensity='ldpi' />",
        "  <screen android:screenSize='small' android:screenDensity='480' />",
        "  <screen android:screenSize='normal' android:screenDensity='480' />",
        "  <screen android:screenSize='large' android:screenDensity='480' />",
        "  <screen android:screenSize='xlarge' android:screenDensity='480' />",
        "</compatible-screens>",
        "</manifest>");
    Path modified = new DensitySpecificManifestProcessor(densities,
        tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
    assertThat((Object) modified).isNotNull();
    checkModification(modified, ImmutableList.<String>of("ldpi", "xxhdpi"));
  }

  @Test public void testMalformedManifest() throws Exception {
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "</manifest>");
    try {
      new DensitySpecificManifestProcessor(ImmutableList.of("xhdpi"),
          tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
      fail();
    } catch (ManifestProcessingException e) {
      assertThat(e).hasMessageThat().contains("must be well-formed");
    }
  }

  @Test public void testNoManifest() throws Exception {
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
    try {
      new DensitySpecificManifestProcessor(ImmutableList.of("xhdpi"),
          tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
      fail();
    } catch (ManifestProcessingException e) {
      assertThat(e).hasMessageThat().contains("Premature end of file.");
    }
  }

  @Test public void testNestedManifest() throws Exception {
    Path manifest = createManifest("<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "          package='com.google.test'>",
        "  <manifest xmlns:android='http://schemas.android.com/apk/res/android'",
        "            package='com.google.test'>",
        "  </manifest>",
        "</manifest>");
    try {
      new DensitySpecificManifestProcessor(ImmutableList.of("xhdpi"),
          tmp.resolve("manifest-filtered/AndroidManifest.xml")).process(manifest);
      fail();
    } catch (ManifestProcessingException e) {
      assertThat(e).hasMessageThat().contains("does not contain exactly one <manifest>");
    }
  }

  @Before
  public void setUpEnvironment() throws Exception {
    fs = Jimfs.newFileSystem();
    tmp = fs.getPath("/tmp");
    Files.createDirectory(tmp);
  }

  @After
  public void cleanUpEnvironment() throws Exception {
    fs.close();
  }

  private Path createManifest(String... lines) throws IOException {
    final Path path = tmp.resolve("AndroidManifest.xml");
    Files.createDirectories(path.getParent());
    Files.deleteIfExists(path);
    BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8);
    writer.write(Joiner.on("\n").join(lines));
    writer.close();
    return path;
  }

  private void checkModification(Path manifest, List<String> densities) throws Exception {
    Set<String> sizeDensities = new HashSet<>();
    for (String density : densities) {
      for (String screenSize : SCREEN_SIZES) {
        sizeDensities.add(screenSize
            + PLAY_STORE_SUPPORTED_DENSITIES.get(density));
      }
    }

    DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
    Document doc = db.parse(Files.newInputStream(manifest));
    NodeList compatibleScreensNodes = doc.getElementsByTagName("compatible-screens");
    assertThat(compatibleScreensNodes.getLength()).isEqualTo(1);
    Node compatibleScreens = compatibleScreensNodes.item(0);
    NodeList screens = doc.getElementsByTagName("screen");
    assertThat(screens.getLength()).isEqualTo(densities.size() * SCREEN_SIZES.size());
    for (int i = 0; i < screens.getLength(); i++) {
      Node s = screens.item(i);
      assertThat(s.getParentNode().isSameNode(compatibleScreens)).isTrue();
      if (s.getNodeType() == Node.ELEMENT_NODE) {
        Element screen = (Element) s;
        assertThat(
                sizeDensities.remove(
                    screen.getAttribute("android:screenSize")
                        + screen.getAttribute("android:screenDensity")))
            .isTrue();
      }
    }
    assertThat(sizeDensities).isEmpty();
  }

  private void checkCompatibleScreensOmitted(Path manifest) throws Exception {
    DocumentBuilder db = DocumentBuilderFactory.newInstance().newDocumentBuilder();
    Document doc = db.parse(Files.newInputStream(manifest));
    NodeList compatibleScreensNodes = doc.getElementsByTagName("compatible-screens");
    assertThat(compatibleScreensNodes.getLength()).isEqualTo(0);
    NodeList screens = doc.getElementsByTagName("screen");
    assertThat(screens.getLength()).isEqualTo(0);
  }

}
