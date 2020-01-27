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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DensitySpecificResourceFilter}. */
@RunWith(JUnit4.class)
public class DensitySpecificResourceFilterTest {

  private Path tmp;

  @Test public void testNoDensityResources() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/layout/derp.xml",
            "test/res/layout-ldrtl/derp.png",
            "test/res/layout-ldrtl-xhdpi/derp.png",
            "test/res/drawable-en/test.png"),
        ImmutableList.of(
            "test/res/layout/derp.xml",
            "test/res/layout-ldrtl/derp.png",
            "test/res/layout-ldrtl-xhdpi/derp.png",
            "test/res/drawable-en/test.png"),
        ImmutableList.of("xxhdpi"));
  }

  @Test public void testDontFilterXml() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-xxhdpi/derp.xml",
            "test/res/drawable-ldpi/derp.xml",
            "test/res/drawable-en/test.png"),
        ImmutableList.of(
            "test/res/drawable-xxhdpi/derp.xml",
            "test/res/drawable-ldpi/derp.xml",
            "test/res/drawable-en/test.png"),
        ImmutableList.of("xxhdpi"));
  }

  @Test public void testNoDpiNoFilter() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-nodpi/test.png",
            "test/res/drawable-xxhdpi/test.png",
            "test/res/drawable-ldpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-nodpi/test.png",
            "test/res/drawable-xxhdpi/test.png"),
        ImmutableList.of("xxhdpi"));
  }

  @Test public void testChoosesExactDensity() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-nodpi/test.png",
            "test/res/drawable/test.png",
            "test/res/drawable-mdpi/derp.png",
            "test/res/drawable-xxhdpi/test.png",
            "test/res/drawable-mdpi/test.png",
            "test/res/drawable-xxxhdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-nodpi/test.png",
            "test/res/drawable/test.png",
            "test/res/drawable-mdpi/derp.png",
            "test/res/drawable-xxhdpi/test.png"),
        ImmutableList.of("xxhdpi"));
  }

  @Test public void testMultipleDensities() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-mdpi/test.png",
            "test/res/drawable-xxhdpi/test.png",
            "test/res/drawable-xxxhdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-xxhdpi/test.png",
            "test/res/drawable-xxxhdpi/test.png"),
        ImmutableList.of("xxhdpi", "xxxhdpi"));
  }

  @Test public void testNonStandardDensity() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-mdpi/test.png",
            "test/res/drawable-hdpi/test.png",
            "test/res/drawable-280dpi/test.png",
            "test/res/drawable-xhdpi/test.png",
            "test/res/drawable-340dpi/test.png",
            "test/res/drawable-xxhdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-280dpi/test.png",
            "test/res/drawable-340dpi/test.png"),
        ImmutableList.of("280dpi", "340dpi"));
  }

  @Test public void testUnknownDensityFails() {
    assertThrows(
        MergingException.class,
        () ->
            checkTransformedResources(
                ImmutableList.<String>of(),
                ImmutableList.<String>of(),
                ImmutableList.of("xxhdpi", "322dpi")));
  }

  @Test public void testPrefersHigherQualityWhenAffinityExact() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-xxxhdpi/test.png",
            "test/res/drawable-mdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-xxxhdpi/test.png"),
        ImmutableList.of("xhdpi"));
  }

  @Test public void testPrefersExactlyDoubleDensity() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-hdpi/test.png",
            "test/res/drawable-mdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-hdpi/test.png"),
        ImmutableList.of("ldpi"));
  }


  @Test public void testGroupsByQualifiers() throws Exception {
    checkTransformedResources(
        ImmutableList.of(
            "test/res/drawable-en-ldrtl-hdpi/test.png",
            "test/res/drawable-mdpi/test.png"),
        ImmutableList.of(
            "test/res/drawable-en-ldrtl-hdpi/test.png",
            "test/res/drawable-mdpi/test.png"),
        ImmutableList.of("ldpi"));
  }

  @Test public void testQualifiers() throws Exception {
    // This list checks that we correctly grab a -xhdpi-v19 version of the drawable when it is not
    // present at our requested density.
    checkTransformedResources(
        ImmutableList.of(
            "res/drawable-mdpi-v11/btn_edit_pressed.png",
            "res/drawable-mdpi/btn_edit_pressed.png",
            "res/drawable-xhdpi-v19/btn_edit_pressed.png",
            "res/drawable-tvdpi-v11/btn_edit_pressed.png",
            "res/drawable-hdpi-v11/btn_edit_pressed.png",
            "res/drawable-xhdpi-v11/btn_edit_pressed.png",
            "res/drawable-xxhdpi-v19/btn_edit_pressed.png",
            "res/drawable-hdpi/btn_edit_pressed.png",
            "res/drawable-tvdpi/btn_edit_pressed.png",
            "res/drawable-xxhdpi/btn_edit_pressed.png",
            "res/drawable-xhdpi/btn_edit_pressed.png",
            "res/drawable-tvdpi-v19/btn_edit_pressed.png",
            "res/drawable-hdpi-v19/btn_edit_pressed.png",
            "res/drawable-xxhdpi-v11/btn_edit_pressed.png"),
        ImmutableList.of(
            "res/drawable-mdpi-v11/btn_edit_pressed.png",
            "res/drawable-mdpi/btn_edit_pressed.png",
            "res/drawable-xhdpi-v19/btn_edit_pressed.png"),
        ImmutableList.of("mdpi"));
  }

  @Test public void fullIntegration() throws Exception {
    createFiles(
        "res/drawable-mdpi-v11/btn_edit_pressed.png",
        "res/drawable-mdpi/btn_edit_pressed.png",
        "res/drawable-xhdpi-v19/btn_edit_pressed.png",
        "res/drawable-tvdpi-v11/btn_edit_pressed.png",
        "res/drawable-hdpi-v11/btn_edit_pressed.png",
        "res/drawable-xhdpi-v11/btn_edit_pressed.png",
        "res/drawable-xxhdpi-v19/btn_edit_pressed.png",
        "res/drawable-hdpi/btn_edit_pressed.png",
        "res/drawable-tvdpi/btn_edit_pressed.png",
        "res/drawable-xxhdpi/btn_edit_pressed.png",
        "res/drawable-xhdpi/btn_edit_pressed.png",
        "res/drawable-tvdpi-v19/btn_edit_pressed.png",
        "res/drawable-hdpi-v19/btn_edit_pressed.png",
        "res/drawable-xxhdpi-v11/btn_edit_pressed.png");

    Path out = Files.createTempDirectory(this.toString());
    Path working = Files.createTempDirectory(this.toString());
    ImmutableList<String> densities = ImmutableList.of("xxhdpi");

    DensitySpecificResourceFilter filter =
        new DensitySpecificResourceFilter(densities, out, working);

    final Path filteredResourceDir = filter.filter(tmp);
    final List<Path> filteredResources = new ArrayList<>();
    try {
      Files.walkFileTree(
          filteredResourceDir,
          EnumSet.of(FileVisitOption.FOLLOW_LINKS),
          Integer.MAX_VALUE,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                throws IOException {
              filteredResources.add(filteredResourceDir.relativize(path));
              return FileVisitResult.CONTINUE;
            }
          });
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    FileSystem fs = FileSystems.getDefault();
    assertThat(filteredResources)
        .containsExactly(
            fs.getPath("res/drawable-xxhdpi/btn_edit_pressed.png"),
            fs.getPath("res/drawable-xxhdpi-v11/btn_edit_pressed.png"),
            fs.getPath("res/drawable-xxhdpi-v19/btn_edit_pressed.png"));
  }

  @Before
  public void setUpEnvironment() throws Exception {
    tmp = Files.createTempDirectory(this.toString());
  }

  private ImmutableList <Path> createFiles(String... pathStrings) throws IOException {
    ImmutableList.Builder<Path> paths = ImmutableList.builder();
    for (String pathString : pathStrings) {
      final Path path = tmp.resolve(pathString);
      Files.createDirectories(path.getParent());
      Files.createFile(path);
      paths.add(path);
    }
    return paths.build();
  }

  private void checkTransformedResources(List<String> resourcePaths,
      List<String> expectedResourcePaths, List<String> densities) throws MergingException {
    List<Path> resources = new ArrayList<>();
    FileSystem fs = FileSystems.getDefault();
    for (String resourcePath : resourcePaths) {
      resources.add(fs.getPath(resourcePath));
    }

    DensitySpecificResourceFilter transformer =
        new DensitySpecificResourceFilter(densities, null, null);
    Set<Path> resourcesToRemove = new HashSet<>(transformer.getResourceToRemove(resources));
    List<String> actualResourcePaths = new ArrayList<>();

    for (Path p : resources) {
      if (!resourcesToRemove.contains(p)) {
        actualResourcePaths.add(p.toString());
      }
    }

    List<String> expected = Lists.transform(expectedResourcePaths, n -> fs.getPath(n).toString());
    assertThat(actualResourcePaths).containsExactlyElementsIn(expected);
  }
}
