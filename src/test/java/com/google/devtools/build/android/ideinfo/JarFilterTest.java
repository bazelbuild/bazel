// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ideinfo;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import com.google.devtools.build.android.ideinfo.JarFilter.JarFilterOptions;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.JavaSourcePackage;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.PackageManifest;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link JarFilter} */
@RunWith(JUnit4.class)
public class JarFilterTest {

  @Rule public TemporaryFolder folder = new TemporaryFolder();

  @Test
  public void testFilterMethod() throws Exception {
    List<String> prefixes =
        ImmutableList.of("com/google/foo/Foo", "com/google/bar/Bar", "com/google/baz/Baz");
    assertThat(JarFilter.shouldKeepClass(prefixes, "com/google/foo/Foo.class")).isTrue();
    assertThat(JarFilter.shouldKeepClass(prefixes, "com/google/foo/Foo$Inner.class")).isTrue();
    assertThat(JarFilter.shouldKeepClass(prefixes, "com/google/bar/Bar.class")).isTrue();
    assertThat(JarFilter.shouldKeepClass(prefixes, "com/google/foo/Foo/NotFoo.class")).isFalse();
    assertThat(JarFilter.shouldKeepClass(prefixes, "wrong/com/google/foo/Foo.class")).isFalse();
  }

  @Test
  public void legacyIntegrationTest() throws Exception {
    PackageManifest packageManifest =
        PackageManifest.newBuilder()
            .addSources(
                JavaSourcePackage.newBuilder()
                    .setArtifactLocation(
                        ArtifactLocation.newBuilder()
                            .setIsSource(true)
                            .setRelativePath("com/google/foo/Foo.java"))
                    .setPackageString("com.google.foo"))
            .addSources(
                JavaSourcePackage.newBuilder()
                    .setArtifactLocation(
                        ArtifactLocation.newBuilder()
                            .setIsSource(true)
                            .setRelativePath("com/google/bar/Bar.java"))
                    .setPackageString("com.google.bar"))
            .addSources(
                JavaSourcePackage.newBuilder()
                    .setArtifactLocation(
                        ArtifactLocation.newBuilder()
                            .setIsSource(true)
                            .setRelativePath("some/path/Test.java"))
                    .setPackageString("com.google.test"))
            .build();
    assertThat(JarFilter.parsePackageManifest(packageManifest))
        .containsExactly("com/google/foo/Foo", "com/google/bar/Bar", "com/google/test/Test");
    File manifest = folder.newFile("foo.manifest");
    try (FileOutputStream outputStream = new FileOutputStream(manifest)) {
      packageManifest.writeTo(outputStream);
    }

    File filterJar = folder.newFile("foo.jar");
    try (ZipOutputStream zo = new ZipOutputStream(new FileOutputStream(filterJar))) {
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo$Inner.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/bar/Bar.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/test/Test.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo2.class"));
      zo.closeEntry();
    }

    File outputJar = folder.newFile("foo-filtered-gen.jar");

    String[] args =
        new String[] {
          "--jars",
          filterJar.getPath(),
          "--output",
          outputJar.getPath(),
          "--manifest",
          manifest.getPath()
        };
    JarFilter.JarFilterOptions options = JarFilter.parseArgs(args);
    JarFilter.main(options);

    List<String> filteredJarNames = Lists.newArrayList();
    try (ZipFile zipFile = new ZipFile(outputJar)) {
      Enumeration<? extends ZipEntry> entries = zipFile.entries();
      while (entries.hasMoreElements()) {
        ZipEntry zipEntry = entries.nextElement();
        filteredJarNames.add(zipEntry.getName());
      }
    }

    assertThat(filteredJarNames)
        .containsExactly(
            "com/google/foo/Foo.class",
            "com/google/foo/Foo$Inner.class",
            "com/google/bar/Bar.class",
            "com/google/test/Test.class");
  }

  @Test
  public void fullIntegrationTest() throws Exception {
    File fooJava = folder.newFile("Foo.java");
    Files.write("package com.google.foo; class Foo { class Inner {} }".getBytes(UTF_8), fooJava);

    File barJava = folder.newFile("Bar.java");
    Files.write("package com.google.foo.bar; class Bar {}".getBytes(UTF_8), barJava);

    File srcJar = folder.newFile("gen.srcjar");
    try (ZipOutputStream zo = new ZipOutputStream(new FileOutputStream(srcJar))) {
      zo.putNextEntry(new ZipEntry("com/google/foo/gen/Gen.java"));
      zo.write("package gen; class Gen {}".getBytes(UTF_8));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/gen/Gen2.java"));
      zo.write("package gen; class Gen2 {}".getBytes(UTF_8));
      zo.closeEntry();
    }

    File src3Jar = folder.newFile("gen3.srcjar");
    try (ZipOutputStream zo = new ZipOutputStream(new FileOutputStream(src3Jar))) {
      zo.putNextEntry(new ZipEntry("com/google/foo/gen/Gen3.java"));
      zo.write("package gen; class Gen3 {}".getBytes(UTF_8));
      zo.closeEntry();
    }

    File filterJar = folder.newFile("foo.jar");
    try (ZipOutputStream zo = new ZipOutputStream(new FileOutputStream(filterJar))) {
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo$Inner.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/bar/Bar.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen2.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen3.class"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo2.class"));
      zo.closeEntry();
    }
    File filterSrcJar = folder.newFile("foo-src.jar");
    try (ZipOutputStream zo = new ZipOutputStream(new FileOutputStream(filterSrcJar))) {
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/bar/Bar.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen2.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("gen/Gen3.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/Foo2.java"));
      zo.closeEntry();
      zo.putNextEntry(new ZipEntry("com/google/foo/bar/Bar2.java"));
      zo.closeEntry();
    }

    File filteredJar = folder.newFile("foo-filtered-gen.jar");
    File filteredSourceJar = folder.newFile("foo-filtered-gen-src.jar");

    String[] args =
        new String[] {
          "--keep_java_files",
          fooJava.getPath() + File.pathSeparator + barJava.getPath(),
          "--keep_source_jars",
          Joiner.on(File.pathSeparator).join(srcJar.getPath(), src3Jar.getPath()),
          "--filter_jars",
          filterJar.getPath(),
          "--filter_source_jars",
          filterSrcJar.getPath(),
          "--filtered_jar",
          filteredJar.getPath(),
          "--filtered_source_jar",
          filteredSourceJar.getPath()
        };
    JarFilterOptions options = JarFilter.parseArgs(args);
    JarFilter.main(options);

    List<String> filteredJarNames = Lists.newArrayList();
    try (ZipFile zipFile = new ZipFile(filteredJar)) {
      Enumeration<? extends ZipEntry> entries = zipFile.entries();
      while (entries.hasMoreElements()) {
        ZipEntry zipEntry = entries.nextElement();
        filteredJarNames.add(zipEntry.getName());
      }
    }

    List<String> filteredSourceJarNames = Lists.newArrayList();
    try (ZipFile zipFile = new ZipFile(filteredSourceJar)) {
      Enumeration<? extends ZipEntry> entries = zipFile.entries();
      while (entries.hasMoreElements()) {
        ZipEntry zipEntry = entries.nextElement();
        filteredSourceJarNames.add(zipEntry.getName());
      }
    }

    assertThat(filteredJarNames)
        .containsExactly(
            "com/google/foo/Foo.class",
            "com/google/foo/Foo$Inner.class",
            "com/google/foo/bar/Bar.class",
            "gen/Gen.class",
            "gen/Gen2.class",
            "gen/Gen3.class");

    assertThat(filteredSourceJarNames)
        .containsExactly(
            "com/google/foo/Foo.java",
            "com/google/foo/bar/Bar.java",
            "gen/Gen.java",
            "gen/Gen2.java",
            "gen/Gen3.java");
  }
}
