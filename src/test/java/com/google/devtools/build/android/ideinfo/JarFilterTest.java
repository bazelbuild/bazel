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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.JavaSourcePackage;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.PackageManifest;

import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link JarFilter}
 */
@RunWith(JUnit4.class)
public class JarFilterTest {

  @Test
  public void testParseCommandLineArguments() throws Exception {
    String[] args = new String[]{
        "--jars",
        "/tmp/1.jar" + File.pathSeparator + "/tmp/2.jar",
        "--output",
        "/tmp/out.jar",
        "--manifest",
        "/tmp/manifest.file",
    };
    JarFilter.JarFilterOptions options = JarFilter.parseArgs(args);
    assertThat(options.jars).containsExactly(
        Paths.get("/tmp/1.jar"),
        Paths.get("/tmp/2.jar")
    );
    assertThat(options.output.toString()).isEqualTo(Paths.get("/tmp/out.jar").toString());
    assertThat(options.manifest.toString()).isEqualTo(Paths.get("/tmp/manifest.file").toString());
  }

  @Test
  public void testFilterMethod() throws Exception {
    List<String> prefixes = ImmutableList.of(
        "com/google/foo/Foo",
        "com/google/bar/Bar",
        "com/google/baz/Baz"
    );
    assertThat(JarFilter.shouldKeep(prefixes, "com/google/foo/Foo.class")).isTrue();
    assertThat(JarFilter.shouldKeep(prefixes, "com/google/foo/Foo$Inner.class")).isTrue();
    assertThat(JarFilter.shouldKeep(prefixes, "com/google/bar/Bar.class")).isTrue();
    assertThat(JarFilter.shouldKeep(prefixes, "com/google/foo/Foo/NotFoo.class")).isFalse();
    assertThat(JarFilter.shouldKeep(prefixes, "wrong/com/google/foo/Foo.class")).isFalse();
  }

  @Test
  public void testManifestParser() throws Exception {
    PackageManifest packageManifest = PackageManifest.newBuilder()
        .addSources(JavaSourcePackage.newBuilder()
            .setArtifactLocation(ArtifactLocation.newBuilder()
                .setIsSource(true)
                .setRelativePath("com/google/foo/Foo.java"))
            .setPackageString("com.google.foo"))
        .addSources(JavaSourcePackage.newBuilder()
            .setArtifactLocation(ArtifactLocation.newBuilder()
                .setIsSource(true)
                .setRelativePath("com/google/bar/Bar.java"))
            .setPackageString("com.google.bar"))
        .addSources(JavaSourcePackage.newBuilder()
            .setArtifactLocation(ArtifactLocation.newBuilder()
                .setIsSource(true)
                .setRelativePath("some/path/Test.java"))
            .setPackageString("com.google.test"))
        .build();
    assertThat(JarFilter.parsePackageManifest(packageManifest)).containsExactly(
        "com/google/foo/Foo",
        "com/google/bar/Bar",
        "com/google/test/Test"
    );
  }
}

