// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link PackageIdentifier}.
 */
@RunWith(JUnit4.class)
public class PackageIdentifierTest {
  @Test
  public void testParsing() throws Exception {
    PackageIdentifier fooA = PackageIdentifier.parse("@foo//a");
    assertThat(fooA.getRepository().strippedName()).isEqualTo("foo");
    assertThat(fooA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(fooA.getRepository().getSourceRoot()).isEqualTo(PathFragment.create("foo"));
    assertThat(fooA.getRepository().getPackagePath())
        .isEqualTo(PathFragment.create("external/foo"));

    PackageIdentifier absoluteA = PackageIdentifier.parse("//a");
    assertThat(absoluteA.getRepository().strippedName()).isEmpty();
    assertThat(absoluteA.getPackageFragment().getPathString()).isEqualTo("a");

    PackageIdentifier plainA = PackageIdentifier.parse("a");
    assertThat(plainA.getRepository().strippedName()).isEmpty();
    assertThat(plainA.getPackageFragment().getPathString()).isEqualTo("a");

    PackageIdentifier mainA = PackageIdentifier.parse("@//a");
    assertThat(mainA.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(mainA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(mainA.getRepository().getSourceRoot()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void testToString() throws Exception {
    PackageIdentifier local = PackageIdentifier.create("", PathFragment.create("bar/baz"));
    assertThat(local.toString()).isEqualTo("bar/baz");
    PackageIdentifier external = PackageIdentifier.create("@foo", PathFragment.create("bar/baz"));
    assertThat(external.toString()).isEqualTo("@foo//bar/baz");
  }

  @Test
  public void testCompareTo() throws Exception {
    PackageIdentifier foo1 = PackageIdentifier.create("@foo", PathFragment.create("bar/baz"));
    PackageIdentifier foo2 = PackageIdentifier.create("@foo", PathFragment.create("bar/baz"));
    PackageIdentifier foo3 = PackageIdentifier.create("@foo", PathFragment.create("bar/bz"));
    PackageIdentifier bar = PackageIdentifier.create("@bar", PathFragment.create("bar/baz"));
    assertThat(foo1.compareTo(foo2)).isEqualTo(0);
    assertThat(foo1.compareTo(foo3)).isLessThan(0);
    assertThat(foo1.compareTo(bar)).isGreaterThan(0);
  }

  @Test
  public void testInvalidPackageName() throws Exception {
    // This shouldn't throw an exception, package names aren't validated.
    PackageIdentifier.create("@foo", PathFragment.create("bar.baz"));
  }

  @Test
  public void testPackageFragmentEquality() throws Exception {
    // Make sure package fragments are canonicalized.
    PackageIdentifier p1 = PackageIdentifier.create("@whatever", PathFragment.create("foo/bar"));
    PackageIdentifier p2 = PackageIdentifier.create("@whatever", PathFragment.create("foo/bar"));
    assertThat(p1.getPackageFragment()).isSameInstanceAs(p2.getPackageFragment());
  }

  @Test
  public void testRunfilesDir() throws Exception {
    assertThat(PackageIdentifier.create("@foo", PathFragment.create("bar/baz")).getRunfilesPath())
        .isEqualTo(PathFragment.create("../foo/bar/baz"));
    assertThat(PackageIdentifier.create("@", PathFragment.create("bar/baz")).getRunfilesPath())
        .isEqualTo(PathFragment.create("bar/baz"));
  }
}
