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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PackageIdentifier}. */
@RunWith(JUnit4.class)
public class PackageIdentifierTest {
  @Test
  public void testParsing() throws Exception {
    PackageIdentifier fooA = PackageIdentifier.parse("@foo//a");
    assertThat(fooA.getRepository().getName()).isEqualTo("foo");
    assertThat(fooA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(fooA.getPackagePath(false)).isEqualTo(PathFragment.create("external/foo/a"));
    assertThat(fooA.getPackagePath(true)).isEqualTo(PathFragment.create("a"));

    PackageIdentifier absoluteA = PackageIdentifier.parse("//a");
    assertThat(absoluteA.getRepository().getName()).isEmpty();
    assertThat(absoluteA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(absoluteA.getPackagePath(false)).isEqualTo(PathFragment.create("a"));
    assertThat(absoluteA.getPackagePath(true)).isEqualTo(PathFragment.create("a"));

    PackageIdentifier plainA = PackageIdentifier.parse("a");
    assertThat(plainA.getRepository().getName()).isEmpty();
    assertThat(plainA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(plainA.getPackagePath(false)).isEqualTo(PathFragment.create("a"));
    assertThat(plainA.getPackagePath(true)).isEqualTo(PathFragment.create("a"));

    PackageIdentifier mainA = PackageIdentifier.parse("@//a");
    assertThat(mainA.getRepository()).isEqualTo(RepositoryName.MAIN);
    assertThat(mainA.getPackageFragment().getPathString()).isEqualTo("a");
    assertThat(mainA.getPackagePath(false)).isEqualTo(PathFragment.create("a"));
    assertThat(mainA.getPackagePath(true)).isEqualTo(PathFragment.create("a"));
  }

  @Test
  public void testToString() throws Exception {
    PackageIdentifier local = PackageIdentifier.create("", PathFragment.create("bar/baz"));
    assertThat(local.toString()).isEqualTo("bar/baz");
    PackageIdentifier external = PackageIdentifier.create("foo", PathFragment.create("bar/baz"));
    assertThat(external.toString()).isEqualTo("@@foo//bar/baz");
  }

  @Test
  public void testCompareTo() throws Exception {
    PackageIdentifier foo1 = PackageIdentifier.create("foo", PathFragment.create("bar/baz"));
    PackageIdentifier foo2 = PackageIdentifier.create("foo", PathFragment.create("bar/baz"));
    PackageIdentifier foo3 = PackageIdentifier.create("foo", PathFragment.create("bar/bz"));
    PackageIdentifier bar = PackageIdentifier.create("bar", PathFragment.create("bar/baz"));
    assertThat(foo1.compareTo(foo2)).isEqualTo(0);
    assertThat(foo1.compareTo(foo3)).isLessThan(0);
    assertThat(foo1.compareTo(bar)).isGreaterThan(0);
  }

  @Test
  public void testInvalidPackageName() throws Exception {
    // This shouldn't throw an exception, package names aren't validated.
    PackageIdentifier.create("foo", PathFragment.create("bar.baz"));
  }

  @Test
  public void testPackageFragmentEquality() throws Exception {
    // Make sure package fragments are canonicalized.
    PackageIdentifier p1 = PackageIdentifier.create("whatever", PathFragment.create("foo/bar"));
    PackageIdentifier p2 = PackageIdentifier.create("whatever", PathFragment.create("foo/bar"));
    assertThat(p1.getPackageFragment()).isSameInstanceAs(p2.getPackageFragment());
  }

  @Test
  public void testRunfilesDir() throws Exception {
    assertThat(PackageIdentifier.create("foo", PathFragment.create("bar/baz")).getRunfilesPath())
        .isEqualTo(PathFragment.create("../foo/bar/baz"));
    assertThat(PackageIdentifier.create("", PathFragment.create("bar/baz")).getRunfilesPath())
        .isEqualTo(PathFragment.create("bar/baz"));
  }

  @Test
  public void testUnambiguousCanonicalForm() throws Exception {
    assertThat(PackageIdentifier.createInMainRepo("foo/bar").getUnambiguousCanonicalForm())
        .isEqualTo("@@//foo/bar");
    assertThat(
            PackageIdentifier.create("foo", PathFragment.create("bar"))
                .getUnambiguousCanonicalForm())
        .isEqualTo("@@foo//bar");
    assertThat(
            PackageIdentifier.create(
                    RepositoryName.create("foo").toNonVisible(RepositoryName.create("bar")),
                    PathFragment.create("baz"))
                .getUnambiguousCanonicalForm())
        .isEqualTo("@@[unknown repo 'foo' requested from @@bar]//baz");
  }

  @Test
  public void testDisplayFormInMainRepository() throws Exception {
    PackageIdentifier pkg =
        PackageIdentifier.create(RepositoryName.MAIN, PathFragment.create("some/pkg"));

    assertThat(pkg.getDisplayForm(RepositoryMapping.ALWAYS_FALLBACK)).isEqualTo("//some/pkg");
    assertThat(
            pkg.getDisplayForm(
                RepositoryMapping.create(
                    ImmutableMap.of("foo", RepositoryName.create("bar")), RepositoryName.MAIN)))
        .isEqualTo("//some/pkg");
  }

  @Test
  public void testDisplayFormInExternalRepository() throws Exception {
    RepositoryName repo = RepositoryName.create("canonical");
    PackageIdentifier pkg = PackageIdentifier.create(repo, PathFragment.create("some/pkg"));

    assertThat(pkg.getDisplayForm(RepositoryMapping.ALWAYS_FALLBACK))
        .isEqualTo("@canonical//some/pkg");
    assertThat(
            pkg.getDisplayForm(
                RepositoryMapping.create(ImmutableMap.of("local", repo), RepositoryName.MAIN)))
        .isEqualTo("@local//some/pkg");
    assertThat(
            pkg.getDisplayForm(
                RepositoryMapping.create(
                    ImmutableMap.of("local", RepositoryName.create("other_repo")),
                    RepositoryName.MAIN)))
        .isEqualTo("@@canonical//some/pkg");
  }
}
