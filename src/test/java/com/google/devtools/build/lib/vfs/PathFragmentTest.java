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
package com.google.devtools.build.lib.vfs;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.vfs.PathFragment.EMPTY_FRAGMENT;
import static com.google.devtools.build.lib.vfs.PathFragment.create;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.File;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link PathFragment}. */
@RunWith(TestParameterInjector.class)
public final class PathFragmentTest {

  @Test
  public void testEqualsAndHashCode() {
    InMemoryFileSystem filesystem = new InMemoryFileSystem(DigestHashFunction.SHA256);

    new EqualsTester()
        .addEqualityGroup(
            create("../relative/path"),
            create("..").getRelative("relative").getRelative("path"),
            create(new File("../relative/path").getPath()))
        .addEqualityGroup(create("something/else"))
        .addEqualityGroup(create("/something/else"))
        .addEqualityGroup(create("/"), create("//////"))
        .addEqualityGroup(create(""), PathFragment.EMPTY_FRAGMENT)
        .addEqualityGroup(filesystem.getPath("/")) // A Path object.
        .testEquals();
  }

  @Test
  public void testHashCodeCache() {
    PathFragment relativePath = create("../relative/path");
    PathFragment rootPath = create("/");

    int oldResult = relativePath.hashCode();
    int rootResult = rootPath.hashCode();
    assertThat(relativePath.hashCode()).isEqualTo(oldResult);
    assertThat(rootPath.hashCode()).isEqualTo(rootResult);
  }

  @Test
  public void testRelativeTo() {
    assertThat(create("foo/bar/baz").relativeTo("foo").getPathString()).isEqualTo("bar/baz");
    assertThat(create("/foo/bar/baz").relativeTo("/foo").getPathString()).isEqualTo("bar/baz");
    assertThat(create("foo/bar/baz").relativeTo("foo/bar").getPathString()).isEqualTo("baz");
    assertThat(create("/foo/bar/baz").relativeTo("/foo/bar").getPathString()).isEqualTo("baz");
    assertThat(create("/foo").relativeTo("/").getPathString()).isEqualTo("foo");
    assertThat(create("foo").relativeTo("").getPathString()).isEqualTo("foo");
    assertThat(create("foo/bar").relativeTo("").getPathString()).isEqualTo("foo/bar");
  }

  @Test
  public void testIsAbsolute() {
    assertThat(create("/absolute/test").isAbsolute()).isTrue();
    assertThat(create("relative/test").isAbsolute()).isFalse();
    assertThat(create(new File("/absolute/test").getPath()).isAbsolute()).isTrue();
    assertThat(create(new File("relative/test").getPath()).isAbsolute()).isFalse();
  }

  @Test
  public void testIsNormalized() {
    assertThat(PathFragment.isNormalized("/absolute/path")).isTrue();
    assertThat(PathFragment.isNormalized("some//path")).isTrue();
    assertThat(PathFragment.isNormalized("some/./path")).isFalse();
    assertThat(PathFragment.isNormalized("../some/path")).isFalse();
    assertThat(PathFragment.isNormalized("./some/path")).isFalse();
    assertThat(PathFragment.isNormalized("some/path/..")).isFalse();
    assertThat(PathFragment.isNormalized("some/path/.")).isFalse();
    assertThat(PathFragment.isNormalized("some/other/../path")).isFalse();
    assertThat(PathFragment.isNormalized("some/other//tricky..path..")).isTrue();
    assertThat(PathFragment.isNormalized("/some/other//tricky..path..")).isTrue();
  }

  @Test
  public void testContainsUpLevelReferences() {
    assertThat(PathFragment.containsUplevelReferences("/absolute/path")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("some//path")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("some/./path")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("../some/path")).isTrue();
    assertThat(PathFragment.containsUplevelReferences("./some/path")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("some/path/..")).isTrue();
    assertThat(PathFragment.containsUplevelReferences("some/path/.")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("some/other/../path")).isTrue();
    assertThat(PathFragment.containsUplevelReferences("some/other//tricky..path..")).isFalse();
    assertThat(PathFragment.containsUplevelReferences("/some/other//tricky..path..")).isFalse();

    // Normalization cannot remove leading uplevel references, so this will be true
    assertThat(create("../some/path").containsUplevelReferences()).isTrue();
    // Normalization will remove these, so no uplevel references left
    assertThat(create("some/path/..").containsUplevelReferences()).isFalse();
  }

  @Test
  public void testRootNodeReturnsRootString() {
    PathFragment rootFragment = create("/");
    assertThat(rootFragment.getPathString()).isEqualTo("/");
  }

  @Test
  public void testGetRelative() {
    assertThat(create("a").getRelative("b").getPathString()).isEqualTo("a/b");
    assertThat(create("a/b").getRelative("c/d").getPathString()).isEqualTo("a/b/c/d");
    assertThat(create("c/d").getRelative("/a/b").getPathString()).isEqualTo("/a/b");
    assertThat(create("a").getRelative("").getPathString()).isEqualTo("a");
    assertThat(create("/").getRelative("").getPathString()).isEqualTo("/");
    assertThat(create("a/b").getRelative("../foo").getPathString()).isEqualTo("a/foo");
    assertThat(create("/a/b").getRelative("../foo").getPathString()).isEqualTo("/a/foo");

    // Make sure any fast path of PathFragment#getRelative(PathFragment) works
    assertThat(create("a/b").getRelative(create("../foo")).getPathString()).isEqualTo("a/foo");
    assertThat(create("/a/b").getRelative(create("../foo")).getPathString()).isEqualTo("/a/foo");

    // Make sure any fast path of PathFragment#getRelative(PathFragment) works
    assertThat(create("c/d").getRelative(create("/a/b")).getPathString()).isEqualTo("/a/b");

    // Test normalization
    assertThat(create("a").getRelative(".").getPathString()).isEqualTo("a");
  }

  @Test
  public void testIsNormalizedRelativePath() {
    assertThat(PathFragment.isNormalizedRelativePath("/a")).isFalse();
    assertThat(PathFragment.isNormalizedRelativePath("a///b")).isFalse();
    assertThat(PathFragment.isNormalizedRelativePath("../a")).isFalse();
    assertThat(PathFragment.isNormalizedRelativePath("a/../b")).isFalse();
    assertThat(PathFragment.isNormalizedRelativePath("a/b")).isTrue();
    assertThat(PathFragment.isNormalizedRelativePath("ab")).isTrue();
  }

  @Test
  public void testContainsSeparator() {
    assertThat(PathFragment.containsSeparator("/a")).isTrue();
    assertThat(PathFragment.containsSeparator("a///b")).isTrue();
    assertThat(PathFragment.containsSeparator("../a")).isTrue();
    assertThat(PathFragment.containsSeparator("a/../b")).isTrue();
    assertThat(PathFragment.containsSeparator("a/b")).isTrue();
    assertThat(PathFragment.containsSeparator("ab")).isFalse();
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = create("../some/path");
    assertThat(pf.getChild("hi")).isEqualTo(create("../some/path/hi"));
    assertThat(pf.getChild("h\\i")).isEqualTo(create("../some/path/h\\i"));
    assertThat(create("../some/path").getChild(".hi")).isEqualTo(create("../some/path/.hi"));
    assertThat(create("../some/path").getChild("..hi")).isEqualTo(create("../some/path/..hi"));
  }

  @Test
  public void testGetChildRejectsInvalidBaseNames() {
    PathFragment pf = create("../some/path");
    assertThrows(IllegalArgumentException.class, () -> pf.getChild("."));
    assertThrows(IllegalArgumentException.class, () -> pf.getChild(".."));
    assertThrows(IllegalArgumentException.class, () -> pf.getChild("x/y"));
    assertThrows(IllegalArgumentException.class, () -> pf.getChild("/y"));
    assertThrows(IllegalArgumentException.class, () -> pf.getChild("y/"));
    assertThrows(IllegalArgumentException.class, () -> pf.getChild(""));
  }

  @Test
  public void testEmptyPathToEmptyPath() {
    assertThat(create("/").getPathString()).isEqualTo("/");
    assertThat(create("").getPathString()).isEqualTo("");
  }

  @Test
  public void testRedundantSlashes() {
    assertThat(create("///").getPathString()).isEqualTo("/");
    assertThat(create("/foo///bar").getPathString()).isEqualTo("/foo/bar");
    assertThat(create("////foo//bar").getPathString()).isEqualTo("/foo/bar");
  }

  @Test
  public void testSimpleNameToSimpleName() {
    assertThat(create("/foo").getPathString()).isEqualTo("/foo");
    assertThat(create("foo").getPathString()).isEqualTo("foo");
  }

  @Test
  public void testSimplePathToSimplePath() {
    assertThat(create("/foo/bar").getPathString()).isEqualTo("/foo/bar");
    assertThat(create("foo/bar").getPathString()).isEqualTo("foo/bar");
  }

  @Test
  public void testStripsTrailingSlash() {
    assertThat(create("/foo/bar/").getPathString()).isEqualTo("/foo/bar");
  }

  @Test
  public void testGetParentDirectory() {
    PathFragment fooBarWiz = create("foo/bar/wiz");
    PathFragment fooBar = create("foo/bar");
    PathFragment foo = create("foo");
    PathFragment empty = create("");
    assertThat(fooBarWiz.getParentDirectory()).isEqualTo(fooBar);
    assertThat(fooBar.getParentDirectory()).isEqualTo(foo);
    assertThat(foo.getParentDirectory()).isEqualTo(empty);
    assertThat(empty.getParentDirectory()).isNull();

    PathFragment fooBarWizAbs = create("/foo/bar/wiz");
    PathFragment fooBarAbs = create("/foo/bar");
    PathFragment fooAbs = create("/foo");
    PathFragment rootAbs = create("/");
    assertThat(fooBarWizAbs.getParentDirectory()).isEqualTo(fooBarAbs);
    assertThat(fooBarAbs.getParentDirectory()).isEqualTo(fooAbs);
    assertThat(fooAbs.getParentDirectory()).isEqualTo(rootAbs);
    assertThat(rootAbs.getParentDirectory()).isNull();
  }

  @Test
  public void testSegmentsCount() {
    assertThat(create("foo/bar").segmentCount()).isEqualTo(2);
    assertThat(create("/foo/bar").segmentCount()).isEqualTo(2);
    assertThat(create("foo//bar").segmentCount()).isEqualTo(2);
    assertThat(create("/foo//bar").segmentCount()).isEqualTo(2);
    assertThat(create("foo/").segmentCount()).isEqualTo(1);
    assertThat(create("/foo/").segmentCount()).isEqualTo(1);
    assertThat(create("foo").segmentCount()).isEqualTo(1);
    assertThat(create("/foo").segmentCount()).isEqualTo(1);
    assertThat(create("/").segmentCount()).isEqualTo(0);
    assertThat(create("").segmentCount()).isEqualTo(0);
  }

  @Test
  public void isSingleSegment_true(@TestParameter({"/foo", "foo"}) String path) {
    assertThat(create(path).isSingleSegment()).isTrue();
  }

  @Test
  public void isSingleSegment_false(
      @TestParameter({"/", "", "/foo/bar", "foo/bar", "/foo/bar/baz", "foo/bar/baz"}) String path) {
    assertThat(create(path).isSingleSegment()).isFalse();
  }

  @Test
  public void isMultiSegment_true(
      @TestParameter({"/foo/bar", "foo/bar", "/foo/bar/baz", "foo/bar/baz"}) String path) {
    assertThat(create(path).isMultiSegment()).isTrue();
  }

  @Test
  public void isMultiSegment_false(@TestParameter({"/", "", "/foo", "foo"}) String path) {
    assertThat(create(path).isMultiSegment()).isFalse();
  }

  @Test
  public void testGetSegment() {
    assertThat(create("foo/bar").getSegment(0)).isEqualTo("foo");
    assertThat(create("foo/bar").getSegment(1)).isEqualTo("bar");
    assertThat(create("/foo/bar").getSegment(0)).isEqualTo("foo");
    assertThat(create("/foo/bar").getSegment(1)).isEqualTo("bar");
    assertThat(create("foo/").getSegment(0)).isEqualTo("foo");
    assertThat(create("/foo/").getSegment(0)).isEqualTo("foo");
    assertThat(create("foo").getSegment(0)).isEqualTo("foo");
    assertThat(create("/foo").getSegment(0)).isEqualTo("foo");
  }

  @Test
  public void segments() {
    assertThat(create("/this/is/a/path").segments())
        .containsExactly("this", "is", "a", "path")
        .inOrder();
  }

  @Test
  public void testBasename() {
    assertThat(create("foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("/foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("foo/").getBaseName()).isEqualTo("foo");
    assertThat(create("/foo/").getBaseName()).isEqualTo("foo");
    assertThat(create("foo").getBaseName()).isEqualTo("foo");
    assertThat(create("/foo").getBaseName()).isEqualTo("foo");
    assertThat(create("/").getBaseName()).isEmpty();
    assertThat(create("").getBaseName()).isEmpty();
  }

  @Test
  public void testFileExtension() {
    assertThat(create("foo.bar").getFileExtension()).isEqualTo("bar");
    assertThat(create("foo.barr").getFileExtension()).isEqualTo("barr");
    assertThat(create("foo.b").getFileExtension()).isEqualTo("b");
    assertThat(create("foo.").getFileExtension()).isEmpty();
    assertThat(create("foo").getFileExtension()).isEmpty();
    assertThat(create(".").getFileExtension()).isEmpty();
    assertThat(create("").getFileExtension()).isEmpty();
    assertThat(create("foo/bar.baz").getFileExtension()).isEqualTo("baz");
    assertThat(create("foo.bar.baz").getFileExtension()).isEqualTo("baz");
    assertThat(create("foo.bar/baz").getFileExtension()).isEmpty();
  }

  @Test
  public void testReplaceName() {
    assertThat(create("foo/bar").replaceName("baz").getPathString()).isEqualTo("foo/baz");
    assertThat(create("/foo/bar").replaceName("baz").getPathString()).isEqualTo("/foo/baz");
    assertThat(create("foo/bar").replaceName("").getPathString()).isEqualTo("foo");
    assertThat(create("foo/").replaceName("baz").getPathString()).isEqualTo("baz");
    assertThat(create("/foo/").replaceName("baz").getPathString()).isEqualTo("/baz");
    assertThat(create("foo").replaceName("baz").getPathString()).isEqualTo("baz");
    assertThat(create("/foo").replaceName("baz").getPathString()).isEqualTo("/baz");
    assertThat(create("/").replaceName("baz")).isNull();
    assertThat(create("/").replaceName("")).isNull();
    assertThat(create("").replaceName("baz")).isNull();
    assertThat(create("").replaceName("")).isNull();

    assertThat(create("foo/bar").replaceName("bar/baz").getPathString()).isEqualTo("foo/bar/baz");
    assertThat(create("foo/bar").replaceName("bar/baz/").getPathString()).isEqualTo("foo/bar/baz");

    // Absolute path arguments will clobber the original path.
    assertThat(create("foo/bar").replaceName("/absolute").getPathString()).isEqualTo("/absolute");
    assertThat(create("foo/bar").replaceName("/").getPathString()).isEqualTo("/");
  }

  @Test
  public void testSubFragment() {
    assertThat(create("/foo/bar/baz").subFragment(0, 3).getPathString()).isEqualTo("/foo/bar/baz");
    assertThat(create("foo/bar/baz").subFragment(0, 3).getPathString()).isEqualTo("foo/bar/baz");
    assertThat(create("/foo/bar/baz").subFragment(0, 2).getPathString()).isEqualTo("/foo/bar");
    assertThat(create("/foo/bar/baz").subFragment(1, 3).getPathString()).isEqualTo("bar/baz");
    assertThat(create("/foo/bar/baz").subFragment(0, 1).getPathString()).isEqualTo("/foo");
    assertThat(create("/foo/bar/baz").subFragment(1, 2).getPathString()).isEqualTo("bar");
    assertThat(create("/foo/bar/baz").subFragment(2, 3).getPathString()).isEqualTo("baz");
    assertThat(create("/foo/bar/baz").subFragment(0, 0).getPathString()).isEqualTo("/");
    assertThat(create("foo/bar/baz").subFragment(0, 0).getPathString()).isEqualTo("");
    assertThat(create("foo/bar/baz").subFragment(1, 1).getPathString()).isEqualTo("");

    assertThat(create("/foo/bar/baz").subFragment(0).getPathString()).isEqualTo("/foo/bar/baz");
    assertThat(create("foo/bar/baz").subFragment(0).getPathString()).isEqualTo("foo/bar/baz");
    assertThat(create("/foo/bar/baz").subFragment(1).getPathString()).isEqualTo("bar/baz");
    assertThat(create("foo/bar/baz").subFragment(1).getPathString()).isEqualTo("bar/baz");
    assertThat(create("foo/bar/baz").subFragment(2).getPathString()).isEqualTo("baz");
    assertThat(create("foo/bar/baz").subFragment(3).getPathString()).isEqualTo("");

    assertThrows(IndexOutOfBoundsException.class, () -> create("foo/bar/baz").subFragment(3, 2));
    assertThrows(IndexOutOfBoundsException.class, () -> create("foo/bar/baz").subFragment(4, 4));
    assertThrows(IndexOutOfBoundsException.class, () -> create("foo/bar/baz").subFragment(3, 2));
    assertThrows(IndexOutOfBoundsException.class, () -> create("foo/bar/baz").subFragment(4));
  }

  @Test
  public void testStartsWith() {
    PathFragment foobar = create("/foo/bar");
    PathFragment foobarRelative = create("foo/bar");

    // (path, prefix) => true
    assertThat(foobar.startsWith(foobar)).isTrue();
    assertThat(foobar.startsWith(create("/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/"))).isTrue();
    assertThat(foobar.startsWith(create("/foo/bar/"))).isTrue(); // Includes trailing slash.

    // (prefix, path) => false
    assertThat(create("/foo").startsWith(foobar)).isFalse();
    assertThat(create("/").startsWith(foobar)).isFalse();

    // (absolute, relative) => false
    assertThat(foobar.startsWith(foobarRelative)).isFalse();
    assertThat(foobarRelative.startsWith(foobar)).isFalse();

    // (relative path, relative prefix) => true
    assertThat(foobarRelative.startsWith(foobarRelative)).isTrue();
    assertThat(foobarRelative.startsWith(create("foo"))).isTrue();
    assertThat(foobarRelative.startsWith(create(""))).isTrue();

    // (path, sibling) => false
    assertThat(create("/foo/wiz").startsWith(foobar)).isFalse();
    assertThat(foobar.startsWith(create("/foo/wiz"))).isFalse();
  }

  @Test
  public void testCheckAllPathsStartWithButAreNotEqualTo() {
    // Check passes:
    PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a/c"), create("a"));

    // Check trivially passes:
    PathFragment.checkAllPathsAreUnder(ImmutableList.of(), create("a"));

    // Check fails when some path does not start with startingWithPath:
    assertThrows(
        IllegalArgumentException.class,
        () -> PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "b/c"), create("a")));

    // Check fails when some path is equal to startingWithPath:
    assertThrows(
        IllegalArgumentException.class,
        () -> PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a"), create("a")));
  }

  @Test
  public void testEndsWith() {
    PathFragment foobar = create("/foo/bar");
    PathFragment foobarRelative = create("foo/bar");

    // (path, suffix) => true
    assertThat(foobar.endsWith(foobar)).isTrue();
    assertThat(foobar.endsWith(create("bar"))).isTrue();
    assertThat(foobar.endsWith(create("foo/bar"))).isTrue();
    assertThat(foobar.endsWith(create("/foo/bar"))).isTrue();
    assertThat(foobar.endsWith(create("/bar"))).isFalse();

    // (prefix, path) => false
    assertThat(create("/foo").endsWith(foobar)).isFalse();
    assertThat(create("/").endsWith(foobar)).isFalse();

    // (suffix, path) => false
    assertThat(create("/bar").endsWith(foobar)).isFalse();
    assertThat(create("bar").endsWith(foobar)).isFalse();
    assertThat(create("").endsWith(foobar)).isFalse();

    // (absolute, relative) => true
    assertThat(foobar.endsWith(foobarRelative)).isTrue();

    // (relative, absolute) => false
    assertThat(foobarRelative.endsWith(foobar)).isFalse();

    // (relative path, relative prefix) => true
    assertThat(foobarRelative.endsWith(foobarRelative)).isTrue();
    assertThat(foobarRelative.endsWith(create("bar"))).isTrue();
    assertThat(foobarRelative.endsWith(create(""))).isTrue();

    // (path, sibling) => false
    assertThat(create("/foo/wiz").endsWith(foobar)).isFalse();
    assertThat(foobar.endsWith(create("/foo/wiz"))).isFalse();
  }

  @Test
  public void testToRelative() {
    assertThat(create("/foo/bar").toRelative()).isEqualTo(create("foo/bar"));
    assertThat(create("/").toRelative()).isEqualTo(create(""));
    assertThrows(IllegalArgumentException.class, () -> create("foo").toRelative());
  }

  @Test
  public void testGetDriveStr() {
    assertThat(create("/foo/bar").getDriveStr()).isEqualTo("/");
    assertThat(create("/").getDriveStr()).isEqualTo("/");
    assertThrows(IllegalArgumentException.class, () -> create("foo").getDriveStr());
  }

  private static List<PathFragment> toPaths(List<String> strs) {
    List<PathFragment> paths = Lists.newArrayList();
    for (String s : strs) {
      paths.add(create(s));
    }
    return paths;
  }

  private static ImmutableSet<PathFragment> toPathsSet(String... strs) {
    ImmutableSet.Builder<PathFragment> builder = ImmutableSet.builder();
    for (String str : strs) {
      builder.add(create(str));
    }
    return builder.build();
  }

  @Test
  public void testCompareTo() {
    List<String> pathStrs =
        ImmutableList.of(
            "",
            "/",
            "foo",
            "/foo",
            "foo/bar",
            "foo.bar",
            "foo/bar.baz",
            "foo/bar/baz",
            "foo/barfile",
            "foo/Bar",
            "Foo/bar");
    List<PathFragment> paths = toPaths(pathStrs);
    // First test that compareTo is self-consistent.
    for (PathFragment x : paths) {
      for (PathFragment y : paths) {
        for (PathFragment z : paths) {
          // Anti-symmetry
          assertThat(-1 * Integer.signum(y.compareTo(x))).isEqualTo(Integer.signum(x.compareTo(y)));
          // Transitivity
          if (x.compareTo(y) > 0 && y.compareTo(z) > 0) {
            assertThat(x.compareTo(z)).isGreaterThan(0);
          }
          // "Substitutability"
          if (x.compareTo(y) == 0) {
            assertThat(Integer.signum(y.compareTo(z))).isEqualTo(Integer.signum(x.compareTo(z)));
          }
          // Consistency with equals
          assertThat(x.equals(y)).isEqualTo((x.compareTo(y) == 0));
        }
      }
    }
    // Now test that compareTo does what we expect.  The exact ordering here doesn't matter much.
    Collections.shuffle(paths);
    Collections.sort(paths);
    List<PathFragment> expectedOrder =
        toPaths(
            ImmutableList.of(
                "",
                "/",
                "/foo",
                "Foo/bar",
                "foo",
                "foo.bar",
                "foo/Bar",
                "foo/bar",
                "foo/bar.baz",
                "foo/bar/baz",
                "foo/barfile"));
    assertThat(paths).isEqualTo(expectedOrder);
  }

  @Test
  public void testGetSafePathString() {
    assertThat(create("/").getSafePathString()).isEqualTo("/");
    assertThat(create("/abc").getSafePathString()).isEqualTo("/abc");
    assertThat(create("").getSafePathString()).isEqualTo(".");
    assertThat(PathFragment.EMPTY_FRAGMENT.getSafePathString()).isEqualTo(".");
    assertThat(create("abc/def").getSafePathString()).isEqualTo("abc/def");
  }

  @Test
  public void testNormalize() {
    assertThat(create("/a/b")).isEqualTo(create("/a/b"));
    assertThat(create("/a/./b")).isEqualTo(create("/a/b"));
    assertThat(create("/a/../b")).isEqualTo(create("/b"));
    assertThat(create("a/b")).isEqualTo(create("a/b"));
    assertThat(create("a/../../b")).isEqualTo(create("../b"));
    assertThat(create("a/../..")).isEqualTo(create(".."));
    assertThat(create("a/../b")).isEqualTo(create("b"));
    assertThat(create("a/b/../b")).isEqualTo(create("a/b"));
    assertThat(create("/..")).isEqualTo(create("/.."));
    assertThat(create("..")).isEqualTo(create(".."));
  }

  @Test
  public void testSegments() {
    assertThat(create("").segmentCount()).isEqualTo(0);
    assertThat(create("a").segmentCount()).isEqualTo(1);
    assertThat(create("a/b").segmentCount()).isEqualTo(2);
    assertThat(create("a/b/c").segmentCount()).isEqualTo(3);
    assertThat(create("/").segmentCount()).isEqualTo(0);
    assertThat(create("/a").segmentCount()).isEqualTo(1);
    assertThat(create("/a/b").segmentCount()).isEqualTo(2);
    assertThat(create("/a/b/c").segmentCount()).isEqualTo(3);

    assertThat(create("").splitToListOfSegments()).isEmpty();
    assertThat(create("a").splitToListOfSegments()).containsExactly("a").inOrder();
    assertThat(create("a/b").splitToListOfSegments()).containsExactly("a", "b").inOrder();
    assertThat(create("a/b/c").splitToListOfSegments()).containsExactly("a", "b", "c").inOrder();
    assertThat(create("/").splitToListOfSegments()).isEmpty();
    assertThat(create("/a").splitToListOfSegments()).containsExactly("a").inOrder();
    assertThat(create("/a/b").splitToListOfSegments()).containsExactly("a", "b").inOrder();
    assertThat(create("/a/b/c").splitToListOfSegments()).containsExactly("a", "b", "c").inOrder();

    assertThat(create("a").getSegment(0)).isEqualTo("a");
    assertThat(create("a/b").getSegment(0)).isEqualTo("a");
    assertThat(create("a/b").getSegment(1)).isEqualTo("b");
    assertThat(create("a/b/c").getSegment(2)).isEqualTo("c");
    assertThat(create("/a").getSegment(0)).isEqualTo("a");
    assertThat(create("/a/b").getSegment(0)).isEqualTo("a");
    assertThat(create("/a/b").getSegment(1)).isEqualTo("b");
    assertThat(create("/a/b/c").getSegment(2)).isEqualTo("c");

    assertThrows(IllegalArgumentException.class, () -> create("").getSegment(0));
    assertThrows(IllegalArgumentException.class, () -> create("a/b").getSegment(2));
  }

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            ImmutableList.of("", "a", "/foo", "foo/bar/baz", "/a/path/fragment/with/lots/of/parts")
                .stream()
                .map(PathFragment::create)
                .collect(toImmutableList()))
        .runTests();
  }

  @Test
  public void testSerializationSimple() throws Exception {
    checkSerialization("a", 6);
  }

  @Test
  public void testSerializationAbsolute() throws Exception {
    checkSerialization("/foo", 9);
   }

  @Test
  public void testSerializationNested() throws Exception {
    checkSerialization("foo/bar/baz", 16);
  }

  private static void checkSerialization(String pathFragmentString, int expectedSize)
      throws Exception {
    PathFragment a = create(pathFragmentString);
    ByteString sa =
        TestUtils.toBytes(new SerializationContext(ImmutableClassToInstanceMap.of()), a);
    assertThat(sa.size()).isEqualTo(expectedSize);

    PathFragment a2 =
        (PathFragment)
            TestUtils.fromBytes(new DeserializationContext(ImmutableClassToInstanceMap.of()), sa);
    assertThat(a2).isEqualTo(a);
  }

  @Test
  public void containsUplevelReference_emptyPath_returnsFalse() {
    assertThat(EMPTY_FRAGMENT.containsUplevelReferences()).isFalse();
  }

  @Test
  public void containsUplevelReference_uplevelOnlyPath_returnsTrue() {
    PathFragment pathFragment = create("..");
    assertThat(pathFragment.containsUplevelReferences()).isTrue();
  }

  @Test
  public void containsUplevelReferences_firstSegmentStartingWithDotDot_returnsFalse() {
    PathFragment pathFragment = create("..file");
    assertThat(pathFragment.containsUplevelReferences()).isFalse();
  }

  @Test
  public void containsUplevelReferences_startsWithUplevelReference_returnsTrue() {
    PathFragment pathFragment = create("../file");
    assertThat(pathFragment.containsUplevelReferences()).isTrue();
  }

  @Test
  public void containsUplevelReferences_uplevelReferenceMidPath_normalizesAndReturnsFalse() {
    PathFragment pathFragment = create("a/../b");

    assertThat(pathFragment.containsUplevelReferences()).isFalse();
    assertThat(pathFragment.getPathString()).isEqualTo("b");
  }

  @Test
  public void containsUplevelReferenes_uplevelReferenceMidGlobalPath_normalizesAndReturnsFalse() {
    PathFragment pathFragment = create("/dir1/dir2/../file");

    assertThat(pathFragment.containsUplevelReferences()).isFalse();
    assertThat(pathFragment.getPathString()).isEqualTo("/dir1/file");
  }
}
