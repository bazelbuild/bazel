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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.File;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This class tests the functionality of the PathFragment.
 */
@RunWith(JUnit4.class)
public class PathFragmentTest {
  @Test
  public void testMergeFourPathsWithAbsolute() {
    assertEquals(
        PathFragment.create("x/y/z/a/b/c/d/e"),
        PathFragment.create(
            PathFragment.create("x/y"),
            PathFragment.create("z/a"),
            PathFragment.create("/b/c"),
            PathFragment.create("d/e")));
  }

  @Test
  public void testCreateInternsPathFragments() {
    String[] firstSegments = new String[] {"hello", "world"};
    PathFragment first = PathFragment.create(
        /*driveLetter=*/ '\0', /*isAbsolute=*/ false, firstSegments);

    String[] secondSegments = new String[] {new String("hello"), new String("world")};
    PathFragment second = PathFragment.create(
        /*driveLetter=*/ '\0', /*isAbsolute=*/ false, secondSegments);

    assertThat(first.segmentCount()).isEqualTo(second.segmentCount());
    for (int i = 0; i < first.segmentCount(); i++) {
      assertThat(first.getSegment(i)).isSameAs(second.getSegment(i));
    }
  }

  @Test
  public void testEqualsAndHashCode() {
    InMemoryFileSystem filesystem = new InMemoryFileSystem();

    new EqualsTester()
        .addEqualityGroup(
            PathFragment.create("../relative/path"),
            PathFragment.create("..").getRelative("relative").getRelative("path"),
            PathFragment.createAlreadyInterned(
                '\0', false, new String[] {"..", "relative", "path"}),
            PathFragment.create(new File("../relative/path")))
        .addEqualityGroup(PathFragment.create("something/else"))
        .addEqualityGroup(PathFragment.create("/something/else"))
        .addEqualityGroup(PathFragment.create("/"), PathFragment.create("//////"))
        .addEqualityGroup(PathFragment.create(""), PathFragment.EMPTY_FRAGMENT)
        .addEqualityGroup(filesystem.getRootDirectory()) // A Path object.
        .testEquals();
  }

  @Test
  public void testHashCodeCache() {
    PathFragment relativePath = PathFragment.create("../relative/path");
    PathFragment rootPath = PathFragment.create("/");

    int oldResult = relativePath.hashCode();
    int rootResult = rootPath.hashCode();
    assertEquals(oldResult, relativePath.hashCode());
    assertEquals(rootResult, rootPath.hashCode());
  }

  private void checkRelativeTo(String path, String base) {
    PathFragment relative = PathFragment.create(path).relativeTo(base);
    assertEquals(
        PathFragment.create(path),
        PathFragment.create(base).getRelative(relative).normalize());
  }

  @Test
  public void testRelativeTo() {
    assertPath("bar/baz", PathFragment.create("foo/bar/baz").relativeTo("foo"));
    assertPath("bar/baz", PathFragment.create("/foo/bar/baz").relativeTo("/foo"));
    assertPath("baz", PathFragment.create("foo/bar/baz").relativeTo("foo/bar"));
    assertPath("baz", PathFragment.create("/foo/bar/baz").relativeTo("/foo/bar"));
    assertPath("foo", PathFragment.create("/foo").relativeTo("/"));
    assertPath("foo", PathFragment.create("foo").relativeTo(""));
    assertPath("foo/bar", PathFragment.create("foo/bar").relativeTo(""));

    checkRelativeTo("foo/bar/baz", "foo");
    checkRelativeTo("/foo/bar/baz", "/foo");
    checkRelativeTo("foo/bar/baz", "foo/bar");
    checkRelativeTo("/foo/bar/baz", "/foo/bar");
    checkRelativeTo("/foo", "/");
    checkRelativeTo("foo", "");
    checkRelativeTo("foo/bar", "");
  }

  @Test
  public void testIsAbsolute() {
    assertTrue(PathFragment.create("/absolute/test").isAbsolute());
    assertFalse(PathFragment.create("relative/test").isAbsolute());
    assertTrue(PathFragment.create(new File("/absolute/test")).isAbsolute());
    assertFalse(PathFragment.create(new File("relative/test")).isAbsolute());
  }

  @Test
  public void testIsNormalized() {
    assertTrue(PathFragment.create("/absolute/path").isNormalized());
    assertTrue(PathFragment.create("some//path").isNormalized());
    assertFalse(PathFragment.create("some/./path").isNormalized());
    assertFalse(PathFragment.create("../some/path").isNormalized());
    assertFalse(PathFragment.create("some/other/../path").isNormalized());
    assertTrue(PathFragment.create("some/other//tricky..path..").isNormalized());
    assertTrue(PathFragment.create("/some/other//tricky..path..").isNormalized());
  }

  @Test
  public void testRootNodeReturnsRootString() {
    PathFragment rootFragment = PathFragment.create("/");
    assertEquals("/", rootFragment.getPathString());
  }

  @Test
  public void testGetPathFragmentDoesNotNormalize() {
    String nonCanonicalPath = "/a/weird/noncanonical/../path/.";
    assertEquals(nonCanonicalPath,
        PathFragment.create(nonCanonicalPath).getPathString());
  }

  @Test
  public void testGetRelative() {
    assertEquals("a/b", PathFragment.create("a").getRelative("b").getPathString());
    assertEquals("a/b/c/d", PathFragment.create("a/b").getRelative("c/d").getPathString());
    assertEquals("/a/b", PathFragment.create("c/d").getRelative("/a/b").getPathString());
    assertEquals("a", PathFragment.create("a").getRelative("").getPathString());
    assertEquals("/", PathFragment.create("/").getRelative("").getPathString());
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = PathFragment.create("../some/path");
    assertEquals(PathFragment.create("../some/path/hi"), pf.getChild("hi"));
  }

  @Test
  public void testGetChildRejectsInvalidBaseNames() {
    PathFragment pf = PathFragment.create("../some/path");
    assertGetChildFails(pf, ".");
    assertGetChildFails(pf, "..");
    assertGetChildFails(pf, "x/y");
    assertGetChildFails(pf, "/y");
    assertGetChildFails(pf, "y/");
    assertGetChildFails(pf, "");
  }

  private void assertGetChildFails(PathFragment pf, String baseName) {
    try {
      pf.getChild(baseName);
      fail();
    } catch (Exception e) { /* Expected. */ }
  }

  // Tests after here test the canonicalization
  private void assertRegular(String expected, String actual) {
    // compare string forms
    assertEquals(
        expected,
        PathFragment.create(actual).getPathString());
    // compare fragment forms
    assertEquals(
        PathFragment.create(expected),
        PathFragment.create(actual));
  }

  @Test
  public void testEmptyPathToEmptyPath() {
    assertRegular("/", "/");
    assertRegular("", "");
  }

  @Test
  public void testRedundantSlashes() {
    assertRegular("/", "///");
    assertRegular("/foo/bar", "/foo///bar");
    assertRegular("/foo/bar", "////foo//bar");
  }

  @Test
  public void testSimpleNameToSimpleName() {
    assertRegular("/foo", "/foo");
    assertRegular("foo", "foo");
  }

  @Test
  public void testSimplePathToSimplePath() {
    assertRegular("/foo/bar", "/foo/bar");
    assertRegular("foo/bar", "foo/bar");
  }

  @Test
  public void testStripsTrailingSlash() {
    assertRegular("/foo/bar", "/foo/bar/");
  }

  @Test
  public void testGetParentDirectory() {
    PathFragment fooBarWiz = PathFragment.create("foo/bar/wiz");
    PathFragment fooBar = PathFragment.create("foo/bar");
    PathFragment foo = PathFragment.create("foo");
    PathFragment empty = PathFragment.create("");
    assertEquals(fooBar, fooBarWiz.getParentDirectory());
    assertEquals(foo, fooBar.getParentDirectory());
    assertEquals(empty, foo.getParentDirectory());
    assertNull(empty.getParentDirectory());

    PathFragment fooBarWizAbs = PathFragment.create("/foo/bar/wiz");
    PathFragment fooBarAbs = PathFragment.create("/foo/bar");
    PathFragment fooAbs = PathFragment.create("/foo");
    PathFragment rootAbs = PathFragment.create("/");
    assertEquals(fooBarAbs, fooBarWizAbs.getParentDirectory());
    assertEquals(fooAbs, fooBarAbs.getParentDirectory());
    assertEquals(rootAbs, fooAbs.getParentDirectory());
    assertNull(rootAbs.getParentDirectory());

    // Note, this is surprising but correct behavior:
    assertEquals(fooBarAbs,
                 PathFragment.create("/foo/bar/..").getParentDirectory());
  }

  @Test
  public void testSegmentsCount() {
    assertEquals(2, PathFragment.create("foo/bar").segmentCount());
    assertEquals(2, PathFragment.create("/foo/bar").segmentCount());
    assertEquals(2, PathFragment.create("foo//bar").segmentCount());
    assertEquals(2, PathFragment.create("/foo//bar").segmentCount());
    assertEquals(1, PathFragment.create("foo/").segmentCount());
    assertEquals(1, PathFragment.create("/foo/").segmentCount());
    assertEquals(1, PathFragment.create("foo").segmentCount());
    assertEquals(1, PathFragment.create("/foo").segmentCount());
    assertEquals(0, PathFragment.create("/").segmentCount());
    assertEquals(0, PathFragment.create("").segmentCount());
  }


  @Test
  public void testGetSegment() {
    assertEquals("foo", PathFragment.create("foo/bar").getSegment(0));
    assertEquals("bar", PathFragment.create("foo/bar").getSegment(1));
    assertEquals("foo", PathFragment.create("/foo/bar").getSegment(0));
    assertEquals("bar", PathFragment.create("/foo/bar").getSegment(1));
    assertEquals("foo", PathFragment.create("foo/").getSegment(0));
    assertEquals("foo", PathFragment.create("/foo/").getSegment(0));
    assertEquals("foo", PathFragment.create("foo").getSegment(0));
    assertEquals("foo", PathFragment.create("/foo").getSegment(0));
  }

  @Test
  public void testBasename() throws Exception {
    assertEquals("bar", PathFragment.create("foo/bar").getBaseName());
    assertEquals("bar", PathFragment.create("/foo/bar").getBaseName());
    assertEquals("foo", PathFragment.create("foo/").getBaseName());
    assertEquals("foo", PathFragment.create("/foo/").getBaseName());
    assertEquals("foo", PathFragment.create("foo").getBaseName());
    assertEquals("foo", PathFragment.create("/foo").getBaseName());
    assertThat(PathFragment.create("/").getBaseName()).isEmpty();
    assertThat(PathFragment.create("").getBaseName()).isEmpty();
  }

  @Test
  public void testFileExtension() throws Exception {
    assertThat(PathFragment.create("foo.bar").getFileExtension()).isEqualTo("bar");
    assertThat(PathFragment.create("foo.barr").getFileExtension()).isEqualTo("barr");
    assertThat(PathFragment.create("foo.b").getFileExtension()).isEqualTo("b");
    assertThat(PathFragment.create("foo.").getFileExtension()).isEmpty();
    assertThat(PathFragment.create("foo").getFileExtension()).isEmpty();
    assertThat(PathFragment.create(".").getFileExtension()).isEmpty();
    assertThat(PathFragment.create("").getFileExtension()).isEmpty();
    assertThat(PathFragment.create("foo/bar.baz").getFileExtension()).isEqualTo("baz");
    assertThat(PathFragment.create("foo.bar.baz").getFileExtension()).isEqualTo("baz");
    assertThat(PathFragment.create("foo.bar/baz").getFileExtension()).isEmpty();
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertEquals(expected, actual.getPathString());
  }

  @Test
  public void testReplaceName() throws Exception {
    assertPath("foo/baz", PathFragment.create("foo/bar").replaceName("baz"));
    assertPath("/foo/baz", PathFragment.create("/foo/bar").replaceName("baz"));
    assertPath("foo", PathFragment.create("foo/bar").replaceName(""));
    assertPath("baz", PathFragment.create("foo/").replaceName("baz"));
    assertPath("/baz", PathFragment.create("/foo/").replaceName("baz"));
    assertPath("baz", PathFragment.create("foo").replaceName("baz"));
    assertPath("/baz", PathFragment.create("/foo").replaceName("baz"));
    assertNull(PathFragment.create("/").replaceName("baz"));
    assertNull(PathFragment.create("/").replaceName(""));
    assertNull(PathFragment.create("").replaceName("baz"));
    assertNull(PathFragment.create("").replaceName(""));

    assertPath("foo/bar/baz", PathFragment.create("foo/bar").replaceName("bar/baz"));
    assertPath("foo/bar/baz", PathFragment.create("foo/bar").replaceName("bar/baz/"));

    // Absolute path arguments will clobber the original path.
    assertPath("/absolute", PathFragment.create("foo/bar").replaceName("/absolute"));
    assertPath("/", PathFragment.create("foo/bar").replaceName("/"));
  }
  @Test
  public void testSubFragment() throws Exception {
    assertPath("/foo/bar/baz",
        PathFragment.create("/foo/bar/baz").subFragment(0, 3));
    assertPath("foo/bar/baz",
        PathFragment.create("foo/bar/baz").subFragment(0, 3));
    assertPath("/foo/bar",
               PathFragment.create("/foo/bar/baz").subFragment(0, 2));
    assertPath("bar/baz",
               PathFragment.create("/foo/bar/baz").subFragment(1, 3));
    assertPath("/foo",
               PathFragment.create("/foo/bar/baz").subFragment(0, 1));
    assertPath("bar",
               PathFragment.create("/foo/bar/baz").subFragment(1, 2));
    assertPath("baz", PathFragment.create("/foo/bar/baz").subFragment(2, 3));
    assertPath("/", PathFragment.create("/foo/bar/baz").subFragment(0, 0));
    assertPath("", PathFragment.create("foo/bar/baz").subFragment(0, 0));
    assertPath("", PathFragment.create("foo/bar/baz").subFragment(1, 1));
    try {
      fail("unexpectedly succeeded: " + PathFragment.create("foo/bar/baz").subFragment(3, 2));
    } catch (IndexOutOfBoundsException e) { /* Expected. */ }
    try {
      fail("unexpectedly succeeded: " + PathFragment.create("foo/bar/baz").subFragment(4, 4));
    } catch (IndexOutOfBoundsException e) { /* Expected. */ }
  }

  @Test
  public void testStartsWith() {
    PathFragment foobar = PathFragment.create("/foo/bar");
    PathFragment foobarRelative = PathFragment.create("foo/bar");

    // (path, prefix) => true
    assertTrue(foobar.startsWith(foobar));
    assertTrue(foobar.startsWith(PathFragment.create("/")));
    assertTrue(foobar.startsWith(PathFragment.create("/foo")));
    assertTrue(foobar.startsWith(PathFragment.create("/foo/")));
    assertTrue(foobar.startsWith(PathFragment.create("/foo/bar/")));  // Includes trailing slash.

    // (prefix, path) => false
    assertFalse(PathFragment.create("/foo").startsWith(foobar));
    assertFalse(PathFragment.create("/").startsWith(foobar));

    // (absolute, relative) => false
    assertFalse(foobar.startsWith(foobarRelative));
    assertFalse(foobarRelative.startsWith(foobar));

    // (relative path, relative prefix) => true
    assertTrue(foobarRelative.startsWith(foobarRelative));
    assertTrue(foobarRelative.startsWith(PathFragment.create("foo")));
    assertTrue(foobarRelative.startsWith(PathFragment.create("")));

    // (path, sibling) => false
    assertFalse(PathFragment.create("/foo/wiz").startsWith(foobar));
    assertFalse(foobar.startsWith(PathFragment.create("/foo/wiz")));

    // Does not normalize.
    PathFragment foodotbar = PathFragment.create("foo/./bar");
    assertTrue(foodotbar.startsWith(foodotbar));
    assertTrue(foodotbar.startsWith(PathFragment.create("foo/.")));
    assertTrue(foodotbar.startsWith(PathFragment.create("foo/./")));
    assertTrue(foodotbar.startsWith(PathFragment.create("foo/./bar")));
    assertFalse(foodotbar.startsWith(PathFragment.create("foo/bar")));
  }

  @Test
  public void testFilterPathsStartingWith() {
    // Retains everything:
    ImmutableSet<PathFragment> allUnderA = toPathsSet("a/b", "a/c", "a/d");
    assertThat(PathFragment.filterPathsStartingWith(allUnderA, PathFragment.create("a")))
        .containsExactlyElementsIn(allUnderA);

    // Retains some but not others:
    ImmutableSet<PathFragment> mixed = toPathsSet("a/b", "a/c", "b/c");
    assertThat(PathFragment.filterPathsStartingWith(mixed,
        PathFragment.create("a"))).containsExactlyElementsIn(toPathsSet("a/b", "a/c"));

    // Retains none:
    assertThat(PathFragment.filterPathsStartingWith(allUnderA, PathFragment.create("b"))).isEmpty();

    // Retains paths equal to the startingWithPath:
    assertThat(PathFragment.filterPathsStartingWith(toPathsSet("a"),
        PathFragment.create("a"))).containsExactlyElementsIn(toPathsSet("a"));

    // Retains everything when startingWithPath is the empty fragment:
    assertThat(PathFragment.filterPathsStartingWith(mixed, PathFragment.EMPTY_FRAGMENT))
        .containsExactlyElementsIn(mixed);

    // Supports multi-segment startingWithPaths:
    assertThat(PathFragment.filterPathsStartingWith(toPathsSet("a/b/c", "a/b/d", "a/c/d"),
        PathFragment.create("a/b"))).containsExactlyElementsIn(toPathsSet("a/b/c", "a/b/d"));
  }

  @Test
  public void testCheckAllPathsStartWithButAreNotEqualTo() {
    // Check passes:
    PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a/c"),
        PathFragment.create("a"));

    // Check trivially passes:
    PathFragment.checkAllPathsAreUnder(ImmutableList.<PathFragment>of(),
        PathFragment.create("a"));

    // Check fails when some path does not start with startingWithPath:
    try {
      PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "b/c"),
          PathFragment.create("a"));
      fail();
    } catch (IllegalArgumentException expected) {
    }

    // Check fails when some path is equal to startingWithPath:
    try {
      PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a"),
          PathFragment.create("a"));
      fail();
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test
  public void testEndsWith() {
    PathFragment foobar = PathFragment.create("/foo/bar");
    PathFragment foobarRelative = PathFragment.create("foo/bar");

    // (path, suffix) => true
    assertTrue(foobar.endsWith(foobar));
    assertTrue(foobar.endsWith(PathFragment.create("bar")));
    assertTrue(foobar.endsWith(PathFragment.create("foo/bar")));
    assertTrue(foobar.endsWith(PathFragment.create("/foo/bar")));
    assertFalse(foobar.endsWith(PathFragment.create("/bar")));

    // (prefix, path) => false
    assertFalse(PathFragment.create("/foo").endsWith(foobar));
    assertFalse(PathFragment.create("/").endsWith(foobar));

    // (suffix, path) => false
    assertFalse(PathFragment.create("/bar").endsWith(foobar));
    assertFalse(PathFragment.create("bar").endsWith(foobar));
    assertFalse(PathFragment.create("").endsWith(foobar));

    // (absolute, relative) => true
    assertTrue(foobar.endsWith(foobarRelative));

    // (relative, absolute) => false
    assertFalse(foobarRelative.endsWith(foobar));

    // (relative path, relative prefix) => true
    assertTrue(foobarRelative.endsWith(foobarRelative));
    assertTrue(foobarRelative.endsWith(PathFragment.create("bar")));
    assertTrue(foobarRelative.endsWith(PathFragment.create("")));

    // (path, sibling) => false
    assertFalse(PathFragment.create("/foo/wiz").endsWith(foobar));
    assertFalse(foobar.endsWith(PathFragment.create("/foo/wiz")));
  }

  static List<PathFragment> toPaths(List<String> strs) {
    List<PathFragment> paths = Lists.newArrayList();
    for (String s : strs) {
      paths.add(PathFragment.create(s));
    }
    return paths;
  }

  static ImmutableSet<PathFragment> toPathsSet(String... strs) {
    ImmutableSet.Builder<PathFragment> builder = ImmutableSet.builder();
    for (String str : strs) {
      builder.add(PathFragment.create(str));
    }
    return builder.build();
  }

  @Test
  public void testCompareTo() throws Exception {
    List<String> pathStrs = ImmutableList.of(
        "", "/", "//", ".", "/./", "foo/.//bar", "foo", "/foo", "foo/bar", "foo/Bar", "Foo/bar");
    List<PathFragment> paths = toPaths(pathStrs);
    // First test that compareTo is self-consistent.
    for (PathFragment x : paths) {
      for (PathFragment y : paths) {
        for (PathFragment z : paths) {
          // Anti-symmetry
          assertEquals(Integer.signum(x.compareTo(y)),
                       -1 * Integer.signum(y.compareTo(x)));
          // Transitivity
          if (x.compareTo(y) > 0 && y.compareTo(z) > 0) {
            assertThat(x.compareTo(z)).isGreaterThan(0);
          }
          // "Substitutability"
          if (x.compareTo(y) == 0) {
            assertEquals(Integer.signum(x.compareTo(z)), Integer.signum(y.compareTo(z)));
          }
          // Consistency with equals
          assertEquals((x.compareTo(y) == 0), x.equals(y));
        }
      }
    }
    // Now test that compareTo does what we expect.  The exact ordering here doesn't matter much,
    // but there are three things to notice: 1. absolute < relative, 2. comparison is lexicographic
    // 3. repeated slashes are ignored. (PathFragment("//") prints as "/").
    Collections.shuffle(paths);
    Collections.sort(paths);
    List<PathFragment> expectedOrder = toPaths(ImmutableList.of(
        "/", "//", "/./", "/foo", "", ".", "Foo/bar", "foo", "foo/.//bar", "foo/Bar", "foo/bar"));
    assertEquals(expectedOrder, paths);
  }

  @Test
  public void testGetSafePathString() {
    assertEquals("/", PathFragment.create("/").getSafePathString());
    assertEquals("/abc", PathFragment.create("/abc").getSafePathString());
    assertEquals(".", PathFragment.create("").getSafePathString());
    assertEquals(".", PathFragment.EMPTY_FRAGMENT.getSafePathString());
    assertEquals("abc/def", PathFragment.create("abc/def").getSafePathString());
  }

  @Test
  public void testNormalize() {
    assertEquals(PathFragment.create("/a/b"), PathFragment.create("/a/b").normalize());
    assertEquals(PathFragment.create("/a/b"), PathFragment.create("/a/./b").normalize());
    assertEquals(PathFragment.create("/b"), PathFragment.create("/a/../b").normalize());
    assertEquals(PathFragment.create("a/b"), PathFragment.create("a/b").normalize());
    assertEquals(PathFragment.create("../b"), PathFragment.create("a/../../b").normalize());
    assertEquals(PathFragment.create(".."), PathFragment.create("a/../..").normalize());
    assertEquals(PathFragment.create("b"), PathFragment.create("a/../b").normalize());
    assertEquals(PathFragment.create("a/b"), PathFragment.create("a/b/../b").normalize());
    assertEquals(PathFragment.create("/.."), PathFragment.create("/..").normalize());
  }

  @Test
  public void testSerializationSimple() throws Exception {
   checkSerialization("a", 91);
  }

  @Test
  public void testSerializationAbsolute() throws Exception {
    checkSerialization("/foo", 94);
   }

  @Test
  public void testSerializationNested() throws Exception {
    checkSerialization("foo/bar/baz", 101);
  }

  private void checkSerialization(String pathFragmentString, int expectedSize) throws Exception {
    PathFragment a = PathFragment.create(pathFragmentString);
    byte[] sa = TestUtils.serializeObject(a);
    assertEquals(expectedSize, sa.length);

    PathFragment a2 = (PathFragment) TestUtils.deserializeObject(sa);
    assertEquals(a, a2);
  }
}
