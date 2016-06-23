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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.util.Collections;
import java.util.List;

/**
 * This class tests the functionality of the PathFragment.
 */
@RunWith(JUnit4.class)
public class PathFragmentTest {
  @Test
  public void testMergeFourPathsWithAbsolute() {
    assertEquals(new PathFragment("x/y/z/a/b/c/d/e"),
        new PathFragment(new PathFragment("x/y"), new PathFragment("z/a"),
            new PathFragment("/b/c"), // absolute!
            new PathFragment("d/e")));
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
        .addEqualityGroup(new PathFragment("../relative/path"),
                          new PathFragment("../relative/path"),
                          new PathFragment(new File("../relative/path")))
        .addEqualityGroup(new PathFragment("something/else"))
        .addEqualityGroup(new PathFragment("/something/else"))
        .addEqualityGroup(new PathFragment("/"),
                          new PathFragment("//////"))
        .addEqualityGroup(new PathFragment(""))
        .addEqualityGroup(filesystem.getRootDirectory())  // A Path object.
        .testEquals();
  }

  @Test
  public void testHashCodeCache() {
    PathFragment relativePath = new PathFragment("../relative/path");
    PathFragment rootPath = new PathFragment("/");

    int oldResult = relativePath.hashCode();
    int rootResult = rootPath.hashCode();
    assertEquals(oldResult, relativePath.hashCode());
    assertEquals(rootResult, rootPath.hashCode());
  }

  private void checkRelativeTo(String path, String base) {
    PathFragment relative = new PathFragment(path).relativeTo(base);
    assertEquals(new PathFragment(path), new PathFragment(base).getRelative(relative).normalize());
  }

  @Test
  public void testRelativeTo() {
    assertPath("bar/baz", new PathFragment("foo/bar/baz").relativeTo("foo"));
    assertPath("bar/baz", new PathFragment("/foo/bar/baz").relativeTo("/foo"));
    assertPath("baz", new PathFragment("foo/bar/baz").relativeTo("foo/bar"));
    assertPath("baz", new PathFragment("/foo/bar/baz").relativeTo("/foo/bar"));
    assertPath("foo", new PathFragment("/foo").relativeTo("/"));
    assertPath("foo", new PathFragment("foo").relativeTo(""));
    assertPath("foo/bar", new PathFragment("foo/bar").relativeTo(""));

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
    assertTrue(new PathFragment("/absolute/test").isAbsolute());
    assertFalse(new PathFragment("relative/test").isAbsolute());
    assertTrue(new PathFragment(new File("/absolute/test")).isAbsolute());
    assertFalse(new PathFragment(new File("relative/test")).isAbsolute());
  }

  @Test
  public void testIsNormalized() {
    assertTrue(new PathFragment("/absolute/path").isNormalized());
    assertTrue(new PathFragment("some//path").isNormalized());
    assertFalse(new PathFragment("some/./path").isNormalized());
    assertFalse(new PathFragment("../some/path").isNormalized());
    assertFalse(new PathFragment("some/other/../path").isNormalized());
    assertTrue(new PathFragment("some/other//tricky..path..").isNormalized());
    assertTrue(new PathFragment("/some/other//tricky..path..").isNormalized());
  }

  @Test
  public void testRootNodeReturnsRootString() {
    PathFragment rootFragment = new PathFragment("/");
    assertEquals("/", rootFragment.getPathString());
  }

  @Test
  public void testGetPathFragmentDoesNotNormalize() {
    String nonCanonicalPath = "/a/weird/noncanonical/../path/.";
    assertEquals(nonCanonicalPath,
        new PathFragment(nonCanonicalPath).getPathString());
  }

  @Test
  public void testGetRelative() {
    assertEquals("a/b", new PathFragment("a").getRelative("b").getPathString());
    assertEquals("a/b/c/d", new PathFragment("a/b").getRelative("c/d").getPathString());
    assertEquals("/a/b", new PathFragment("c/d").getRelative("/a/b").getPathString());
    assertEquals("a", new PathFragment("a").getRelative("").getPathString());
    assertEquals("/", new PathFragment("/").getRelative("").getPathString());
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = new PathFragment("../some/path");
    assertEquals(new PathFragment("../some/path/hi"), pf.getChild("hi"));
  }

  @Test
  public void testGetChildRejectsInvalidBaseNames() {
    PathFragment pf = new PathFragment("../some/path");
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
    assertEquals(expected, new PathFragment(actual).getPathString()); // compare string forms
    assertEquals(new PathFragment(expected), new PathFragment(actual)); // compare fragment forms
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
    PathFragment fooBarWiz = new PathFragment("foo/bar/wiz");
    PathFragment fooBar = new PathFragment("foo/bar");
    PathFragment foo = new PathFragment("foo");
    PathFragment empty = new PathFragment("");
    assertEquals(fooBar, fooBarWiz.getParentDirectory());
    assertEquals(foo, fooBar.getParentDirectory());
    assertEquals(empty, foo.getParentDirectory());
    assertNull(empty.getParentDirectory());

    PathFragment fooBarWizAbs = new PathFragment("/foo/bar/wiz");
    PathFragment fooBarAbs = new PathFragment("/foo/bar");
    PathFragment fooAbs = new PathFragment("/foo");
    PathFragment rootAbs = new PathFragment("/");
    assertEquals(fooBarAbs, fooBarWizAbs.getParentDirectory());
    assertEquals(fooAbs, fooBarAbs.getParentDirectory());
    assertEquals(rootAbs, fooAbs.getParentDirectory());
    assertNull(rootAbs.getParentDirectory());

    // Note, this is surprising but correct behavior:
    assertEquals(fooBarAbs,
                 new PathFragment("/foo/bar/..").getParentDirectory());
  }

  @Test
  public void testSegmentsCount() {
    assertEquals(2, new PathFragment("foo/bar").segmentCount());
    assertEquals(2, new PathFragment("/foo/bar").segmentCount());
    assertEquals(2, new PathFragment("foo//bar").segmentCount());
    assertEquals(2, new PathFragment("/foo//bar").segmentCount());
    assertEquals(1, new PathFragment("foo/").segmentCount());
    assertEquals(1, new PathFragment("/foo/").segmentCount());
    assertEquals(1, new PathFragment("foo").segmentCount());
    assertEquals(1, new PathFragment("/foo").segmentCount());
    assertEquals(0, new PathFragment("/").segmentCount());
    assertEquals(0, new PathFragment("").segmentCount());
  }


  @Test
  public void testGetSegment() {
    assertEquals("foo", new PathFragment("foo/bar").getSegment(0));
    assertEquals("bar", new PathFragment("foo/bar").getSegment(1));
    assertEquals("foo", new PathFragment("/foo/bar").getSegment(0));
    assertEquals("bar", new PathFragment("/foo/bar").getSegment(1));
    assertEquals("foo", new PathFragment("foo/").getSegment(0));
    assertEquals("foo", new PathFragment("/foo/").getSegment(0));
    assertEquals("foo", new PathFragment("foo").getSegment(0));
    assertEquals("foo", new PathFragment("/foo").getSegment(0));
  }

  @Test
  public void testBasename() throws Exception {
    assertEquals("bar", new PathFragment("foo/bar").getBaseName());
    assertEquals("bar", new PathFragment("/foo/bar").getBaseName());
    assertEquals("foo", new PathFragment("foo/").getBaseName());
    assertEquals("foo", new PathFragment("/foo/").getBaseName());
    assertEquals("foo", new PathFragment("foo").getBaseName());
    assertEquals("foo", new PathFragment("/foo").getBaseName());
    assertThat(new PathFragment("/").getBaseName()).isEmpty();
    assertThat(new PathFragment("").getBaseName()).isEmpty();
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertEquals(expected, actual.getPathString());
  }

  @Test
  public void testReplaceName() throws Exception {
    assertPath("foo/baz", new PathFragment("foo/bar").replaceName("baz"));
    assertPath("/foo/baz", new PathFragment("/foo/bar").replaceName("baz"));
    assertPath("foo", new PathFragment("foo/bar").replaceName(""));
    assertPath("baz", new PathFragment("foo/").replaceName("baz"));
    assertPath("/baz", new PathFragment("/foo/").replaceName("baz"));
    assertPath("baz", new PathFragment("foo").replaceName("baz"));
    assertPath("/baz", new PathFragment("/foo").replaceName("baz"));
    assertNull(new PathFragment("/").replaceName("baz"));
    assertNull(new PathFragment("/").replaceName(""));
    assertNull(new PathFragment("").replaceName("baz"));
    assertNull(new PathFragment("").replaceName(""));

    assertPath("foo/bar/baz", new PathFragment("foo/bar").replaceName("bar/baz"));
    assertPath("foo/bar/baz", new PathFragment("foo/bar").replaceName("bar/baz/"));

    // Absolute path arguments will clobber the original path.
    assertPath("/absolute", new PathFragment("foo/bar").replaceName("/absolute"));
    assertPath("/", new PathFragment("foo/bar").replaceName("/"));
  }
  @Test
  public void testSubFragment() throws Exception {
    assertPath("/foo/bar/baz",
        new PathFragment("/foo/bar/baz").subFragment(0, 3));
    assertPath("foo/bar/baz",
        new PathFragment("foo/bar/baz").subFragment(0, 3));
    assertPath("/foo/bar",
               new PathFragment("/foo/bar/baz").subFragment(0, 2));
    assertPath("bar/baz",
               new PathFragment("/foo/bar/baz").subFragment(1, 3));
    assertPath("/foo",
               new PathFragment("/foo/bar/baz").subFragment(0, 1));
    assertPath("bar",
               new PathFragment("/foo/bar/baz").subFragment(1, 2));
    assertPath("baz", new PathFragment("/foo/bar/baz").subFragment(2, 3));
    assertPath("/", new PathFragment("/foo/bar/baz").subFragment(0, 0));
    assertPath("", new PathFragment("foo/bar/baz").subFragment(0, 0));
    assertPath("", new PathFragment("foo/bar/baz").subFragment(1, 1));
    try {
      fail("unexpectedly succeeded: " + new PathFragment("foo/bar/baz").subFragment(3, 2));
    } catch (IndexOutOfBoundsException e) { /* Expected. */ }
    try {
      fail("unexpectedly succeeded: " + new PathFragment("foo/bar/baz").subFragment(4, 4));
    } catch (IndexOutOfBoundsException e) { /* Expected. */ }
  }

  @Test
  public void testStartsWith() {
    PathFragment foobar = new PathFragment("/foo/bar");
    PathFragment foobarRelative = new PathFragment("foo/bar");

    // (path, prefix) => true
    assertTrue(foobar.startsWith(foobar));
    assertTrue(foobar.startsWith(new PathFragment("/")));
    assertTrue(foobar.startsWith(new PathFragment("/foo")));
    assertTrue(foobar.startsWith(new PathFragment("/foo/")));
    assertTrue(foobar.startsWith(new PathFragment("/foo/bar/")));  // Includes trailing slash.

    // (prefix, path) => false
    assertFalse(new PathFragment("/foo").startsWith(foobar));
    assertFalse(new PathFragment("/").startsWith(foobar));

    // (absolute, relative) => false
    assertFalse(foobar.startsWith(foobarRelative));
    assertFalse(foobarRelative.startsWith(foobar));

    // (relative path, relative prefix) => true
    assertTrue(foobarRelative.startsWith(foobarRelative));
    assertTrue(foobarRelative.startsWith(new PathFragment("foo")));
    assertTrue(foobarRelative.startsWith(new PathFragment("")));

    // (path, sibling) => false
    assertFalse(new PathFragment("/foo/wiz").startsWith(foobar));
    assertFalse(foobar.startsWith(new PathFragment("/foo/wiz")));

    // Does not normalize.
    PathFragment foodotbar = new PathFragment("foo/./bar");
    assertTrue(foodotbar.startsWith(foodotbar));
    assertTrue(foodotbar.startsWith(new PathFragment("foo/.")));
    assertTrue(foodotbar.startsWith(new PathFragment("foo/./")));
    assertTrue(foodotbar.startsWith(new PathFragment("foo/./bar")));
    assertFalse(foodotbar.startsWith(new PathFragment("foo/bar")));
  }

  @Test
  public void testFilterPathsStartingWith() {
    // Retains everything:
    ImmutableSet<PathFragment> allUnderA = toPathsSet("a/b", "a/c", "a/d");
    assertThat(PathFragment.filterPathsStartingWith(allUnderA, new PathFragment("a")))
        .containsExactlyElementsIn(allUnderA);

    // Retains some but not others:
    ImmutableSet<PathFragment> mixed = toPathsSet("a/b", "a/c", "b/c");
    assertThat(PathFragment.filterPathsStartingWith(mixed,
        new PathFragment("a"))).containsExactlyElementsIn(toPathsSet("a/b", "a/c"));

    // Retains none:
    assertThat(PathFragment.filterPathsStartingWith(allUnderA, new PathFragment("b"))).isEmpty();

    // Retains paths equal to the startingWithPath:
    assertThat(PathFragment.filterPathsStartingWith(toPathsSet("a"),
        new PathFragment("a"))).containsExactlyElementsIn(toPathsSet("a"));

    // Retains everything when startingWithPath is the empty fragment:
    assertThat(PathFragment.filterPathsStartingWith(mixed, PathFragment.EMPTY_FRAGMENT))
        .containsExactlyElementsIn(mixed);

    // Supports multi-segment startingWithPaths:
    assertThat(PathFragment.filterPathsStartingWith(toPathsSet("a/b/c", "a/b/d", "a/c/d"),
        new PathFragment("a/b"))).containsExactlyElementsIn(toPathsSet("a/b/c", "a/b/d"));
  }

  @Test
  public void testCheckAllPathsStartWithButAreNotEqualTo() {
    // Check passes:
    PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a/c"),
        new PathFragment("a"));

    // Check trivially passes:
    PathFragment.checkAllPathsAreUnder(ImmutableList.<PathFragment>of(),
        new PathFragment("a"));

    // Check fails when some path does not start with startingWithPath:
    try {
      PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "b/c"),
          new PathFragment("a"));
      fail();
    } catch (IllegalArgumentException expected) {
    }

    // Check fails when some path is equal to startingWithPath:
    try {
      PathFragment.checkAllPathsAreUnder(toPathsSet("a/b", "a"),
          new PathFragment("a"));
      fail();
    } catch (IllegalArgumentException expected) {
    }
  }

  @Test
  public void testEndsWith() {
    PathFragment foobar = new PathFragment("/foo/bar");
    PathFragment foobarRelative = new PathFragment("foo/bar");

    // (path, suffix) => true
    assertTrue(foobar.endsWith(foobar));
    assertTrue(foobar.endsWith(new PathFragment("bar")));
    assertTrue(foobar.endsWith(new PathFragment("foo/bar")));
    assertTrue(foobar.endsWith(new PathFragment("/foo/bar")));
    assertFalse(foobar.endsWith(new PathFragment("/bar")));

    // (prefix, path) => false
    assertFalse(new PathFragment("/foo").endsWith(foobar));
    assertFalse(new PathFragment("/").endsWith(foobar));

    // (suffix, path) => false
    assertFalse(new PathFragment("/bar").endsWith(foobar));
    assertFalse(new PathFragment("bar").endsWith(foobar));
    assertFalse(new PathFragment("").endsWith(foobar));

    // (absolute, relative) => true
    assertTrue(foobar.endsWith(foobarRelative));

    // (relative, absolute) => false
    assertFalse(foobarRelative.endsWith(foobar));

    // (relative path, relative prefix) => true
    assertTrue(foobarRelative.endsWith(foobarRelative));
    assertTrue(foobarRelative.endsWith(new PathFragment("bar")));
    assertTrue(foobarRelative.endsWith(new PathFragment("")));

    // (path, sibling) => false
    assertFalse(new PathFragment("/foo/wiz").endsWith(foobar));
    assertFalse(foobar.endsWith(new PathFragment("/foo/wiz")));
  }

  static List<PathFragment> toPaths(List<String> strs) {
    List<PathFragment> paths = Lists.newArrayList();
    for (String s : strs) {
      paths.add(new PathFragment(s));
    }
    return paths;
  }

  static ImmutableSet<PathFragment> toPathsSet(String... strs) {
    ImmutableSet.Builder<PathFragment> builder = ImmutableSet.builder();
    for (String str : strs) {
      builder.add(new PathFragment(str));
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
    assertEquals("/", new PathFragment("/").getSafePathString());
    assertEquals("/abc", new PathFragment("/abc").getSafePathString());
    assertEquals(".", new PathFragment("").getSafePathString());
    assertEquals(".", PathFragment.EMPTY_FRAGMENT.getSafePathString());
    assertEquals("abc/def", new PathFragment("abc/def").getSafePathString());
  }

  @Test
  public void testNormalize() {
    assertEquals(new PathFragment("/a/b"), new PathFragment("/a/b").normalize());
    assertEquals(new PathFragment("/a/b"), new PathFragment("/a/./b").normalize());
    assertEquals(new PathFragment("/b"), new PathFragment("/a/../b").normalize());
    assertEquals(new PathFragment("a/b"), new PathFragment("a/b").normalize());
    assertEquals(new PathFragment("../b"), new PathFragment("a/../../b").normalize());
    assertEquals(new PathFragment(".."), new PathFragment("a/../..").normalize());
    assertEquals(new PathFragment("b"), new PathFragment("a/../b").normalize());
    assertEquals(new PathFragment("a/b"), new PathFragment("a/b/../b").normalize());
    assertEquals(new PathFragment("/.."), new PathFragment("/..").normalize());
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
    PathFragment a = new PathFragment(pathFragmentString);
    byte[] sa = TestUtils.serializeObject(a);
    assertEquals(expectedSize, sa.length);

    PathFragment a2 = (PathFragment) TestUtils.deserializeObject(sa);
    assertEquals(a, a2);
  }
}
