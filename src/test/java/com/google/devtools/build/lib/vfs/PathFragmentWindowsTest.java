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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This class tests the functionality of the PathFragment.
 */
@RunWith(JUnit4.class)
public class PathFragmentWindowsTest {

  @Test
  public void testWindowsSeparator() {
    assertEquals("bar/baz", PathFragment.create("bar\\baz").toString());
    assertEquals("C:/bar/baz", PathFragment.create("c:\\bar\\baz").toString());
  }

  @Test
  public void testIsAbsoluteWindows() {
    assertTrue(PathFragment.create("C:/").isAbsolute());
    assertTrue(PathFragment.create("C:/").isAbsolute());
    assertTrue(PathFragment.create("C:/foo").isAbsolute());
    assertTrue(PathFragment.create("d:/foo/bar").isAbsolute());

    assertFalse(PathFragment.create("*:/").isAbsolute());

    // C: is not an absolute path, it points to the current active directory on drive C:.
    assertFalse(PathFragment.create("C:").isAbsolute());
    assertFalse(PathFragment.create("C:foo").isAbsolute());
  }

  @Test
  public void testAbsoluteAndAbsoluteLookingPaths() {
    PathFragment p1 = PathFragment.create("/c");
    assertThat(p1.isAbsolute()).isTrue();
    assertThat(p1.getDriveLetter()).isEqualTo('\0');
    assertThat(p1.getSegments()).containsExactly("c");

    PathFragment p2 = PathFragment.create("/c/");
    assertThat(p2.isAbsolute()).isTrue();
    assertThat(p2.getDriveLetter()).isEqualTo('\0');
    assertThat(p2.getSegments()).containsExactly("c");

    PathFragment p3 = PathFragment.create("C:/");
    assertThat(p3.isAbsolute()).isTrue();
    assertThat(p3.getDriveLetter()).isEqualTo('C');
    assertThat(p3.getSegments()).isEmpty();

    PathFragment p4 = PathFragment.create("C:");
    assertThat(p4.isAbsolute()).isFalse();
    assertThat(p4.getDriveLetter()).isEqualTo('C');
    assertThat(p4.getSegments()).isEmpty();

    PathFragment p5 = PathFragment.create("/c:");
    assertThat(p5.isAbsolute()).isTrue();
    assertThat(p5.getDriveLetter()).isEqualTo('\0');
    assertThat(p5.getSegments()).containsExactly("c:");

    assertThat(p1).isEqualTo(p2);
    assertThat(p1).isNotEqualTo(p3);
    assertThat(p1).isNotEqualTo(p4);
    assertThat(p1).isNotEqualTo(p5);
    assertThat(p3).isNotEqualTo(p4);
    assertThat(p3).isNotEqualTo(p5);
    assertThat(p4).isNotEqualTo(p5);
  }

  @Test
  public void testIsAbsoluteWindowsBackslash() {
    assertTrue(PathFragment.create(new File("C:\\blah")).isAbsolute());
    assertTrue(PathFragment.create(new File("C:\\")).isAbsolute());
    assertTrue(PathFragment.create(new File("\\blah")).isAbsolute());
    assertTrue(PathFragment.create(new File("\\")).isAbsolute());
  }

  @Test
  public void testIsNormalizedWindows() {
    assertTrue(PathFragment.create("C:/").isNormalized());
    assertTrue(PathFragment.create("C:/absolute/path").isNormalized());
    assertFalse(PathFragment.create("C:/absolute/./path").isNormalized());
    assertFalse(PathFragment.create("C:/absolute/../path").isNormalized());
  }

  @Test
  public void testRootNodeReturnsRootStringWindows() {
    PathFragment rootFragment = PathFragment.create("C:/");
    assertEquals("C:/", rootFragment.getPathString());
  }

  @Test
  public void testGetRelativeWindows() {
    assertEquals("C:/a/b", PathFragment.create("C:/a").getRelative("b").getPathString());
    assertEquals("C:/a/b/c/d", PathFragment.create("C:/a/b").getRelative("c/d").getPathString());
    assertEquals("C:/b", PathFragment.create("C:/a").getRelative("C:/b").getPathString());
    assertEquals("C:/c/d", PathFragment.create("C:/a/b").getRelative("C:/c/d").getPathString());
    assertEquals("C:/b", PathFragment.create("a").getRelative("C:/b").getPathString());
    assertEquals("C:/c/d", PathFragment.create("a/b").getRelative("C:/c/d").getPathString());
  }

  private void assertGetRelative(String path, String relative, PathFragment expected)
      throws Exception {
    PathFragment actual = PathFragment.create(path).getRelative(relative);
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
    assertThat(actual.getDriveLetter()).isEqualTo(expected.getDriveLetter());
    assertThat(actual.hashCode()).isEqualTo(expected.hashCode());
  }

  private void assertRelativeTo(String path, String relativeTo, String... expectedPathSegments)
      throws Exception {
    PathFragment expected = PathFragment.createAlreadyInterned('\0', false, expectedPathSegments);
    PathFragment actual = PathFragment.create(path).relativeTo(relativeTo);
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
    assertThat(actual.getDriveLetter()).isEqualTo(expected.getDriveLetter());
    assertThat(actual.hashCode()).isEqualTo(expected.hashCode());
  }

  private void assertCantComputeRelativeTo(String path, String relativeTo) throws Exception {
    try {
      PathFragment.create(path).relativeTo(relativeTo);
      Assert.fail("expected failure");
    } catch (Exception e) {
      assertThat(e.getMessage()).contains("is not beneath");
    }
  }

  private static PathFragment makePath(char drive, boolean absolute, String... segments) {
    return PathFragment.createAlreadyInterned(drive, absolute, segments);
  }

  @Test
  public void testGetRelativeMixed() throws Exception {
    assertGetRelative("a", "b", makePath('\0', false, "a", "b"));
    assertGetRelative("a", "/b", makePath('\0', true, "b"));
    assertGetRelative("a", "E:b", makePath('\0', false, "a", "b"));
    assertGetRelative("a", "E:/b", makePath('E', true, "b"));

    assertGetRelative("/a", "b", makePath('\0', true, "a", "b"));
    assertGetRelative("/a", "/b", makePath('\0', true, "b"));
    assertGetRelative("/a", "E:b", makePath('\0', true, "a", "b"));
    assertGetRelative("/a", "E:/b", makePath('E', true, "b"));

    assertGetRelative("D:a", "b", makePath('D', false, "a", "b"));
    assertGetRelative("D:a", "/b", makePath('D', true, "b"));
    assertGetRelative("D:a", "E:b", makePath('D', false, "a", "b"));
    assertGetRelative("D:a", "E:/b", makePath('E', true, "b"));

    assertGetRelative("D:/a", "b", makePath('D', true, "a", "b"));
    assertGetRelative("D:/a", "/b", makePath('D', true, "b"));
    assertGetRelative("D:/a", "E:b", makePath('D', true, "a", "b"));
    assertGetRelative("D:/a", "E:/b", makePath('E', true, "b"));
  }

  @Test
  public void testRelativeTo() throws Exception {
    assertRelativeTo("", "");
    assertCantComputeRelativeTo("", "a");

    assertRelativeTo("a", "", "a");
    assertRelativeTo("a", "a");
    assertCantComputeRelativeTo("a", "b");
    assertRelativeTo("a/b", "a", "b");

    assertRelativeTo("C:", "");
    assertRelativeTo("C:", "C:");
    assertCantComputeRelativeTo("C:/", "");
    assertRelativeTo("C:/", "C:/");
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = PathFragment.create("../some/path");
    assertEquals(PathFragment.create("../some/path/hi"), pf.getChild("hi"));
  }

  // Tests after here test the canonicalization
  private void assertRegular(String expected, String actual) {
    PathFragment exp = PathFragment.create(expected);
    PathFragment act = PathFragment.create(actual);
    assertThat(exp.getPathString()).isEqualTo(expected);
    assertThat(act.getPathString()).isEqualTo(expected);
    assertThat(act).isEqualTo(exp);
    assertThat(act.hashCode()).isEqualTo(exp.hashCode());
  }

  @Test
  public void testEmptyPathToEmptyPathWindows() {
    assertRegular("C:/", "C:/");
  }

  private void assertAllEqual(PathFragment... ps) {
    assertThat(ps.length).isGreaterThan(1);
    for (int i = 1; i < ps.length; i++) {
      String msg = "comparing items 0 and " + i;
      assertWithMessage(msg + " for getPathString")
          .that(ps[i].getPathString())
          .isEqualTo(ps[0].getPathString());
      assertWithMessage(msg + " for equals").that(ps[0]).isEqualTo(ps[i]);
      assertWithMessage(msg + " for hashCode").that(ps[0].hashCode()).isEqualTo(ps[i].hashCode());
    }
  }

  @Test
  public void testEmptyRelativePathToEmptyPathWindows() {
    // Surprising but correct behavior: a PathFragment made of just a drive identifier (and not the
    // absolute path "C:/") is equal not only to the empty fragment, but (therefore) also to other
    // drive identifiers.
    // This makes sense if you consider that these are still empty paths, the drive letter adds no
    // information to the path itself.
    assertAllEqual(
        PathFragment.EMPTY_FRAGMENT,
        PathFragment.create("C:"),
        PathFragment.create("D:"),
        PathFragment.createAlreadyInterned('\0', false, new String[0]),
        PathFragment.createAlreadyInterned('C', false, new String[0]),
        PathFragment.createAlreadyInterned('D', false, new String[0]));
    assertAllEqual(PathFragment.create("/c"), PathFragment.create("/c/"));
    assertThat(PathFragment.create("C:/")).isNotEqualTo(PathFragment.create("/c"));
    assertThat(PathFragment.create("C:/foo")).isNotEqualTo(PathFragment.create("/c/foo"));

    assertThat(PathFragment.create("C:/")).isNotEqualTo(PathFragment.create("C:"));
    assertThat(PathFragment.create("C:/").getPathString())
        .isNotEqualTo(PathFragment.create("C:").getPathString());
  }

  @Test
  public void testConfusingSemanticsOfDriveLettersInRelativePaths() {
    // This test serves to document the current confusing semantics of non-empty relative windows
    // paths that have drive letters. Also note the above testEmptyRelativePathToEmptyPathWindows
    // which documents the confusing semantics of empty relative windows paths that have drive
    // letters.
    //
    // TODO(laszlocsomor): Reevaluate the current semantics. Depending on the results of that,
    // consider not storing the drive letter in relative windows paths.
    PathFragment cColonFoo = PathFragment.create("C:foo");
    PathFragment dColonFoo = PathFragment.create("D:foo");
    PathFragment foo = PathFragment.create("foo");
    assertThat(cColonFoo).isNotEqualTo(dColonFoo);
    assertThat(cColonFoo).isNotEqualTo(foo);
    assertThat(dColonFoo).isNotEqualTo(foo);
    assertThat(cColonFoo.segmentCount()).isEqualTo(dColonFoo.segmentCount());
    assertThat(cColonFoo.segmentCount()).isEqualTo(foo.segmentCount());
    assertThat(cColonFoo.startsWith(dColonFoo)).isTrue();
    assertThat(cColonFoo.startsWith(foo)).isTrue();
    assertThat(foo.startsWith(cColonFoo)).isTrue();
    assertThat(cColonFoo.getPathString()).isEqualTo("foo");
    assertThat(cColonFoo.getPathString()).isEqualTo(dColonFoo.getPathString());
    assertThat(cColonFoo.getPathString()).isEqualTo(foo.getPathString());
  }

  @Test
  public void testWindowsVolumeUppercase() {
    assertRegular("C:/", "c:/");
  }

  @Test
  public void testRedundantSlashesWindows() {
    assertRegular("C:/", "C:///");
    assertRegular("C:/foo/bar", "C:/foo///bar");
    assertRegular("C:/foo/bar", "C:////foo//bar");
  }

  @Test
  public void testSimpleNameToSimpleNameWindows() {
    assertRegular("C:/foo", "C:/foo");
  }

  @Test
  public void testStripsTrailingSlashWindows() {
    assertRegular("C:/foo/bar", "C:/foo/bar/");
  }

  @Test
  public void testGetParentDirectoryWindows() {
    PathFragment fooBarWizAbs = PathFragment.create("C:/foo/bar/wiz");
    PathFragment fooBarAbs = PathFragment.create("C:/foo/bar");
    PathFragment fooAbs = PathFragment.create("C:/foo");
    PathFragment rootAbs = PathFragment.create("C:/");
    assertEquals(fooBarAbs, fooBarWizAbs.getParentDirectory());
    assertEquals(fooAbs, fooBarAbs.getParentDirectory());
    assertEquals(rootAbs, fooAbs.getParentDirectory());
    assertNull(rootAbs.getParentDirectory());

    // Note, this is suprising but correct behaviour:
    assertEquals(fooBarAbs,
                 PathFragment.create("C:/foo/bar/..").getParentDirectory());
  }

  @Test
  public void testSegmentsCountWindows() {
    assertEquals(1, PathFragment.create("C:/foo").segmentCount());
    assertEquals(0, PathFragment.create("C:/").segmentCount());
  }

  @Test
  public void testGetSegmentWindows() {
    assertEquals("foo", PathFragment.create("C:/foo/bar").getSegment(0));
    assertEquals("bar", PathFragment.create("C:/foo/bar").getSegment(1));
    assertEquals("foo", PathFragment.create("C:/foo/").getSegment(0));
    assertEquals("foo", PathFragment.create("C:/foo").getSegment(0));
  }

  @Test
  public void testBasenameWindows() throws Exception {
    assertEquals("bar", PathFragment.create("C:/foo/bar").getBaseName());
    assertEquals("foo", PathFragment.create("C:/foo").getBaseName());
    // Never return the drive name as a basename.
    assertThat(PathFragment.create("C:/").getBaseName()).isEmpty();
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertEquals(expected, actual.getPathString());
  }

  @Test
  public void testReplaceNameWindows() throws Exception {
    assertPath("C:/foo/baz", PathFragment.create("C:/foo/bar").replaceName("baz"));
    assertNull(PathFragment.create("C:/").replaceName("baz"));
  }

  @Test
  public void testStartsWithWindows() {
    assertTrue(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:/foo")));
    assertTrue(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:/")));
    assertTrue(PathFragment.create("C:foo/bar").startsWith(PathFragment.create("C:")));
    assertTrue(PathFragment.create("C:/").startsWith(PathFragment.create("C:/")));
    assertTrue(PathFragment.create("C:").startsWith(PathFragment.create("C:")));

    // The first path is absolute, the second is not.
    assertFalse(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:")));
    assertFalse(PathFragment.create("C:/").startsWith(PathFragment.create("C:")));
  }

  @Test
  public void testEndsWithWindows() {
    assertTrue(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("bar")));
    assertTrue(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("foo/bar")));
    assertTrue(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("C:/foo/bar")));
    assertTrue(PathFragment.create("C:/").endsWith(PathFragment.create("C:/")));
  }

  @Test
  public void testGetSafePathStringWindows() {
    assertEquals("C:/", PathFragment.create("C:/").getSafePathString());
    assertEquals("C:/abc", PathFragment.create("C:/abc").getSafePathString());
    assertEquals("C:/abc/def", PathFragment.create("C:/abc/def").getSafePathString());
  }

  @Test
  public void testNormalizeWindows() {
    assertEquals(PathFragment.create("C:/a/b"), PathFragment.create("C:/a/b").normalize());
    assertEquals(PathFragment.create("C:/a/b"), PathFragment.create("C:/a/./b").normalize());
    assertEquals(PathFragment.create("C:/b"), PathFragment.create("C:/a/../b").normalize());
    assertEquals(PathFragment.create("C:/../b"), PathFragment.create("C:/../b").normalize());
  }
}
