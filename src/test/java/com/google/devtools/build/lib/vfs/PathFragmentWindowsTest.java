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
    assertEquals("bar/baz", new PathFragment("bar\\baz").toString());
    assertEquals("C:/bar/baz", new PathFragment("c:\\bar\\baz").toString());
  }

  @Test
  public void testIsAbsoluteWindows() {
    assertTrue(new PathFragment("C:/").isAbsolute());
    assertTrue(new PathFragment("C:/").isAbsolute());
    assertTrue(new PathFragment("C:/foo").isAbsolute());
    assertTrue(new PathFragment("d:/foo/bar").isAbsolute());

    assertFalse(new PathFragment("*:/").isAbsolute());

    // C: is not an absolute path, it points to the current active directory on drive C:.
    assertFalse(new PathFragment("C:").isAbsolute());
    assertFalse(new PathFragment("C:foo").isAbsolute());
  }

  @Test
  public void testAbsoluteAndAbsoluteLookingPaths() {
    PathFragment p1 = new PathFragment("/c");
    assertThat(p1.isAbsolute()).isTrue();
    assertThat(p1.getDriveLetter()).isEqualTo('\0');
    assertThat(p1.getSegments()).containsExactly("c");

    PathFragment p2 = new PathFragment("/c/");
    assertThat(p2.isAbsolute()).isTrue();
    assertThat(p2.getDriveLetter()).isEqualTo('\0');
    assertThat(p2.getSegments()).containsExactly("c");

    PathFragment p3 = new PathFragment("C:/");
    assertThat(p3.isAbsolute()).isTrue();
    assertThat(p3.getDriveLetter()).isEqualTo('C');
    assertThat(p3.getSegments()).isEmpty();

    PathFragment p4 = new PathFragment("C:");
    assertThat(p4.isAbsolute()).isFalse();
    assertThat(p4.getDriveLetter()).isEqualTo('C');
    assertThat(p4.getSegments()).isEmpty();

    PathFragment p5 = new PathFragment("/c:");
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
    assertTrue(new PathFragment(new File("C:\\blah")).isAbsolute());
    assertTrue(new PathFragment(new File("C:\\")).isAbsolute());
    assertTrue(new PathFragment(new File("\\blah")).isAbsolute());
    assertTrue(new PathFragment(new File("\\")).isAbsolute());
  }

  @Test
  public void testIsNormalizedWindows() {
    assertTrue(new PathFragment("C:/").isNormalized());
    assertTrue(new PathFragment("C:/absolute/path").isNormalized());
    assertFalse(new PathFragment("C:/absolute/./path").isNormalized());
    assertFalse(new PathFragment("C:/absolute/../path").isNormalized());
  }

  @Test
  public void testRootNodeReturnsRootStringWindows() {
    PathFragment rootFragment = new PathFragment("C:/");
    assertEquals("C:/", rootFragment.getPathString());
  }

  @Test
  public void testGetRelativeWindows() {
    assertEquals("C:/a/b", new PathFragment("C:/a").getRelative("b").getPathString());
    assertEquals("C:/a/b/c/d", new PathFragment("C:/a/b").getRelative("c/d").getPathString());
    assertEquals("C:/b", new PathFragment("C:/a").getRelative("C:/b").getPathString());
    assertEquals("C:/c/d", new PathFragment("C:/a/b").getRelative("C:/c/d").getPathString());
    assertEquals("C:/b", new PathFragment("a").getRelative("C:/b").getPathString());
    assertEquals("C:/c/d", new PathFragment("a/b").getRelative("C:/c/d").getPathString());
  }

  private void assertGetRelative(String path, String relative, PathFragment expected)
      throws Exception {
    PathFragment actual = new PathFragment(path).getRelative(relative);
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
    assertThat(actual.getDriveLetter()).isEqualTo(expected.getDriveLetter());
    assertThat(actual.hashCode()).isEqualTo(expected.hashCode());
  }

  private void assertRelativeTo(String path, String relativeTo, String... expectedPathSegments)
      throws Exception {
    PathFragment expected = new PathFragment('\0', false, expectedPathSegments);
    PathFragment actual = new PathFragment(path).relativeTo(relativeTo);
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
    assertThat(actual.getDriveLetter()).isEqualTo(expected.getDriveLetter());
    assertThat(actual.hashCode()).isEqualTo(expected.hashCode());
  }

  private void assertCantComputeRelativeTo(String path, String relativeTo) throws Exception {
    try {
      new PathFragment(path).relativeTo(relativeTo);
      Assert.fail("expected failure");
    } catch (Exception e) {
      assertThat(e.getMessage()).contains("is not beneath");
    }
  }

  private static PathFragment makePath(char drive, boolean absolute, String... segments) {
    return new PathFragment(drive, absolute, segments);
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
    PathFragment pf = new PathFragment("../some/path");
    assertEquals(new PathFragment("../some/path/hi"), pf.getChild("hi"));
  }

  // Tests after here test the canonicalization
  private void assertRegular(String expected, String actual) {
    PathFragment exp = new PathFragment(expected);
    PathFragment act = new PathFragment(actual);
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
        new PathFragment("C:"),
        new PathFragment("D:"),
        new PathFragment('\0', false, new String[0]),
        new PathFragment('C', false, new String[0]),
        new PathFragment('D', false, new String[0]));
    assertAllEqual(new PathFragment("/c"), new PathFragment("/c/"));
    assertThat(new PathFragment("C:/")).isNotEqualTo(new PathFragment("/c"));
    assertThat(new PathFragment("C:/foo")).isNotEqualTo(new PathFragment("/c/foo"));

    assertThat(new PathFragment("C:/")).isNotEqualTo(new PathFragment("C:"));
    assertThat(new PathFragment("C:/").getPathString())
        .isNotEqualTo(new PathFragment("C:").getPathString());
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
    PathFragment fooBarWizAbs = new PathFragment("C:/foo/bar/wiz");
    PathFragment fooBarAbs = new PathFragment("C:/foo/bar");
    PathFragment fooAbs = new PathFragment("C:/foo");
    PathFragment rootAbs = new PathFragment("C:/");
    assertEquals(fooBarAbs, fooBarWizAbs.getParentDirectory());
    assertEquals(fooAbs, fooBarAbs.getParentDirectory());
    assertEquals(rootAbs, fooAbs.getParentDirectory());
    assertNull(rootAbs.getParentDirectory());

    // Note, this is suprising but correct behaviour:
    assertEquals(fooBarAbs,
                 new PathFragment("C:/foo/bar/..").getParentDirectory());
  }

  @Test
  public void testSegmentsCountWindows() {
    assertEquals(1, new PathFragment("C:/foo").segmentCount());
    assertEquals(0, new PathFragment("C:/").segmentCount());
  }

  @Test
  public void testGetSegmentWindows() {
    assertEquals("foo", new PathFragment("C:/foo/bar").getSegment(0));
    assertEquals("bar", new PathFragment("C:/foo/bar").getSegment(1));
    assertEquals("foo", new PathFragment("C:/foo/").getSegment(0));
    assertEquals("foo", new PathFragment("C:/foo").getSegment(0));
  }

  @Test
  public void testBasenameWindows() throws Exception {
    assertEquals("bar", new PathFragment("C:/foo/bar").getBaseName());
    assertEquals("foo", new PathFragment("C:/foo").getBaseName());
    // Never return the drive name as a basename.
    assertThat(new PathFragment("C:/").getBaseName()).isEmpty();
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertEquals(expected, actual.getPathString());
  }

  @Test
  public void testReplaceNameWindows() throws Exception {
    assertPath("C:/foo/baz", new PathFragment("C:/foo/bar").replaceName("baz"));
    assertNull(new PathFragment("C:/").replaceName("baz"));
  }

  @Test
  public void testStartsWithWindows() {
    assertTrue(new PathFragment("C:/foo/bar").startsWith(new PathFragment("C:/foo")));
    assertTrue(new PathFragment("C:/foo/bar").startsWith(new PathFragment("C:/")));
    assertTrue(new PathFragment("C:foo/bar").startsWith(new PathFragment("C:")));
    assertTrue(new PathFragment("C:/").startsWith(new PathFragment("C:/")));
    assertTrue(new PathFragment("C:").startsWith(new PathFragment("C:")));

    // The first path is absolute, the second is not.
    assertFalse(new PathFragment("C:/foo/bar").startsWith(new PathFragment("C:")));
    assertFalse(new PathFragment("C:/").startsWith(new PathFragment("C:")));
  }

  @Test
  public void testEndsWithWindows() {
    assertTrue(new PathFragment("C:/foo/bar").endsWith(new PathFragment("bar")));
    assertTrue(new PathFragment("C:/foo/bar").endsWith(new PathFragment("foo/bar")));
    assertTrue(new PathFragment("C:/foo/bar").endsWith(new PathFragment("C:/foo/bar")));
    assertTrue(new PathFragment("C:/").endsWith(new PathFragment("C:/")));
  }

  @Test
  public void testGetSafePathStringWindows() {
    assertEquals("C:/", new PathFragment("C:/").getSafePathString());
    assertEquals("C:/abc", new PathFragment("C:/abc").getSafePathString());
    assertEquals("C:/abc/def", new PathFragment("C:/abc/def").getSafePathString());
  }

  @Test
  public void testNormalizeWindows() {
    assertEquals(new PathFragment("C:/a/b"), new PathFragment("C:/a/b").normalize());
    assertEquals(new PathFragment("C:/a/b"), new PathFragment("C:/a/./b").normalize());
    assertEquals(new PathFragment("C:/b"), new PathFragment("C:/a/../b").normalize());
    assertEquals(new PathFragment("C:/../b"), new PathFragment("C:/../b").normalize());
  }
}
