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
import static org.junit.Assert.fail;

import java.io.File;
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
    assertThat(PathFragment.create("bar\\baz").toString()).isEqualTo("bar/baz");
    assertThat(PathFragment.create("c:\\bar\\baz").toString()).isEqualTo("C:/bar/baz");
  }

  @Test
  public void testIsAbsoluteWindows() {
    assertThat(PathFragment.create("C:/").isAbsolute()).isTrue();
    assertThat(PathFragment.create("C:/").isAbsolute()).isTrue();
    assertThat(PathFragment.create("C:/foo").isAbsolute()).isTrue();
    assertThat(PathFragment.create("d:/foo/bar").isAbsolute()).isTrue();

    assertThat(PathFragment.create("*:/").isAbsolute()).isFalse();
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

    PathFragment p5 = PathFragment.create("/c:");
    assertThat(p5.isAbsolute()).isTrue();
    assertThat(p5.getDriveLetter()).isEqualTo('\0');
    assertThat(p5.getSegments()).containsExactly("c:");

    assertThat(p1).isEqualTo(p2);
    assertThat(p1).isNotEqualTo(p3);
    assertThat(p1).isNotEqualTo(p5);
    assertThat(p3).isNotEqualTo(p5);
  }

  @Test
  public void testIsAbsoluteWindowsBackslash() {
    assertThat(PathFragment.create(new File("C:\\blah")).isAbsolute()).isTrue();
    assertThat(PathFragment.create(new File("C:\\")).isAbsolute()).isTrue();
    assertThat(PathFragment.create(new File("\\blah")).isAbsolute()).isTrue();
    assertThat(PathFragment.create(new File("\\")).isAbsolute()).isTrue();
  }

  @Test
  public void testIsNormalizedWindows() {
    assertThat(PathFragment.create("C:/").isNormalized()).isTrue();
    assertThat(PathFragment.create("C:/absolute/path").isNormalized()).isTrue();
    assertThat(PathFragment.create("C:/absolute/./path").isNormalized()).isFalse();
    assertThat(PathFragment.create("C:/absolute/../path").isNormalized()).isFalse();
  }

  @Test
  public void testRootNodeReturnsRootStringWindows() {
    PathFragment rootFragment = PathFragment.create("C:/");
    assertThat(rootFragment.getPathString()).isEqualTo("C:/");
  }

  @Test
  public void testGetRelativeWindows() {
    assertThat(PathFragment.create("C:/a").getRelative("b").getPathString()).isEqualTo("C:/a/b");
    assertThat(PathFragment.create("C:/a/b").getRelative("c/d").getPathString())
        .isEqualTo("C:/a/b/c/d");
    assertThat(PathFragment.create("C:/a").getRelative("C:/b").getPathString()).isEqualTo("C:/b");
    assertThat(PathFragment.create("C:/a/b").getRelative("C:/c/d").getPathString())
        .isEqualTo("C:/c/d");
    assertThat(PathFragment.create("a").getRelative("C:/b").getPathString()).isEqualTo("C:/b");
    assertThat(PathFragment.create("a/b").getRelative("C:/c/d").getPathString())
        .isEqualTo("C:/c/d");
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
      fail("expected failure");
    } catch (Exception e) {
      assertThat(e).hasMessageThat().contains("is not beneath");
    }
  }

  private static PathFragment makePath(char drive, boolean absolute, String... segments) {
    return PathFragment.createAlreadyInterned(drive, absolute, segments);
  }

  @Test
  public void testGetRelativeMixed() throws Exception {
    assertGetRelative("a", "b", makePath('\0', false, "a", "b"));
    assertGetRelative("a", "/b", makePath('\0', true, "b"));
    assertGetRelative("a", "E:/b", makePath('E', true, "b"));

    assertGetRelative("/a", "b", makePath('\0', true, "a", "b"));
    assertGetRelative("/a", "/b", makePath('\0', true, "b"));
    assertGetRelative("/a", "E:/b", makePath('E', true, "b"));

    assertGetRelative("D:/a", "b", makePath('D', true, "a", "b"));
    assertGetRelative("D:/a", "/b", makePath('D', true, "b"));
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

    assertCantComputeRelativeTo("C:/", "");
    assertRelativeTo("C:/", "C:/");
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = PathFragment.create("../some/path");
    assertThat(pf.getChild("hi")).isEqualTo(PathFragment.create("../some/path/hi"));
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
    assertThat(fooBarWizAbs.getParentDirectory()).isEqualTo(fooBarAbs);
    assertThat(fooBarAbs.getParentDirectory()).isEqualTo(fooAbs);
    assertThat(fooAbs.getParentDirectory()).isEqualTo(rootAbs);
    assertThat(rootAbs.getParentDirectory()).isNull();

    // Note, this is suprising but correct behaviour:
    assertThat(PathFragment.create("C:/foo/bar/..").getParentDirectory()).isEqualTo(fooBarAbs);
  }

  @Test
  public void testSegmentsCountWindows() {
    assertThat(PathFragment.create("C:/foo").segmentCount()).isEqualTo(1);
    assertThat(PathFragment.create("C:/").segmentCount()).isEqualTo(0);
  }

  @Test
  public void testGetSegmentWindows() {
    assertThat(PathFragment.create("C:/foo/bar").getSegment(0)).isEqualTo("foo");
    assertThat(PathFragment.create("C:/foo/bar").getSegment(1)).isEqualTo("bar");
    assertThat(PathFragment.create("C:/foo/").getSegment(0)).isEqualTo("foo");
    assertThat(PathFragment.create("C:/foo").getSegment(0)).isEqualTo("foo");
  }

  @Test
  public void testBasenameWindows() throws Exception {
    assertThat(PathFragment.create("C:/foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(PathFragment.create("C:/foo").getBaseName()).isEqualTo("foo");
    // Never return the drive name as a basename.
    assertThat(PathFragment.create("C:/").getBaseName()).isEmpty();
  }

  private static void assertPath(String expected, PathFragment actual) {
    assertThat(actual.getPathString()).isEqualTo(expected);
  }

  @Test
  public void testReplaceNameWindows() throws Exception {
    assertPath("C:/foo/baz", PathFragment.create("C:/foo/bar").replaceName("baz"));
    assertThat(PathFragment.create("C:/").replaceName("baz")).isNull();
  }

  @Test
  public void testStartsWithWindows() {
    assertThat(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:/foo")))
        .isTrue();
    assertThat(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:/"))).isTrue();
    assertThat(PathFragment.create("C:/").startsWith(PathFragment.create("C:/"))).isTrue();

    // The first path is absolute, the second is not.
    assertThat(PathFragment.create("C:/foo/bar").startsWith(PathFragment.create("C:"))).isFalse();
    assertThat(PathFragment.create("C:/").startsWith(PathFragment.create("C:"))).isFalse();
  }

  @Test
  public void testEndsWithWindows() {
    assertThat(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("bar"))).isTrue();
    assertThat(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("foo/bar"))).isTrue();
    assertThat(PathFragment.create("C:/foo/bar").endsWith(PathFragment.create("C:/foo/bar")))
        .isTrue();
    assertThat(PathFragment.create("C:/").endsWith(PathFragment.create("C:/"))).isTrue();
  }

  @Test
  public void testGetSafePathStringWindows() {
    assertThat(PathFragment.create("C:/").getSafePathString()).isEqualTo("C:/");
    assertThat(PathFragment.create("C:/abc").getSafePathString()).isEqualTo("C:/abc");
    assertThat(PathFragment.create("C:/abc/def").getSafePathString()).isEqualTo("C:/abc/def");
  }

  @Test
  public void testNormalizeWindows() {
    assertThat(PathFragment.create("C:/a/b").normalize()).isEqualTo(PathFragment.create("C:/a/b"));
    assertThat(PathFragment.create("C:/a/./b").normalize())
        .isEqualTo(PathFragment.create("C:/a/b"));
    assertThat(PathFragment.create("C:/a/../b").normalize()).isEqualTo(PathFragment.create("C:/b"));
    assertThat(PathFragment.create("C:/../b").normalize())
        .isEqualTo(PathFragment.create("C:/../b"));
  }

  @Test
  public void testWindowsDriveRelativePaths() throws Exception {
    // On Windows, paths that look like "C:foo" mean "foo relative to the current directory
    // of drive C:\".
    // Bazel doesn't resolve such paths, and just takes them literally like normal path segments.
    // If the user attempts to open files under such paths, the file system API will give an error.
    assertThat(PathFragment.create("C:").isAbsolute()).isFalse();
    assertThat(PathFragment.create("C:").getDriveLetter()).isEqualTo('\0');
    assertThat(PathFragment.create("C:").getSegments()).containsExactly("C:");
  }
}
