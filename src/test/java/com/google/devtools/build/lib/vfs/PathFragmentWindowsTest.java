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
import static com.google.devtools.build.lib.vfs.PathFragment.create;
import static org.junit.Assert.assertThrows;

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
    assertThat(create("bar\\baz").toString()).isEqualTo("bar/baz");
    assertThat(create("c:\\bar\\baz").toString()).isEqualTo("C:/bar/baz");
  }

  @Test
  public void testIsAbsoluteWindows() {
    assertThat(create("C:/").isAbsolute()).isTrue();
    assertThat(create("C:/").isAbsolute()).isTrue();
    assertThat(create("C:/foo").isAbsolute()).isTrue();
    assertThat(create("d:/foo/bar").isAbsolute()).isTrue();

    assertThat(create("*:/").isAbsolute()).isFalse();
  }

  @Test
  public void testAbsoluteAndAbsoluteLookingPaths() {
    assertThat(create("/c").isAbsolute()).isTrue();
    assertThat(create("/c").getSegments()).containsExactly("c");

    assertThat(create("/c/").isAbsolute()).isTrue();
    assertThat(create("/c/").getSegments()).containsExactly("c");

    assertThat(create("C:/").isAbsolute()).isTrue();
    assertThat(create("C:/").getSegments()).isEmpty();

    PathFragment p5 = create("/c:");
    assertThat(p5.isAbsolute()).isTrue();
    assertThat(p5.getSegments()).containsExactly("c:");
    assertThat(create("C:").isAbsolute()).isFalse();

    assertThat(create("/c:").isAbsolute()).isTrue();
    assertThat(create("/c:").getSegments()).containsExactly("c:");

    assertThat(create("/c")).isEqualTo(create("/c/"));
    assertThat(create("/c")).isNotEqualTo(create("C:/"));
    assertThat(create("/c")).isNotEqualTo(create("C:"));
    assertThat(create("/c")).isNotEqualTo(create("/c:"));
    assertThat(create("C:/")).isNotEqualTo(create("C:"));
    assertThat(create("C:/")).isNotEqualTo(create("/c:"));
  }

  @Test
  public void testIsAbsoluteWindowsBackslash() {
    assertThat(create(new File("C:\\blah").getPath()).isAbsolute()).isTrue();
    assertThat(create(new File("C:\\").getPath()).isAbsolute()).isTrue();
    assertThat(create(new File("\\blah").getPath()).isAbsolute()).isTrue();
    assertThat(create(new File("\\").getPath()).isAbsolute()).isTrue();
  }

  @Test
  public void testRootNodeReturnsRootStringWindows() {
    assertThat(create("C:/").getPathString()).isEqualTo("C:/");
  }

  @Test
  public void testGetRelativeWindows() {
    assertThat(create("C:/a").getRelative("b").getPathString()).isEqualTo("C:/a/b");
    assertThat(create("C:/a/b").getRelative("c/d").getPathString()).isEqualTo("C:/a/b/c/d");
    assertThat(create("C:/a").getRelative("C:/b").getPathString()).isEqualTo("C:/b");
    assertThat(create("C:/a/b").getRelative("C:/c/d").getPathString()).isEqualTo("C:/c/d");
    assertThat(create("a").getRelative("C:/b").getPathString()).isEqualTo("C:/b");
    assertThat(create("a/b").getRelative("C:/c/d").getPathString()).isEqualTo("C:/c/d");
  }

  @Test
  public void testGetRelativeMixed() {
    assertThat(create("a").getRelative("b")).isEqualTo(create("a/b"));
    assertThat(create("a").getRelative("/b")).isEqualTo(create("/b"));
    assertThat(create("a").getRelative("E:/b")).isEqualTo(create("E:/b"));

    assertThat(create("/a").getRelative("b")).isEqualTo(create("/a/b"));
    assertThat(create("/a").getRelative("/b")).isEqualTo(create("/b"));
    assertThat(create("/a").getRelative("E:/b")).isEqualTo(create("E:/b"));

    assertThat(create("D:/a").getRelative("b")).isEqualTo(create("D:/a/b"));
    assertThat(create("D:/a").getRelative("/b")).isEqualTo(create("/b"));
    assertThat(create("D:/a").getRelative("E:/b")).isEqualTo(create("E:/b"));
  }

  @Test
  public void testRelativeTo() {
    assertThat(create("").relativeTo("").getPathString()).isEqualTo("");
    assertThrows(IllegalArgumentException.class, () -> create("").relativeTo("a"));

    assertThat(create("a").relativeTo("")).isEqualTo(create("a"));
    assertThat(create("a").relativeTo("a").getPathString()).isEqualTo("");
    assertThrows(IllegalArgumentException.class, () -> create("a").relativeTo("b"));
    assertThat(create("a/b").relativeTo("a")).isEqualTo(create("b"));

    assertThrows(IllegalArgumentException.class, () -> create("C:/").relativeTo(""));
    assertThat(create("C:/").relativeTo("C:/").getPathString()).isEqualTo("");
  }

  @Test
  public void testGetChildWorks() {
    assertThat(create("../some/path").getChild("hi")).isEqualTo(create("../some/path/hi"));
    assertThat(create("../some/path").getChild(".hi")).isEqualTo(create("../some/path/.hi"));
    assertThat(create("../some/path").getChild("..hi")).isEqualTo(create("../some/path/..hi"));
  }

  @Test
  public void testGetChildRejectsInvalidBaseNames() {
    assertThrows(IllegalArgumentException.class, () -> create("").getChild("."));
    assertThrows(IllegalArgumentException.class, () -> create("").getChild(".."));
    assertThrows(IllegalArgumentException.class, () -> create("").getChild("multi/segment"));
    assertThrows(IllegalArgumentException.class, () -> create("").getChild("multi\\segment"));
  }

  @Test
  public void testEmptyPathToEmptyPathWindows() {
    assertThat(create("C:/")).isEqualTo(create("C:/"));
  }

  @Test
  public void testWindowsVolumeUppercase() {
    assertThat(create("C:/")).isEqualTo(create("c:/"));
  }

  @Test
  public void testRedundantSlashesWindows() {
    assertThat(create("C:/")).isEqualTo(create("C:///"));
    assertThat(create("C:/foo/bar")).isEqualTo(create("C:/foo///bar"));
    assertThat(create("C:/foo/bar")).isEqualTo(create("C:////foo//bar"));
  }

  @Test
  public void testSimpleNameToSimpleNameWindows() {
    assertThat(create("C:/foo")).isEqualTo(create("C:/foo"));
  }

  @Test
  public void testStripsTrailingSlashWindows() {
    assertThat(create("C:/foo/bar")).isEqualTo(create("C:/foo/bar/"));
  }

  @Test
  public void testGetParentDirectoryWindows() {
    assertThat(create("C:/foo/bar/wiz").getParentDirectory()).isEqualTo(create("C:/foo/bar"));
    assertThat(create("C:/foo/bar").getParentDirectory()).isEqualTo(create("C:/foo"));
    assertThat(create("C:/foo").getParentDirectory()).isEqualTo(create("C:/"));
    assertThat(create("C:/").getParentDirectory()).isNull();
  }

  @Test
  public void testSegmentsCountWindows() {
    assertThat(create("C:/foo").segmentCount()).isEqualTo(1);
    assertThat(create("C:/").segmentCount()).isEqualTo(0);
  }

  @Test
  public void testGetSegmentWindows() {
    assertThat(create("C:/foo/bar").getSegment(0)).isEqualTo("foo");
    assertThat(create("C:/foo/bar").getSegment(1)).isEqualTo("bar");
    assertThat(create("C:/foo/").getSegment(0)).isEqualTo("foo");
    assertThat(create("C:/foo").getSegment(0)).isEqualTo("foo");
  }

  @Test
  public void testBasenameWindows() throws Exception {
    assertThat(create("C:/foo/bar").getBaseName()).isEqualTo("bar");
    assertThat(create("C:/foo").getBaseName()).isEqualTo("foo");
    // Never return the drive name as a basename.
    assertThat(create("C:/").getBaseName()).isEmpty();
  }

  @Test
  public void testReplaceNameWindows() throws Exception {
    assertThat(create("C:/foo/bar").replaceName("baz").getPathString()).isEqualTo("C:/foo/baz");
    assertThat(create("C:/").replaceName("baz")).isNull();
  }

  @Test
  public void testStartsWithWindows() {
    assertThat(create("C:/foo/bar").startsWith(create("C:/foo"))).isTrue();
    assertThat(create("C:/foo/bar").startsWith(create("C:/"))).isTrue();
    assertThat(create("C:/").startsWith(create("C:/"))).isTrue();

    // The first path is absolute, the second is not.
    assertThat(create("C:/foo/bar").startsWith(create("C:"))).isFalse();
    assertThat(create("C:/").startsWith(create("C:"))).isFalse();
  }

  @Test
  public void testEndsWithWindows() {
    assertThat(create("C:/foo/bar").endsWith(create("bar"))).isTrue();
    assertThat(create("C:/foo/bar").endsWith(create("foo/bar"))).isTrue();
    assertThat(create("C:/foo/bar").endsWith(create("C:/foo/bar"))).isTrue();
    assertThat(create("C:/").endsWith(create("C:/"))).isTrue();
  }

  @Test
  public void testGetSafePathStringWindows() {
    assertThat(create("C:/").getSafePathString()).isEqualTo("C:/");
    assertThat(create("C:/abc").getSafePathString()).isEqualTo("C:/abc");
    assertThat(create("C:/abc/def").getSafePathString()).isEqualTo("C:/abc/def");
  }

  @Test
  public void testNormalizeWindows() {
    assertThat(create("C:/a/b")).isEqualTo(create("C:/a/b"));
    assertThat(create("C:/a/./b")).isEqualTo(create("C:/a/b"));
    assertThat(create("C:/a/../b")).isEqualTo(create("C:/b"));
    assertThat(create("C:/../b")).isEqualTo(create("C:/../b"));
  }

  @Test
  public void testWindowsDriveRelativePaths() {
    // On Windows, paths that look like "C:foo" mean "foo relative to the current directory
    // of drive C:\".
    // Bazel doesn't resolve such paths, and just takes them literally like normal path segments.
    // If the user attempts to open files under such paths, the file system API will give an error.
    assertThat(create("C:").isAbsolute()).isFalse();
    assertThat(create("C:").getSegments()).containsExactly("C:");
  }

  @Test
  public void testToRelative() {
    assertThat(create("C:/foo/bar").toRelative()).isEqualTo(create("foo/bar"));
    assertThat(create("C:/").toRelative()).isEqualTo(create(""));
    assertThrows(IllegalArgumentException.class, () -> create("foo").toRelative());
  }

  @Test
  public void testGetDriveStr() {
    assertThat(create("C:/foo/bar").getDriveStr()).isEqualTo("C:/");
    assertThat(create("C:/").getDriveStr()).isEqualTo("C:/");
    assertThrows(IllegalArgumentException.class, () -> create("foo").getDriveStr());
  }
}
