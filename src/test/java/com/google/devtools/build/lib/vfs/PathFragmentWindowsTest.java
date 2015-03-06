// Copyright 2014 Google Inc. All rights reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;

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

  @Test
  public void testGetRelativeMixed() {
    assertEquals("/b", new PathFragment("C:/a").getRelative("/b").getPathString());
    assertEquals("C:/b", new PathFragment("/a").getRelative("C:/b").getPathString());
  }

  @Test
  public void testGetChildWorks() {
    PathFragment pf = new PathFragment("../some/path");
    assertEquals(new PathFragment("../some/path/hi"), pf.getChild("hi"));
  }

  // Tests after here test the canonicalization
  private void assertRegular(String expected, String actual) {
    assertEquals(expected, new PathFragment(actual).getPathString()); // compare string forms
    assertEquals(new PathFragment(expected), new PathFragment(actual)); // compare fragment forms
  }

  @Test
  public void testEmptyPathToEmptyPathWindows() {
    assertRegular("C:/", "C:/");
  }

  @Test
  public void testEmptyRelativePathToEmptyPathWindows() {
    assertRegular("C:", "C:");
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
