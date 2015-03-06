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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.testing.EqualsTester;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * This tests how canonical paths and non-canonical paths are equal with each
 * other, and also how paths from different filesystems behave with each other.
 */
@RunWith(JUnit4.class)
public class UnixPathEqualityTest {

  private FileSystem otherUnixFs;
  private FileSystem unixFs;

  @Before
  public void setUp() throws Exception {
    unixFs = new UnixFileSystem();
    otherUnixFs = new UnixFileSystem();
    assertTrue(unixFs != otherUnixFs);
  }

  private void assertTwoWayEquals(Object obj1, Object obj2) {
    assertEquals(obj2, obj1);
    new EqualsTester().addEqualityGroup(obj1, obj2).testEquals();
  }

  private void assertTwoWayNotEquals(Object obj1, Object obj2) {
    assertFalse(obj1.equals(obj2));
    assertFalse(obj2.equals(obj1));
  }

  @Test
  public void testPathsAreEqualEvenIfNotCanonical() {
    // This path is already canonical, so there's no difference between
    // the canonical / nonCanonical path, as far as equals is concerned
    Path nonCanonical = unixFs.getPath("/a/canonical/unix/path");
    Path canonical = unixFs.getPath("/a/canonical/unix/path");
    assertTwoWayEquals(nonCanonical, canonical);
  }

  @Test
  public void testPathsAreNeverEqualWithStrings() {
    // Make sure that paths aren't equal to plain old strings
    Path nonCanonical = unixFs.getPath("/a/non/../canonical/unix/path");
    Path canonical = unixFs.getPath("/a/non/../canonical/unix/path");
    assertTwoWayNotEquals(nonCanonical, "/a/non/../canonical/unix/path");
    assertTwoWayNotEquals(canonical, "/a/non/../canonical/unix/path");
  }

  @Test
  public void testCanonicalPathsFromDifferentFileSystemsAreNeverEqual() {
    Path canonical = unixFs.getPath("/canonical/path");
    Path otherCanonical = otherUnixFs.getPath("/canonical/path");
    assertTwoWayNotEquals(canonical, otherCanonical);
  }

  @Test
  public void testNonCanonicalPathsFromDifferentFileSystemsAreNeverEqual() {
    Path nonCanonical = unixFs.getPath("/non/canonical/path");
    Path otherNonCanonical = otherUnixFs.getPath("/non/canonical/path");
    assertTwoWayNotEquals(nonCanonical, otherNonCanonical);
  }

  @Test
  public void testCrossFilesystemStartsWithReturnsFalse() {
    assertFalse(unixFs.getPath("/a").startsWith(otherUnixFs.getPath("/b")));
  }

  @Test
  public void testCrossFilesystemOperationsForbidden() throws Exception {
    Path a = unixFs.getPath("/a");
    Path b = otherUnixFs.getPath("/b");

    try {
      a.renameTo(b);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("different filesystems");
    }

    try {
      a.relativeTo(b);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("different filesystems");
    }

    try {
      a.createSymbolicLink(b);
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage()).contains("different filesystems");
    }
  }
}
