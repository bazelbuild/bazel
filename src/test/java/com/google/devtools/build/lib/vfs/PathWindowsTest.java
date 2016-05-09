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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for windows aspects of {@link Path}.
 */
@RunWith(JUnit4.class)
public class PathWindowsTest {
  private FileSystem filesystem;
  private Path root;

  @Before
  public final void initializeFileSystem() throws Exception  {
    filesystem = new InMemoryFileSystem(BlazeClock.instance());
    root = filesystem.getRootDirectory();
    Path first = root.getChild("first");
    first.createDirectory();
  }

  private void assertAsFragmentWorks(String expected) {
    assertEquals(new PathFragment(expected), filesystem.getPath(expected).asFragment());
  }

  @Test
  public void testWindowsPath() {
    Path p = filesystem.getPath("C:/foo/bar");
    assertEquals("C:/foo/bar", p.getPathString());
    assertEquals("C:/foo/bar", p.toString());
  }

  @Test
  public void testAsFragmentWindows() {
    assertAsFragmentWorks("C:/");
    assertAsFragmentWorks("C://");
    assertAsFragmentWorks("C:/first");
    assertAsFragmentWorks("C:/first/x/y");
    assertAsFragmentWorks("C:/first/x/y.foo");
  }

  @Test
  public void testGetRelativeWithFragmentWindows() {
    Path dir = filesystem.getPath("C:/first/x");
    assertEquals("C:/first/x/y",
                 dir.getRelative(new PathFragment("y")).toString());
    assertEquals("C:/first/x/x",
                 dir.getRelative(new PathFragment("./x")).toString());
    assertEquals("C:/first/y",
                 dir.getRelative(new PathFragment("../y")).toString());
    assertEquals("C:/first/y",
        dir.getRelative(new PathFragment("../y")).toString());
    assertEquals("C:/y",
        dir.getRelative(new PathFragment("../../../y")).toString());
  }

  @Test
  public void testGetRelativeWithAbsoluteFragmentWindows() {
    Path root = filesystem.getPath("C:/first/x");
    assertEquals("C:/x/y",
                 root.getRelative(new PathFragment("C:/x/y")).toString());
  }

  @Test
  public void testGetRelativeWithAbsoluteStringWorksWindows() {
    Path root = filesystem.getPath("C:/first/x");
    assertEquals("C:/x/y", root.getRelative("C:/x/y").toString());
  }

  @Test
  public void testParentOfRootIsRootWindows() {
    assertSame(root, root.getRelative(".."));

    assertSame(root.getRelative("dots"),
               root.getRelative("broken/../../dots"));
  }

  @Test
  public void testStartsWithWorksOnWindows() {
    assertStartsWithReturnsOnWindows(true, "C:/first/x", "C:/first/x/y");
    assertStartsWithReturnsOnWindows(true, "c:/first/x", "C:/FIRST/X/Y");
    assertStartsWithReturnsOnWindows(true, "C:/FIRST/X", "c:/first/x/y");
  }

  private void assertStartsWithReturnsOnWindows(boolean expected,
      String ancestor,
      String descendant) {
    FileSystem windowsFileSystem = new WindowsFileSystem();
    Path parent = windowsFileSystem.getPath(ancestor);
    Path child = windowsFileSystem.getPath(descendant);
    assertEquals(expected, child.startsWith(parent));
  }
}
