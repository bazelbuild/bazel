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
import static org.junit.Assert.assertSame;

import com.google.common.base.Predicate;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import com.google.devtools.build.lib.vfs.WindowsFileSystem.WindowsPathFactory;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.ArrayList;
import java.util.List;
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
    filesystem =
        new InMemoryFileSystem(BlazeClock.instance()) {
          @Override
          protected PathFactory getPathFactory() {
            return WindowsPathFactory.INSTANCE;
          }
        };
    root = filesystem.getRootDirectory().getRelative("C:/");
    root.createDirectory();

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
  public void testAbsoluteUnixPathIsRelativeToWindowsUnixRoot() {
    Path actual = root.getRelative("/foo/bar");
    Path expected = root.getRelative("C:/fake/msys/foo/bar");
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  public void testAbsoluteUnixPathReferringToDriveIsRecognized() {
    Path actual = root.getRelative("/c/foo");
    Path expected = root.getRelative("C:/foo");
    Path weird = root.getRelative("/c:");
    assertThat(actual.getPathString()).isEqualTo(expected.getPathString());
    assertThat(actual).isEqualTo(expected);
    assertThat(weird).isNotEqualTo(expected);
  }

  @Test
  public void testStartsWithWorksOnWindows() {
    assertStartsWithReturnsOnWindows(true, "C:/first/x", "C:/first/x/y");
    assertStartsWithReturnsOnWindows(true, "c:/first/x", "C:/FIRST/X/Y");
    assertStartsWithReturnsOnWindows(true, "C:/FIRST/X", "c:/first/x/y");
    assertStartsWithReturnsOnWindows(true, "/", "C:/");
    assertStartsWithReturnsOnWindows(false, "C:/", "/");
    assertStartsWithReturnsOnWindows(false, "C:/", "D:/");
    assertStartsWithReturnsOnWindows(false, "C:/", "D:/foo");
  }

  @Test
  public void testGetRelative() {
    Path root = filesystem.getPath("C:\\first\\x");
    Path other = root.getRelative("a\\b\\c");
    assertThat(other.asFragment().getPathString()).isEqualTo("C:/first/x/a/b/c");
  }

  private void assertStartsWithReturnsOnWindows(boolean expected,
      String ancestor,
      String descendant) {
    FileSystem windowsFileSystem = new WindowsFileSystem();
    Path parent = windowsFileSystem.getPath(ancestor);
    Path child = windowsFileSystem.getPath(descendant);
    assertEquals(expected, child.startsWith(parent));
  }

  @Test
  public void testChildRegistrationWithTranslatedPaths() {
    // Ensure the Path to "/usr" (actually "C:/fake/msys/usr") is created, path parents/children
    // properly registered.
    Path usrPath = root.getRelative("/usr");

    // Assert that "usr" is not registered as a child of "/".
    final List<String> children = new ArrayList<>(2);
    root.applyToChildren(
        new Predicate<Path>() {
          @Override
          public boolean apply(Path input) {
            children.add(input.getPathString());
            return true;
          }
        });
    assertThat(children).containsExactly("C:/fake", "C:/first");

    // Assert that "usr" is registered as a child of "C:/fake/msys/".
    children.clear();
    root.getRelative("C:/fake/msys")
        .applyToChildren(
            new Predicate<Path>() {
              @Override
              public boolean apply(Path input) {
                children.add(input.getPathString());
                return true;
              }
            });
    assertThat(children).containsExactly("C:/fake/msys/usr");

    assertThat(usrPath).isEqualTo(root.getRelative("C:/fake/msys/usr"));
  }
}
