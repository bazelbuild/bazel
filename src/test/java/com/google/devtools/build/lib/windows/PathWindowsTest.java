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
package com.google.devtools.build.lib.windows;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem.WindowsPath;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for windows aspects of {@link Path}. */
@RunWith(JUnit4.class)
public class PathWindowsTest {

  private static final class MockShortPathResolver implements Function<String, String> {
    public List<String> resolutionQueries = new ArrayList<>();

    // Full path to resolved child mapping.
    public Map<String, String> resolutions = new HashMap<>();

    @Override
    public String apply(String path) {
      path = path.toLowerCase();
      resolutionQueries.add(path);
      return resolutions.get(path);
    }
  }

  private FileSystem filesystem;
  private WindowsPath root;
  private final MockShortPathResolver shortPathResolver = new MockShortPathResolver();

  @Before
  public final void initializeFileSystem() throws Exception {
    filesystem =
        new InMemoryFileSystem(BlazeClock.instance()) {
          @Override
          protected PathFactory getPathFactory() {
            return WindowsFileSystem.getPathFactoryForTesting(shortPathResolver);
          }

          @Override
          public boolean isFilePathCaseSensitive() {
            return false;
          }
        };
    root = (WindowsPath) filesystem.getRootDirectory().getRelative("C:/");
    root.createDirectory();
  }

  private void assertAsFragmentWorks(String expected) {
    assertThat(filesystem.getPath(expected).asFragment()).isEqualTo(PathFragment.create(expected));
  }

  @Test
  public void testWindowsPath() {
    Path p = filesystem.getPath("C:/foo/bar");
    assertThat(p.getPathString()).isEqualTo("C:/foo/bar");
    assertThat(p.toString()).isEqualTo("C:/foo/bar");
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
    assertThat(dir.getRelative(PathFragment.create("y")).toString()).isEqualTo("C:/first/x/y");
    assertThat(dir.getRelative(PathFragment.create("./x")).toString()).isEqualTo("C:/first/x/x");
    assertThat(dir.getRelative(PathFragment.create("../y")).toString()).isEqualTo("C:/first/y");
    assertThat(dir.getRelative(PathFragment.create("../y")).toString()).isEqualTo("C:/first/y");
    assertThat(dir.getRelative(PathFragment.create("../../../y")).toString()).isEqualTo("C:/y");
  }

  @Test
  public void testGetRelativeWithAbsoluteFragmentWindows() {
    Path x = filesystem.getPath("C:/first/x");
    assertThat(x.getRelative(PathFragment.create("C:/x/y")).toString()).isEqualTo("C:/x/y");
  }

  @Test
  public void testGetRelativeWithAbsoluteStringWorksWindows() {
    Path x = filesystem.getPath("C:/first/x");
    assertThat(x.getRelative("C:/x/y").toString()).isEqualTo("C:/x/y");
  }

  @Test
  public void testParentOfRootIsRootWindows() {
    assertThat(root).isSameAs(root.getRelative(".."));
    assertThat(root.getRelative("dots")).isSameAs(root.getRelative("broken/../../dots"));
  }

  @Test
  public void testStartsWithWorksOnWindows() {
    assertStartsWithReturnsOnWindows(true, "C:/first/x", "C:/first/x/y");
    assertStartsWithReturnsOnWindows(true, "c:/first/x", "C:/FIRST/X/Y");
    assertStartsWithReturnsOnWindows(true, "C:/FIRST/X", "c:/first/x/y");
    assertStartsWithReturnsOnWindows(false, "C:/", "/");
    assertStartsWithReturnsOnWindows(false, "C:/", "D:/");
    assertStartsWithReturnsOnWindows(false, "C:/", "D:/foo");
  }

  @Test
  public void testGetRelative() {
    Path x = filesystem.getPath("C:\\first\\x");
    Path other = x.getRelative("a\\b\\c");
    assertThat(other.asFragment().getPathString()).isEqualTo("C:/first/x/a/b/c");
  }

  private static void assertStartsWithReturnsOnWindows(
      boolean expected, String ancestor, String descendant) {
    FileSystem windowsFileSystem = new WindowsFileSystem();
    Path parent = windowsFileSystem.getPath(ancestor);
    Path child = windowsFileSystem.getPath(descendant);
    assertThat(child.startsWith(parent)).isEqualTo(expected);
  }

  @Test
  public void testResolvesShortenedPaths() {
    shortPathResolver.resolutions.put("d:/progra~1", "program files");
    shortPathResolver.resolutions.put("d:/program files/micros~1", "microsoft something");
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/~bar~1", "~bar_hello");

    // Assert normal shortpath resolution.
    Path normal = root.getRelative("d:/progra~1/micros~1/foo/~bar~1/baz");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(normal.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/~bar_hello/baz");
    // Assert that we only try to resolve the path segments that look like they may be shortened.
    assertThat(shortPathResolver.resolutionQueries)
        .containsExactly(
            "d:/progra~1",
            "d:/program files/micros~1",
            "d:/program files/microsoft something/foo/~bar~1");

    // Assert resolving a path that has a segment which doesn't exist but later will.
    shortPathResolver.resolutionQueries.clear();
    Path notYetExistent = root.getRelative("d:/progra~1/micros~1/foo/will~1.exi/bar");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(notYetExistent.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/will~1.exi/bar");
    // Assert that we only try to resolve the path segments that look like they may be shortened.
    assertThat(shortPathResolver.resolutionQueries)
        .containsExactly(
            "d:/progra~1",
            "d:/program files/micros~1",
            "d:/program files/microsoft something/foo/will~1.exi");

    // Assert that the paths we failed to resolve don't get cached.
    final List<String> children = new ArrayList<>(2);
    Predicate<Path> collector =
        new Predicate<Path>() {
          @Override
          public boolean apply(Path child) {
            children.add(child.getPathString());
            return true;
          }
        };

    WindowsPath msRoot = (WindowsPath) root.getRelative("d:/progra~1/micros~1");
    assertThat(msRoot.getPathString()).isEqualTo("D:/program files/microsoft something");
    msRoot.applyToChildren(collector);
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(children).containsExactly("D:/program files/microsoft something/foo");

    // Assert that the non-resolvable path was not cached.
    children.clear();
    WindowsPath foo = (WindowsPath) msRoot.getRelative("foo");
    foo.applyToChildren(collector);
    assertThat(children).containsExactly("D:/program files/microsoft something/foo/~bar_hello");

    // Pretend that a path we already failed to resolve once came into existence.
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/will~1.exi", "will.exist");

    // Assert that this time we can resolve the previously non-existent path.
    shortPathResolver.resolutionQueries.clear();
    Path nowExists = root.getRelative("d:/progra~1/micros~1/foo/will~1.exi/bar");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(nowExists.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/will.exist/bar");
    // Assert that we only try to resolve the path segments that look like they may be shortened.
    assertThat(shortPathResolver.resolutionQueries)
        .containsExactly(
            "d:/progra~1",
            "d:/program files/micros~1",
            "d:/program files/microsoft something/foo/will~1.exi");

    // Assert that this time we cached the previously non-existent path.
    children.clear();
    foo.applyToChildren(collector);
    // The path strings have upper-case drive letters because that's how path printing works.
    children.clear();
    foo.applyToChildren(collector);
    assertThat(children)
        .containsExactly(
            "D:/program files/microsoft something/foo/~bar_hello",
            "D:/program files/microsoft something/foo/will.exist");
  }

  @Test
  public void testCaseInsensitivePathFragment() {
    // equals
    assertThat(PathFragment.create("c:/FOO/BAR")).isEqualTo(PathFragment.create("c:\\foo\\bar"));
    assertThat(PathFragment.create("c:/FOO/BAR")).isNotEqualTo(PathFragment.create("d:\\foo\\bar"));
    assertThat(PathFragment.create("c:/FOO/BAR")).isNotEqualTo(PathFragment.create("/foo/bar"));
    // equals for the string representation
    assertThat(PathFragment.create("c:/FOO/BAR").toString())
        .isNotEqualTo(PathFragment.create("c:/foo/bar").toString());
    // hashCode
    assertThat(PathFragment.create("c:/FOO/BAR").hashCode())
        .isEqualTo(PathFragment.create("c:\\foo\\bar").hashCode());
    assertThat(PathFragment.create("c:/FOO/BAR").hashCode())
        .isNotEqualTo(PathFragment.create("d:\\foo\\bar").hashCode());
    assertThat(PathFragment.create("c:/FOO/BAR").hashCode())
        .isNotEqualTo(PathFragment.create("/foo/bar").hashCode());
    // compareTo
    assertThat(PathFragment.create("c:/FOO/BAR").compareTo(PathFragment.create("c:\\foo\\bar")))
        .isEqualTo(0);
    assertThat(PathFragment.create("c:/FOO/BAR").compareTo(PathFragment.create("d:\\foo\\bar")))
        .isLessThan(0);
    assertThat(PathFragment.create("c:/FOO/BAR").compareTo(PathFragment.create("/foo/bar")))
        .isGreaterThan(0);
    // startsWith
    assertThat(PathFragment.create("c:/FOO/BAR").startsWith(PathFragment.create("c:\\foo")))
        .isTrue();
    assertThat(PathFragment.create("c:/FOO/BAR").startsWith(PathFragment.create("d:\\foo")))
        .isFalse();
    // endsWith
    assertThat(PathFragment.create("c:/FOO/BAR/BAZ").endsWith(PathFragment.create("bar\\baz")))
        .isTrue();
    assertThat(PathFragment.create("c:/FOO/BAR/BAZ").endsWith(PathFragment.create("/bar/baz")))
        .isFalse();
    assertThat(PathFragment.create("c:/FOO/BAR/BAZ").endsWith(PathFragment.create("d:\\bar\\baz")))
        .isFalse();
    // relativeTo
    assertThat(
            PathFragment.create("c:/FOO/BAR/BAZ/QUX")
                .relativeTo(PathFragment.create("c:\\foo\\bar")))
        .isEqualTo(PathFragment.create("Baz/Qux"));
  }

  @Test
  public void testCaseInsensitiveRootedPath() {
    Path ancestor = filesystem.getPath("C:\\foo\\bar");
    assertThat(ancestor).isInstanceOf(WindowsPath.class);
    Path child = filesystem.getPath("C:\\FOO\\Bar\\baz");
    assertThat(child).isInstanceOf(WindowsPath.class);
    assertThat(child.startsWith(ancestor)).isTrue();
    assertThat(child.relativeTo(ancestor)).isEqualTo(PathFragment.create("baz"));
    RootedPath actual = RootedPath.toRootedPath(ancestor, child);
    assertThat(actual.getRoot()).isEqualTo(ancestor);
    assertThat(actual.getRelativePath()).isEqualTo(PathFragment.create("baz"));
  }

  @Test
  public void testToURI() {
    // See https://blogs.msdn.microsoft.com/ie/2006/12/06/file-uris-in-windows/
    Path p = root.getRelative("Temp\\Foo Bar.txt");
    URI uri = p.toURI();
    assertThat(uri.toString()).isEqualTo("file:///C:/Temp/Foo%20Bar.txt");
  }
}
