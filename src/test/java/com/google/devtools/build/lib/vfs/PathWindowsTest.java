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
import static com.google.devtools.build.lib.vfs.WindowsFileSystem.SHORT_NAME_MATCHER;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.Path.PathFactory;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
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
  private Path root;
  private final MockShortPathResolver shortPathResolver = new MockShortPathResolver();

  @Before
  public final void initializeFileSystem() throws Exception {
    filesystem =
        new InMemoryFileSystem(BlazeClock.instance()) {
          @Override
          protected PathFactory getPathFactory() {
            return WindowsFileSystem.getPathFactoryForTesting(shortPathResolver);
          }
        };
    root = filesystem.getRootDirectory().getRelative("C:/");
    root.createDirectory();
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
    assertEquals("C:/first/x/y", dir.getRelative(new PathFragment("y")).toString());
    assertEquals("C:/first/x/x", dir.getRelative(new PathFragment("./x")).toString());
    assertEquals("C:/first/y", dir.getRelative(new PathFragment("../y")).toString());
    assertEquals("C:/first/y", dir.getRelative(new PathFragment("../y")).toString());
    assertEquals("C:/y", dir.getRelative(new PathFragment("../../../y")).toString());
  }

  @Test
  public void testGetRelativeWithAbsoluteFragmentWindows() {
    Path x = filesystem.getPath("C:/first/x");
    assertEquals("C:/x/y", x.getRelative(new PathFragment("C:/x/y")).toString());
  }

  @Test
  public void testGetRelativeWithAbsoluteStringWorksWindows() {
    Path x = filesystem.getPath("C:/first/x");
    assertEquals("C:/x/y", x.getRelative("C:/x/y").toString());
  }

  @Test
  public void testParentOfRootIsRootWindows() {
    assertSame(root, root.getRelative(".."));
    assertSame(root.getRelative("dots"), root.getRelative("broken/../../dots"));
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
  public void testChildRegistrationWithTranslatedPaths() {
    // Ensure the Path to "/usr" (actually "C:/fake/msys/usr") is created, path parents/children
    // properly registered.
    Path usrPath = root.getRelative("/usr");
    root.getRelative("dummy_path");

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
    assertThat(children).containsAllOf("C:/fake", "C:/dummy_path");

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

  @Test
  public void testShortNameMatcher() {
    assertThat(SHORT_NAME_MATCHER.apply("abc")).isFalse(); // no ~ in the name
    assertThat(SHORT_NAME_MATCHER.apply("abc~")).isFalse(); // no number after the ~
    assertThat(SHORT_NAME_MATCHER.apply("~abc")).isFalse(); // no ~ followed by number
    assertThat(SHORT_NAME_MATCHER.apply("too_long_path")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("too_long_path~1")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("abcd~1234")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("h~1")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("h~12")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("h~12.")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("h~12.a")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("h~12.abc")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("h~123456")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hellow~1")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hellow~1.")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hellow~1.a")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hellow~1.abc")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hello~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hellow~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hello~12")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hello~12.")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hello~12.a")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hello~12.abc")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("hello~12.abcd")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hellow~12")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hellow~12.")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hellow~12.a")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("hellow~12.ab")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("~h~1")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~1.")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~1.a")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~1.abc")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(SHORT_NAME_MATCHER.apply("~h~12")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~12~1")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~12~1.")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~12~1.a")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~12~1.abc")).isTrue();
    assertThat(SHORT_NAME_MATCHER.apply("~h~12~1.abcd")).isFalse(); // too long for 8dot3
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

    Path msRoot = root.getRelative("d:/progra~1/micros~1");
    assertThat(msRoot.getPathString()).isEqualTo("D:/program files/microsoft something");
    msRoot.applyToChildren(collector);
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(children).containsExactly("D:/program files/microsoft something/foo");

    // Assert that the non-resolvable path was not cached.
    children.clear();
    msRoot.getRelative("foo").applyToChildren(collector);
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
    msRoot.getRelative("foo").applyToChildren(collector);
    // The path strings have upper-case drive letters because that's how path printing works.
    children.clear();
    msRoot.getRelative("foo").applyToChildren(collector);
    assertThat(children)
        .containsExactly(
            "D:/program files/microsoft something/foo/~bar_hello",
            "D:/program files/microsoft something/foo/will.exist");
  }
}
