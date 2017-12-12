// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.vfs.LocalPath.OsPathPolicy;
import com.google.devtools.build.lib.vfs.LocalPath.WindowsOsPathPolicy;
import com.google.devtools.build.lib.vfs.LocalPath.WindowsOsPathPolicy.ShortPathResolver;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests windows-specific parts of {@link LocalPath} */
@RunWith(JUnit4.class)
public class WindowsLocalPathTest extends LocalPathAbstractTest {

  private static final class MockShortPathResolver implements ShortPathResolver {
    // Full path to resolved child mapping.
    private Map<String, String> resolutions = new HashMap<>();

    @Override
    public String resolveShortPath(String path) {
      String[] segments = path.split("[\\\\/]+");
      String result = "";
      for (int i = 0; i < segments.length; ) {
        String segment = segments[i];
        String queryString = (result + segment).toLowerCase();
        segment = resolutions.getOrDefault(queryString, segment);
        result = result + segment;
        ++i;
        if (i != segments.length) {
          result += "/";
        }
      }
      return result;
    }
  }

  private final MockShortPathResolver shortPathResolver = new MockShortPathResolver();

  @Override
  protected OsPathPolicy getFilePathOs() {
    return new WindowsOsPathPolicy(shortPathResolver);
  }

  @Test
  public void testEqualsAndHashcodeWindows() {
    new EqualsTester()
        .addEqualityGroup(create("a/b"), create("A/B"))
        .addEqualityGroup(create("/a/b"), create("/A/B"))
        .addEqualityGroup(create("c:/a/b"), create("C:\\A\\B"))
        .addEqualityGroup(create("something/else"))
        .testEquals();
  }

  @Test
  public void testCaseIsPreserved() {
    assertThat(create("a/B").getPathString()).isEqualTo("a/B");
  }

  @Test
  public void testNormalizeWindows() {
    assertThat(create("C:/")).isEqualTo(create("C:/"));
    assertThat(create("c:/")).isEqualTo(create("C:/"));
    assertThat(create("c:\\")).isEqualTo(create("C:/"));
    assertThat(create("c:\\foo\\..\\bar\\")).isEqualTo(create("C:/bar"));
  }

  @Test
  public void testStartsWithWindows() {
    assertThat(create("C:/").startsWith(create("C:/"))).isTrue();
    assertThat(create("C:/foo").startsWith(create("C:/"))).isTrue();
    assertThat(create("C:/foo").startsWith(create("D:/"))).isFalse();

    // Case insensitivity test
    assertThat(create("C:/foo/bar").startsWith(create("C:/FOO"))).isTrue();
  }

  @Test
  public void testGetParentDirectoryWindows() {
    assertThat(create("C:/foo").getParentDirectory()).isEqualTo(create("C:/"));
    assertThat(create("C:/").getParentDirectory()).isNull();
  }

  @Test
  public void testisAbsoluteWindows() {
    assertThat(create("C:/").isAbsolute()).isTrue();
    // test that msys paths turn into absolute paths
    assertThat(create("/").isAbsolute()).isTrue();
  }

  @Test
  public void testRelativeToWindows() {
    assertThat(create("C:/foo").relativeTo(create("C:/"))).isEqualTo(create("foo"));
    // Case insensitivity test
    assertThat(create("C:/foo/bar").relativeTo(create("C:/FOO"))).isEqualTo(create("bar"));
    MoreAsserts.assertThrows(
        IllegalArgumentException.class, () -> create("D:/foo").relativeTo(create("C:/")));
  }

  @Test
  public void testAbsoluteUnixPathIsRelativeToWindowsUnixRoot() {
    assertThat(create("/").getPathString()).isEqualTo("C:/fake/msys");
    assertThat(create("/foo/bar").getPathString()).isEqualTo("C:/fake/msys/foo/bar");
    assertThat(create("/foo/bar").getPathString()).isEqualTo("C:/fake/msys/foo/bar");
  }

  @Test
  public void testAbsoluteUnixPathReferringToDriveIsRecognized() {
    assertThat(create("/c/foo").getPathString()).isEqualTo("C:/foo");
    assertThat(create("/c/foo").getPathString()).isEqualTo("C:/foo");
    assertThat(create("/c:").getPathString()).isNotEqualTo("C:/foo");
  }

  @Test
  public void testResolvesShortenedPaths() {
    shortPathResolver.resolutions.put("d:/progra~1", "program files");
    shortPathResolver.resolutions.put("d:/program files/micros~1", "microsoft something");
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/~bar~1", "~bar_hello");

    // Assert normal shortpath resolution.
    LocalPath normal = create("d:/progra~1/micros~1/foo/~bar~1/baz");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(normal.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/~bar_hello/baz");
    LocalPath notYetExistent = create("d:/progra~1/micros~1/foo/will~1.exi/bar");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(notYetExistent.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/will~1.exi/bar");

    LocalPath msRoot = create("d:/progra~1/micros~1");
    assertThat(msRoot.getPathString()).isEqualTo("D:/program files/microsoft something");

    // Pretend that a path we already failed to resolve once came into existence.
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/will~1.exi", "will.exist");

    // Assert that this time we can resolve the previously non-existent path.
    LocalPath nowExists = create("d:/progra~1/micros~1/foo/will~1.exi/bar");
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(nowExists.getPathString())
        .isEqualTo("D:/program files/microsoft something/foo/will.exist/bar");

    // Assert relative paths that look like short paths are untouched
    assertThat(create("progra~1").getPathString()).isEqualTo("progra~1");
  }
}
