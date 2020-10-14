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
import static org.junit.Assert.assertThrows;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.WindowsOsPathPolicy.ShortPathResolver;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests windows-specific parts of {@link Path} */
@RunWith(JUnit4.class)
public class WindowsPathTest extends PathAbstractTest {

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

  @Test
  public void testEqualsAndHashcodeWindows() {
    new EqualsTester()
        .addEqualityGroup(create("/a/b"), create("/A/B"))
        .addEqualityGroup(create("c:/a/b"), create("C:\\A\\B"))
        .addEqualityGroup(create("C:/something/else"))
        .testEquals();
  }

  @Test
  public void testCaseIsPreserved() {
    assertThat(create("C:/a/B").getPathString()).isEqualTo("C:/a/B");
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
    assertThat(create("/").getParentDirectory()).isNull();
  }

  @Test
  public void testParentOfRootIsRootWindows() {
    assertThat(create("C:/..")).isEqualTo(create("C:/"));
    assertThat(create("C:/../../../../../..")).isEqualTo(create("C:/"));
    assertThat(create("C:/../../../foo")).isEqualTo(create("C:/foo"));
  }

  @Test
  public void testRelativeToWindows() {
    assertThat(create("C:/foo").relativeTo(create("C:/")).getPathString()).isEqualTo("foo");
    // Case insensitivity test
    assertThat(create("C:/foo/bar").relativeTo(create("C:/FOO")).getPathString()).isEqualTo("bar");
    assertThrows(IllegalArgumentException.class, () -> create("D:/foo").relativeTo(create("C:/")));
  }

  @Test
  public void testResolvesShortenedPaths() {
    MockShortPathResolver shortPathResolver = new MockShortPathResolver();
    WindowsOsPathPolicy osPathPolicy = new WindowsOsPathPolicy(shortPathResolver);
    shortPathResolver.resolutions.put("d:/progra~1", "program files");
    shortPathResolver.resolutions.put("d:/program files/micros~1", "microsoft something");
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/~bar~1", "~bar_hello");

    // Assert normal shortpath resolution.
    assertThat(normalize(osPathPolicy, "d:/progra~1/micros~1/foo/~bar~1/baz"))
        .isEqualTo("D:/program files/microsoft something/foo/~bar_hello/baz");
    assertThat(normalize(osPathPolicy, "d:/progra~1/micros~1/foo/will~1.exi/bar"))
        .isEqualTo("D:/program files/microsoft something/foo/will~1.exi/bar");

    assertThat(normalize(osPathPolicy, "d:/progra~1/micros~1"))
        .isEqualTo("D:/program files/microsoft something");

    // Pretend that a path we already failed to resolve once came into existence.
    shortPathResolver.resolutions.put(
        "d:/program files/microsoft something/foo/will~1.exi", "will.exist");

    // Assert that this time we can resolve the previously non-existent path.
    // The path string has an upper-case drive letter because that's how path printing works.
    assertThat(normalize(osPathPolicy, "d:/progra~1/micros~1/foo/will~1.exi/bar"))
        .isEqualTo("D:/program files/microsoft something/foo/will.exist/bar");

    // Check needsToNormalized
    assertThat(osPathPolicy.needsToNormalize("d:/progra~1/micros~1/foo/will~1.exi/bar"))
        .isEqualTo(WindowsOsPathPolicy.NEEDS_SHORT_PATH_NORMALIZATION);
    assertThat(osPathPolicy.needsToNormalize("will~1.exi"))
        .isEqualTo(WindowsOsPathPolicy.NEEDS_SHORT_PATH_NORMALIZATION);
    assertThat(osPathPolicy.needsToNormalize("d:/no-normalization"))
        .isEqualTo(WindowsOsPathPolicy.NORMALIZED);
  }

  private static String normalize(OsPathPolicy osPathPolicy, String str) {
    return osPathPolicy.normalize(str, osPathPolicy.needsToNormalize(str));
  }
}
