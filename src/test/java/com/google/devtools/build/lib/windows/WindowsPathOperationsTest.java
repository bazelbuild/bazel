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
package com.google.devtools.build.lib.windows;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.util.WindowsTestUtil;
import java.io.File;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WindowsPathOperations}. */
@RunWith(JUnit4.class)
@TestSpec(supportedOs = OS.WINDOWS)
public class WindowsPathOperationsTest {

  private String scratchRoot;
  private WindowsTestUtil testUtil;

  @Before
  public void setUp() throws Exception {
    scratchRoot = new File(System.getenv("TEST_TMPDIR"), "x").getAbsolutePath();
    testUtil = new WindowsTestUtil(scratchRoot);
    cleanupScratchDir();
  }

  @After
  public void cleanupScratchDir() throws Exception {
    testUtil.deleteAllUnder("");
  }

  @Test
  public void testShortNameMatcher() {
    assertThat(WindowsPathOperations.isShortPath("abc")).isFalse(); // no ~ in the name
    assertThat(WindowsPathOperations.isShortPath("abc~")).isFalse(); // no number after the ~
    assertThat(WindowsPathOperations.isShortPath("~abc")).isFalse(); // no ~ followed by number
    assertThat(WindowsPathOperations.isShortPath("too_long_path")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("too_long_path~1"))
        .isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("abcd~1234")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("h~1")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("h~12")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("h~12.")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("h~12.a")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("h~12.abc")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("h~123456")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hellow~1")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hellow~1.")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hellow~1.a")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hellow~1.abc")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hello~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hellow~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hello~12")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hello~12.")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hello~12.a")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hello~12.abc")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("hello~12.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hellow~12")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hellow~12.")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hellow~12.a")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("hellow~12.ab")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("~h~1")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~1.")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~1.a")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~1.abc")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsPathOperations.isShortPath("~h~12")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~12~1")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~12~1.")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~12~1.a")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~12~1.abc")).isTrue();
    assertThat(WindowsPathOperations.isShortPath("~h~12~1.abcd")).isFalse(); // too long for 8dot3
  }

  @Test
  public void testGetLongPath() throws Exception {
    File foo = testUtil.scratchDir("foo").toAbsolutePath().toFile();
    assertThat(foo.exists()).isTrue();
    assertThat(WindowsPathOperations.getLongPath(foo.getAbsolutePath())).endsWith("foo");

    String longPath = foo.getAbsolutePath() + "\\will.exist\\helloworld.txt";
    String shortPath = foo.getAbsolutePath() + "\\will~1.exi\\hellow~1.txt";

    // Assert that the long path resolution fails for non-existent file.
    assertThat(assertThrows(IOException.class, () -> WindowsPathOperations.getLongPath(longPath)))
        .hasMessageThat()
        .contains("GetLongPathName");
    assertThat(assertThrows(IOException.class, () -> WindowsPathOperations.getLongPath(shortPath)))
        .hasMessageThat()
        .contains("GetLongPathName");

    // Create the file, assert that long path resolution works and is correct.
    File helloFile =
        testUtil.scratchFile("foo/will.exist/helloworld.txt", "hello").toAbsolutePath().toFile();
    assertThat(helloFile.getAbsolutePath()).isEqualTo(longPath);
    assertThat(helloFile.exists()).isTrue();
    assertThat(new File(longPath).exists()).isTrue();
    assertThat(new File(shortPath).exists()).isTrue();
    assertThat(WindowsPathOperations.getLongPath(longPath)).endsWith("will.exist/helloworld.txt");
    assertThat(WindowsPathOperations.getLongPath(shortPath)).endsWith("will.exist/helloworld.txt");

    // Delete the file and the directory, assert that long path resolution fails for them.
    assertThat(helloFile.delete()).isTrue();
    assertThat(helloFile.getParentFile().delete()).isTrue();

    assertThat(assertThrows(IOException.class, () -> WindowsPathOperations.getLongPath(longPath)))
        .hasMessageThat()
        .contains("GetLongPathName");

    assertThat(assertThrows(IOException.class, () -> WindowsPathOperations.getLongPath(shortPath)))
        .hasMessageThat()
        .contains("GetLongPathName");

    // Create the directory and file with different names, but same 8dot3 names, assert that the
    // resolution is still correct.
    helloFile =
        testUtil
            .scratchFile("foo/will.exist_again/hellowelt.txt", "hello")
            .toAbsolutePath()
            .toFile();
    assertThat(helloFile.exists()).isTrue();
    assertThat(new File(shortPath).exists()).isTrue();
    assertThat(WindowsPathOperations.getLongPath(shortPath))
        .endsWith("will.exist_again/hellowelt.txt");
    assertThat(WindowsPathOperations.getLongPath(foo + "\\will.exist_again\\hellowelt.txt"))
        .endsWith("will.exist_again/hellowelt.txt");

    assertThat(assertThrows(IOException.class, () -> WindowsPathOperations.getLongPath(longPath)))
        .hasMessageThat()
        .contains("GetLongPathName");
  }
}
