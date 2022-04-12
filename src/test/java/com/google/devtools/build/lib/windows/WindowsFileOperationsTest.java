// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.util.WindowsTestUtil;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link WindowsFileOperations}. */
@RunWith(JUnit4.class)
@TestSpec(supportedOs = OS.WINDOWS)
public class WindowsFileOperationsTest {

  private String scratchRoot;
  private WindowsTestUtil testUtil;

  @Before
  public void loadJni() throws Exception {
    scratchRoot = new File(System.getenv("TEST_TMPDIR"), "x").getAbsolutePath();
    testUtil = new WindowsTestUtil(scratchRoot);
    cleanupScratchDir();
  }

  @After
  public void cleanupScratchDir() throws Exception {
    testUtil.deleteAllUnder("");
  }

  @Test
  public void testMockJunctionCreation() throws Exception {
    String root = testUtil.scratchDir("dir").getParent().toString();
    testUtil.scratchFile("dir/file.txt", "hello");
    testUtil.createJunctions(ImmutableMap.of("junc", "dir"));
    String[] children = new File(root + "/junc").list();
    assertThat(children).isNotNull();
    assertThat(children).hasLength(1);
    assertThat(Arrays.asList(children)).containsExactly("file.txt");
  }

  @Test
  public void testSymlinkCreation() throws Exception {
    File helloFile = testUtil.scratchFile("file.txt", "hello").toFile();
    File symlinkFile = new File(scratchRoot, "symlink");
    testUtil.createSymlinks(ImmutableMap.of("symlink", "file.txt"));

    assertThat(WindowsFileOperations.isSymlinkOrJunction(symlinkFile.toString())).isTrue();
    assertThat(symlinkFile.exists()).isTrue();

    // Assert deleting the symlink does not remove the target file.
    assertThat(WindowsFileOperations.deletePath(symlinkFile.toString())).isTrue();
    assertThat(helloFile.exists()).isTrue();
    try {
      WindowsFileOperations.isSymlinkOrJunction(symlinkFile.toString());
      fail("Expected to throw: Symlink should no longer exist.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("path does not exist");
    }
  }

  @Test
  public void testSymlinkCreationFailsForDirectory() throws Exception {
    testUtil.scratchDir("dir").toFile();

    try {
      testUtil.createSymlinks(ImmutableMap.of("symlink", "dir"));
      fail("Expected to throw: Symlinks to a directory should fail.");
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("target is a directory");
    }
  }

  @Test
  public void testIsJunction() throws Exception {
    final Map<String, String> junctions = new HashMap<>();
    junctions.put("shrtpath/a", "shrttrgt");
    junctions.put("shrtpath/b", "longtargetpath");
    junctions.put("shrtpath/c", "longta~1");
    junctions.put("longlinkpath/a", "shrttrgt");
    junctions.put("longlinkpath/b", "longtargetpath");
    junctions.put("longlinkpath/c", "longta~1");
    junctions.put("abbrev~1/a", "shrttrgt");
    junctions.put("abbrev~1/b", "longtargetpath");
    junctions.put("abbrev~1/c", "longta~1");

    String root = testUtil.scratchDir("shrtpath").getParent().toAbsolutePath().toString();
    testUtil.scratchDir("longlinkpath");
    testUtil.scratchDir("abbreviated");
    testUtil.scratchDir("control/a");
    testUtil.scratchDir("control/b");
    testUtil.scratchDir("control/c");

    testUtil.scratchFile("shrttrgt/file1.txt", "hello");
    testUtil.scratchFile("longtargetpath/file2.txt", "hello");

    testUtil.createJunctions(junctions);

    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\shrtpath\\a")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\shrtpath\\b")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\shrtpath\\c")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longlinkpath\\a")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longlinkpath\\b")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longlinkpath\\c")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longli~1\\a")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longli~1\\b")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longli~1\\c")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbreviated\\a")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbreviated\\b")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbreviated\\c")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbrev~1\\a")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbrev~1\\b")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\abbrev~1\\c")).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\control\\a")).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\control\\b")).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\control\\c")).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\shrttrgt\\file1.txt")).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longtargetpath\\file2.txt"))
        .isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(root + "\\longta~1\\file2.txt")).isFalse();
    try {
      WindowsFileOperations.isSymlinkOrJunction(root + "\\non-existent");
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("path does not exist");
    }
    assertThat(Arrays.asList(new File(root + "/shrtpath/a").list())).containsExactly("file1.txt");
    assertThat(Arrays.asList(new File(root + "/shrtpath/b").list())).containsExactly("file2.txt");
    assertThat(Arrays.asList(new File(root + "/shrtpath/c").list())).containsExactly("file2.txt");
    assertThat(Arrays.asList(new File(root + "/longlinkpath/a").list()))
        .containsExactly("file1.txt");
    assertThat(Arrays.asList(new File(root + "/longlinkpath/b").list()))
        .containsExactly("file2.txt");
    assertThat(Arrays.asList(new File(root + "/longlinkpath/c").list()))
        .containsExactly("file2.txt");
    assertThat(Arrays.asList(new File(root + "/abbreviated/a").list()))
        .containsExactly("file1.txt");
    assertThat(Arrays.asList(new File(root + "/abbreviated/b").list()))
        .containsExactly("file2.txt");
    assertThat(Arrays.asList(new File(root + "/abbreviated/c").list()))
        .containsExactly("file2.txt");
  }

  @Test
  public void testIsJunctionIsTrueForDanglingJunction() throws Exception {
    java.nio.file.Path helloPath = testUtil.scratchFile("target\\hello.txt", "hello");
    testUtil.createJunctions(ImmutableMap.of("link", "target"));

    File linkPath = new File(helloPath.getParent().getParent().toFile(), "link");
    assertThat(Arrays.asList(linkPath.list())).containsExactly("hello.txt");
    assertThat(WindowsFileOperations.isSymlinkOrJunction(linkPath.getAbsolutePath())).isTrue();

    assertThat(helloPath.toFile().delete()).isTrue();
    assertThat(helloPath.getParent().toFile().delete()).isTrue();
    assertThat(helloPath.getParent().toFile().exists()).isFalse();
    assertThat(Arrays.asList(linkPath.getParentFile().list())).containsExactly("link");

    assertThat(WindowsFileOperations.isSymlinkOrJunction(linkPath.getAbsolutePath())).isTrue();
    assertThat(
            Files.exists(
                linkPath.toPath(), WindowsFileSystem.symlinkOpts(/* followSymlinks */ false)))
        .isTrue();
    assertThat(
            Files.exists(
                linkPath.toPath(), WindowsFileSystem.symlinkOpts(/* followSymlinks */ true)))
        .isFalse();
  }

  @Test
  public void testIsJunctionHandlesFilesystemChangesCorrectly() throws Exception {
    File helloFile =
        testUtil.scratchFile("target\\helloworld.txt", "hello").toAbsolutePath().toFile();

    // Assert that a file is identified as not a junction.
    String longPath = helloFile.getAbsolutePath();
    String shortPath = new File(helloFile.getParentFile(), "hellow~1.txt").getAbsolutePath();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(longPath)).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(shortPath)).isFalse();

    // Assert that after deleting the file and creating a junction with the same path, it is
    // identified as a junction.
    assertThat(helloFile.delete()).isTrue();
    testUtil.createJunctions(ImmutableMap.of("target\\helloworld.txt", "target"));
    assertThat(WindowsFileOperations.isSymlinkOrJunction(longPath)).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(shortPath)).isTrue();

    // Assert that after deleting the file and creating a directory with the same path, it is
    // identified as not a junction.
    assertThat(helloFile.delete()).isTrue();
    assertThat(helloFile.mkdir()).isTrue();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(longPath)).isFalse();
    assertThat(WindowsFileOperations.isSymlinkOrJunction(shortPath)).isFalse();
  }

  @Test
  public void testGetLongPath() throws Exception {
    File foo = testUtil.scratchDir("foo").toAbsolutePath().toFile();
    assertThat(foo.exists()).isTrue();
    assertThat(WindowsFileOperations.getLongPath(foo.getAbsolutePath())).endsWith("foo");

    String longPath = foo.getAbsolutePath() + "\\will.exist\\helloworld.txt";
    String shortPath = foo.getAbsolutePath() + "\\will~1.exi\\hellow~1.txt";

    // Assert that the long path resolution fails for non-existent file.
    try {
      WindowsFileOperations.getLongPath(longPath);
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("GetLongPathName");
    }
    try {
      WindowsFileOperations.getLongPath(shortPath);
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("GetLongPathName");
    }

    // Create the file, assert that long path resolution works and is correct.
    File helloFile =
        testUtil.scratchFile("foo/will.exist/helloworld.txt", "hello").toAbsolutePath().toFile();
    assertThat(helloFile.getAbsolutePath()).isEqualTo(longPath);
    assertThat(helloFile.exists()).isTrue();
    assertThat(new File(longPath).exists()).isTrue();
    assertThat(new File(shortPath).exists()).isTrue();
    assertThat(WindowsFileOperations.getLongPath(longPath)).endsWith("will.exist/helloworld.txt");
    assertThat(WindowsFileOperations.getLongPath(shortPath)).endsWith("will.exist/helloworld.txt");

    // Delete the file and the directory, assert that long path resolution fails for them.
    assertThat(helloFile.delete()).isTrue();
    assertThat(helloFile.getParentFile().delete()).isTrue();
    try {
      WindowsFileOperations.getLongPath(longPath);
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("GetLongPathName");
    }
    try {
      WindowsFileOperations.getLongPath(shortPath);
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("GetLongPathName");
    }

    // Create the directory and file with different names, but same 8dot3 names, assert that the
    // resolution is still correct.
    helloFile =
        testUtil
            .scratchFile("foo/will.exist_again/hellowelt.txt", "hello")
            .toAbsolutePath()
            .toFile();
    assertThat(new File(shortPath).exists()).isTrue();
    assertThat(WindowsFileOperations.getLongPath(shortPath))
        .endsWith("will.exist_again/hellowelt.txt");
    assertThat(WindowsFileOperations.getLongPath(foo + "\\will.exist_again\\hellowelt.txt"))
        .endsWith("will.exist_again/hellowelt.txt");
    try {
      WindowsFileOperations.getLongPath(longPath);
      fail("expected to throw");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("GetLongPathName");
    }
  }
}
