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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
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

/** Unit tests for {@link WindowsFileSystem}. */
@RunWith(JUnit4.class)
@TestSpec(supportedOs = OS.WINDOWS)
public class WindowsFileSystemTest {

  private WindowsFileSystem fs;
  private Path scratchRoot;
  private WindowsTestUtil testUtil;

  @Before
  public void loadJni() throws Exception {
    fs = new WindowsFileSystem(DigestHashFunction.SHA256, /*createSymbolicLinks=*/ false);
    scratchRoot = fs.getPath(System.getenv("TEST_TMPDIR")).getRelative("test").getRelative("x");
    scratchRoot.createDirectoryAndParents();
    testUtil = new WindowsTestUtil(scratchRoot.getPathString());
    cleanupScratchDir();
  }

  @After
  public void cleanupScratchDir() throws Exception {
    testUtil.deleteAllUnder("");
  }

  @Test
  public void testCanWorkWithJunctionSymlinks() throws Exception {
    testUtil.scratchFile("dir\\hello.txt", "hello");
    testUtil.scratchDir("non_existent");
    testUtil.createJunctions(ImmutableMap.of("junc", "dir", "junc_bad", "non_existent"));

    Path juncPath = testUtil.createVfsPath(fs, "junc");
    Path dirPath = testUtil.createVfsPath(fs, "dir");
    Path juncBadPath = testUtil.createVfsPath(fs, "junc_bad");
    Path nonExistentPath = testUtil.createVfsPath(fs, "non_existent");

    // Test junction creation.
    assertThat(juncPath.exists(Symlinks.NOFOLLOW)).isTrue();
    assertThat(dirPath.exists(Symlinks.NOFOLLOW)).isTrue();
    assertThat(juncBadPath.exists(Symlinks.NOFOLLOW)).isTrue();
    assertThat(nonExistentPath.exists(Symlinks.NOFOLLOW)).isTrue();

    // Test recognizing and dereferencing a directory junction.
    assertThat(juncPath.isSymbolicLink()).isTrue();
    assertThat(juncPath.isDirectory(Symlinks.FOLLOW)).isTrue();
    assertThat(juncPath.isDirectory(Symlinks.NOFOLLOW)).isFalse();
    assertThat(juncPath.getDirectoryEntries())
        .containsExactly(testUtil.createVfsPath(fs, "junc\\hello.txt"));

    // Test deleting a directory junction.
    assertThat(juncPath.delete()).isTrue();
    assertThat(juncPath.exists(Symlinks.NOFOLLOW)).isFalse();

    // Test recognizing a dangling directory junction.
    assertThat(nonExistentPath.delete()).isTrue();
    assertThat(nonExistentPath.exists(Symlinks.NOFOLLOW)).isFalse();
    assertThat(juncBadPath.exists(Symlinks.NOFOLLOW)).isTrue();
    // TODO(bazel-team): fix https://github.com/bazelbuild/bazel/issues/1690 and uncomment the
    // assertion below.
    // assertThat(fs.isSymbolicLink(juncBadPath)).isTrue();
    assertThat(fs.isDirectory(juncBadPath.asFragment(), /* followSymlinks */ true)).isFalse();
    assertThat(fs.isDirectory(juncBadPath.asFragment(), /* followSymlinks */ false)).isFalse();

    // Test deleting a dangling junction.
    assertThat(juncBadPath.delete()).isTrue();
    assertThat(juncBadPath.exists(Symlinks.NOFOLLOW)).isFalse();
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

    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "shrtpath/a"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "shrtpath/b"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "shrtpath/c"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longlinkpath/a"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longlinkpath/b"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longlinkpath/c"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longli~1/a"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longli~1/b"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longli~1/c"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbreviated/a"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbreviated/b"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbreviated/c"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbrev~1/a"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbrev~1/b"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "abbrev~1/c"))).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "control/a"))).isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "control/b"))).isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "control/c"))).isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "shrttrgt/file1.txt")))
        .isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longtargetpath/file2.txt")))
        .isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(root, "longta~1/file2.txt")))
        .isFalse();
    try {
      WindowsFileSystem.isSymlinkOrJunction(new File(root, "non-existent"));
      fail("expected failure");
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
    assertThat(WindowsFileSystem.isSymlinkOrJunction(linkPath)).isTrue();

    assertThat(helloPath.toFile().delete()).isTrue();
    assertThat(helloPath.getParent().toFile().delete()).isTrue();
    assertThat(helloPath.getParent().toFile().exists()).isFalse();
    assertThat(Arrays.asList(linkPath.getParentFile().list())).containsExactly("link");

    assertThat(WindowsFileSystem.isSymlinkOrJunction(linkPath)).isTrue();
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
    File longPath =
        testUtil.scratchFile("target\\helloworld.txt", "hello").toAbsolutePath().toFile();
    File shortPath = new File(longPath.getParentFile(), "hellow~1.txt");
    assertThat(WindowsFileSystem.isSymlinkOrJunction(longPath)).isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(shortPath)).isFalse();

    assertThat(longPath.delete()).isTrue();
    testUtil.createJunctions(ImmutableMap.of("target\\helloworld.txt", "target"));
    assertThat(WindowsFileSystem.isSymlinkOrJunction(longPath)).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(shortPath)).isTrue();

    assertThat(longPath.delete()).isTrue();
    assertThat(longPath.mkdir()).isTrue();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(longPath)).isFalse();
    assertThat(WindowsFileSystem.isSymlinkOrJunction(shortPath)).isFalse();
  }

  @Test
  public void testShortPathResolution() throws Exception {
    String shortPath = "shortp~1.res/foo/withsp~1/bar/~witht~1/hello.txt";
    String longPath = "shortpath.resolution/foo/with spaces/bar/~with tilde/hello.txt";
    testUtil.scratchFile(longPath, "hello");
    Path p = scratchRoot.getRelative(shortPath);
    assertThat(p.getPathString()).endsWith(longPath);
    assertThat(p).isEqualTo(scratchRoot.getRelative(shortPath));
    assertThat(p).isEqualTo(scratchRoot.getRelative(longPath));
    assertThat(scratchRoot.getRelative(shortPath)).isEqualTo(p);
    assertThat(scratchRoot.getRelative(longPath)).isEqualTo(p);
  }

  @Test
  public void testUnresolvableShortPathWhichIsThenCreated() throws Exception {
    String shortPath = "unreso~1.sho/foo/will~1.exi/bar/hello.txt";
    String longPath = "unresolvable.shortpath/foo/will.exist/bar/hello.txt";
    // Assert that we can create an unresolvable path.
    Path p = scratchRoot.getRelative(shortPath);
    assertThat(p.getPathString()).endsWith(shortPath);
    // Assert that we can then create the whole path, and can now resolve the short form.
    testUtil.scratchFile(longPath, "hello");
    Path q = scratchRoot.getRelative(shortPath);
    assertThat(q.getPathString()).endsWith(longPath);
    assertThat(p).isNotEqualTo(q);
  }

  /**
   * Test the scenario when a short path resolves to different long ones over time.
   *
   * <p>This can happen if the user deletes a directory during the bazel server's lifetime, then
   * recreates it with the same name prefix such that the resulting directory's 8dot3 name is the
   * same as the old one's.
   */
  @Test
  public void testShortPathResolvesToDifferentPathsOverTime() throws Exception {
    Path p1 = scratchRoot.getRelative("longpa~1");
    Path p2 = scratchRoot.getRelative("longpa~1");
    assertThat(p1.exists()).isFalse();
    assertThat(p1).isEqualTo(p2);

    testUtil.scratchDir("longpathnow");
    Path q1 = scratchRoot.getRelative("longpa~1");
    assertThat(q1.exists()).isTrue();
    assertThat(q1).isEqualTo(scratchRoot.getRelative("longpathnow"));

    // Delete the original resolution of "longpa~1" ("longpathnow").
    assertThat(q1.delete()).isTrue();
    assertThat(q1.exists()).isFalse();

    // Create a directory whose 8dot3 name is also "longpa~1" but its long name is different.
    testUtil.scratchDir("longpaththen");
    Path r1 = scratchRoot.getRelative("longpa~1");
    assertThat(r1.exists()).isTrue();
    assertThat(r1).isEqualTo(scratchRoot.getRelative("longpaththen"));
  }

  @Test
  public void testCreateSymbolicLink() throws Exception {
    // Create the `scratchRoot` directory.
    assertThat(scratchRoot.createDirectory()).isTrue();
    // Create symlink with directory target, relative path.
    Path link1 = scratchRoot.getRelative("link1");
    fs.createSymbolicLink(link1.asFragment(), PathFragment.create(".."));
    // Create symlink with directory target, absolute path.
    Path link2 = scratchRoot.getRelative("link2");
    fs.createSymbolicLink(link2.asFragment(), scratchRoot.getRelative("link1").asFragment());
    // Create scratch files that'll be symlink targets.
    testUtil.scratchFile("foo.txt", "hello");
    testUtil.scratchFile("bar.txt", "hello");
    // Create symlink with file target, relative path.
    Path link3 = scratchRoot.getRelative("link3");
    fs.createSymbolicLink(link3.asFragment(), PathFragment.create("foo.txt"));
    // Create symlink with file target, absolute path.
    Path link4 = scratchRoot.getRelative("link4");
    fs.createSymbolicLink(link4.asFragment(), scratchRoot.getRelative("bar.txt").asFragment());
    // Assert that link1 and link2 are true junctions and have the right contents.
    for (Path p : ImmutableList.of(link1, link2)) {
      assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(p.getPathString()))).isTrue();
      assertThat(p.isSymbolicLink()).isTrue();
      assertThat(
              Iterables.transform(
                  Arrays.asList(new File(p.getPathString()).listFiles()),
                  new Function<File, String>() {
                    @Override
                    public String apply(File input) {
                      return input.getName();
                    }
                  }))
          .containsExactly("x");
    }
    // Assert that link3 and link4 are copies of files.
    for (Path p : ImmutableList.of(link3, link4)) {
      assertThat(WindowsFileSystem.isSymlinkOrJunction(new File(p.getPathString()))).isFalse();
      assertThat(p.isSymbolicLink()).isFalse();
      assertThat(p.isFile()).isTrue();
    }
  }

  @Test
  public void testCreateSymbolicLinkWithRealSymlinks() throws Exception {
    fs = new WindowsFileSystem(DigestHashFunction.SHA256, /*createSymbolicLinks=*/ true);
    java.nio.file.Path helloPath = testUtil.scratchFile("hello.txt", "hello");
    PathFragment targetFragment = PathFragment.create(helloPath.toString());
    Path linkPath = scratchRoot.getRelative("link.txt");
    fs.createSymbolicLink(linkPath.asFragment(), targetFragment);

    assertThat(linkPath.isSymbolicLink()).isTrue();
    assertThat(linkPath.readSymbolicLink()).isEqualTo(targetFragment);

    // Assert deleting the symbolic link keeps the target file.
    linkPath.delete();
    assertThat(linkPath.exists()).isFalse();
    assertThat(helloPath.toFile().exists()).isTrue();
  }

  @Test
  public void testReadJunction() throws Exception {
    testUtil.scratchFile("dir\\hello.txt", "hello");
    testUtil.createJunctions(ImmutableMap.of("junc", "dir"));

    Path dirPath = testUtil.createVfsPath(fs, "dir");
    Path juncPath = testUtil.createVfsPath(fs, "junc");

    assertThat(dirPath.isDirectory()).isTrue();
    assertThat(juncPath.isDirectory()).isTrue();

    assertThat(dirPath.isSymbolicLink()).isFalse();
    assertThat(juncPath.isSymbolicLink()).isTrue();

    try {
      testUtil.createVfsPath(fs, "does-not-exist").readSymbolicLink();
      fail("expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().matches(".*path does not exist");
    }

    try {
      testUtil.createVfsPath(fs, "dir\\hello.txt").readSymbolicLink();
      fail("expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().matches(".*is not a symlink");
    }

    try {
      dirPath.readSymbolicLink();
      fail("expected exception");
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().matches(".*is not a symlink");
    }

    assertThat(juncPath.readSymbolicLink()).isEqualTo(dirPath.asFragment());
  }
}
