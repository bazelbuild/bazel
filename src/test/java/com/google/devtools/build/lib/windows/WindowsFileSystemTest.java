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
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.DefaultSyscallCache;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem.NotASymlinkException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.windows.util.WindowsTestUtil;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link WindowsFileSystem}. */
@RunWith(TestParameterInjector.class)
@TestSpec(supportedOs = OS.WINDOWS)
public class WindowsFileSystemTest {
  @TestParameter boolean createSymbolicLinks;

  private WindowsFileSystem fs;
  private Path scratchRoot;
  private WindowsTestUtil testUtil;

  @Before
  public void createScratchDir() throws Exception {
    fs = new WindowsFileSystem(DigestHashFunction.SHA256, createSymbolicLinks);
    scratchRoot = TestUtils.createUniqueTmpDir(fs);
    testUtil = new WindowsTestUtil(scratchRoot.getPathString());
  }

  @After
  public void destroyScratchDir() throws Exception {
    scratchRoot.deleteTree();
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

    assertThrows(
        FileNotFoundException.class,
        () -> WindowsFileSystem.isSymlinkOrJunction(new File(root, "non-existent")));

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
  public void testSymbolicLinkToExistingFile(@TestParameter SymlinkTargetType targetType)
      throws Exception {
    Path linkPath = scratchRoot.getRelative("link");
    Path targetPath = scratchRoot.getRelative("target");
    assertThat(targetPath.getParentDirectory().exists()).isTrue();
    assertThat(targetPath.getParentDirectory().isDirectory()).isTrue();
    FileSystemUtils.writeContentAsLatin1(targetPath, "hello");

    linkPath.createSymbolicLink(targetPath, targetType);

    if (createSymbolicLinks) {
      assertThat(linkPath.isSymbolicLink()).isTrue();
      assertThat(linkPath.readSymbolicLink()).isEqualTo(targetPath.asFragment());
    } else {
      assertThat(linkPath.isSymbolicLink()).isFalse();
      assertThrows(NotASymlinkException.class, () -> linkPath.readSymbolicLink());
    }
    assertThat(linkPath.exists()).isTrue();
    assertThat(linkPath.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(linkPath, ISO_8859_1)).isEqualTo("hello");

    linkPath.delete();
    assertThat(linkPath.exists()).isFalse();
    assertThat(targetPath.exists()).isTrue();
  }

  @Test
  public void testSymbolicLinkToExistingDirectory(@TestParameter SymlinkTargetType targetType)
      throws Exception {
    Path linkPath = scratchRoot.getRelative("link");
    Path linkChildPath = linkPath.getRelative("hello.txt");
    Path targetPath = scratchRoot.getRelative("target");
    Path targetChildPath = targetPath.getRelative("hello.txt");
    targetPath.createDirectory();
    FileSystemUtils.writeContentAsLatin1(targetChildPath, "hello");

    linkPath.createSymbolicLink(targetPath, targetType);

    assertThat(linkPath.isSymbolicLink()).isTrue();
    assertThat(linkPath.readSymbolicLink()).isEqualTo(targetPath.asFragment());
    assertThat(linkPath.exists()).isTrue();
    assertThat(linkPath.isDirectory()).isTrue();
    assertThat(linkChildPath.exists()).isTrue();
    assertThat(linkChildPath.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(linkChildPath, ISO_8859_1)).isEqualTo("hello");

    linkPath.delete();
    assertThat(linkPath.exists()).isFalse();
    assertThat(targetPath.exists()).isTrue();
  }

  @Test
  public void testCreateSymbolicLinkToNonExistingTargetOfUnspecifiedType() throws Exception {
    Path linkPath = scratchRoot.getRelative("link");
    Path targetPath = scratchRoot.getRelative("target");

    linkPath.createSymbolicLink(targetPath, SymlinkTargetType.UNSPECIFIED);

    assertThat(linkPath.isSymbolicLink()).isTrue();
    assertThat(linkPath.readSymbolicLink()).isEqualTo(targetPath.asFragment());
    assertThat(linkPath.exists()).isFalse();

    // Check that a dangling symlink is preferred over a dangling junction when supported.
    // Do this by creating a target of the corresponding type and verifying that it can be accessed.
    if (createSymbolicLinks) {
      FileSystemUtils.writeContentAsLatin1(targetPath, "hello");
      assertThat(linkPath.exists()).isTrue();
      assertThat(linkPath.isFile()).isTrue();
      assertThat(FileSystemUtils.readContent(linkPath, ISO_8859_1)).isEqualTo("hello");
    } else {
      targetPath.createDirectory();
      assertThat(linkPath.exists()).isTrue();
      assertThat(linkPath.isDirectory()).isTrue();
    }
  }

  @Test
  public void testCreateSymbolicLinkToNonExistingTargetOfFileType() throws Exception {
    // This is only expected to work if symlinks are enabled.
    // Otherwise, our only recourse is to create a dangling junction, which does not work for files.
    assumeTrue(createSymbolicLinks);

    Path linkPath = scratchRoot.getRelative("link");
    Path targetPath = scratchRoot.getRelative("target");

    linkPath.createSymbolicLink(targetPath, SymlinkTargetType.FILE);

    assertThat(linkPath.isSymbolicLink()).isTrue();
    assertThat(linkPath.readSymbolicLink()).isEqualTo(targetPath.asFragment());

    FileSystemUtils.writeContentAsLatin1(targetPath, "hello");

    assertThat(linkPath.exists()).isTrue();
    assertThat(linkPath.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(linkPath, ISO_8859_1)).isEqualTo("hello");
  }

  @Test
  public void testCreateSymbolicLinkToNonExistingTargetOfDirectoryType() throws Exception {
    Path linkPath = scratchRoot.getRelative("link");
    Path linkChildPath = linkPath.getRelative("hello.txt");
    Path targetPath = scratchRoot.getRelative("target");
    Path targetChildPath = targetPath.getRelative("hello.txt");

    linkPath.createSymbolicLink(targetPath, SymlinkTargetType.DIRECTORY);

    assertThat(linkPath.isSymbolicLink()).isTrue();
    assertThat(linkPath.readSymbolicLink()).isEqualTo(targetPath.asFragment());

    targetPath.createDirectory();
    FileSystemUtils.writeContentAsLatin1(targetChildPath, "hello");

    assertThat(linkPath.exists()).isTrue();
    assertThat(linkPath.isDirectory()).isTrue();
    assertThat(linkChildPath.exists()).isTrue();
    assertThat(linkChildPath.isFile()).isTrue();
    assertThat(FileSystemUtils.readContent(linkChildPath, ISO_8859_1)).isEqualTo("hello");
  }

  @Test
  public void testReadSymbolicLinkForFile() throws Exception {
    Path filePath = scratchRoot.getRelative("file");
    FileSystemUtils.writeContentAsLatin1(filePath, "hello");

    assertThrows(NotASymlinkException.class, filePath::readSymbolicLink);
  }

  @Test
  public void testReadSymbolicLinkForDirectory() throws Exception {
    Path dirPath = scratchRoot.getRelative("dir");
    dirPath.createDirectory();

    assertThrows(NotASymlinkException.class, dirPath::readSymbolicLink);
  }

  @Test
  public void testReadSymbolicLinkForNonexistentPath() throws Exception {
    Path nonexistentPath = scratchRoot.getRelative("nonexistent");

    assertThrows(FileNotFoundException.class, nonexistentPath::readSymbolicLink);
  }

  @Test
  public void testReadOnlyAttribute() throws Exception {
    testUtil.scratchFile("dir\\hello.txt", "hello");
    testUtil.createJunctions(ImmutableMap.of("junc", "dir"));

    Path dir = testUtil.createVfsPath(fs, "dir");
    Path file = testUtil.createVfsPath(fs, "dir\\hello.txt");
    Path dirViaJunction = testUtil.createVfsPath(fs, "junc");
    Path fileViaJunction = testUtil.createVfsPath(fs, "junc\\hello.txt");

    assertWritable(dir);
    dir.setWritable(false); // no-op
    assertWritable(dir);
    dir.setWritable(true); // no-op
    assertWritable(dir);

    assertWritable(dirViaJunction);
    dirViaJunction.setWritable(false); // no-op
    assertWritable(dirViaJunction);
    dirViaJunction.setWritable(true); // no-op
    assertWritable(dirViaJunction);

    assertWritable(file);
    file.setWritable(false);
    assertNotWritable(file);
    file.setWritable(true);
    assertWritable(file);

    assertThat(fileViaJunction.isWritable()).isTrue();
    fileViaJunction.setWritable(false);
    assertNotWritable(fileViaJunction);
    fileViaJunction.setWritable(true);
    assertWritable(fileViaJunction);
  }

  private static void assertWritable(Path path) throws Exception {
    assertThat(path.isWritable()).isTrue();
    assertThat(path.stat().getPermissions()).isEqualTo(0755);
  }

  private static void assertNotWritable(Path path) throws Exception {
    assertThat(path.isWritable()).isFalse();
    assertThat(path.stat().getPermissions()).isEqualTo(0555);
  }

  @Test
  public void testTypeViaReaddirCache(
      @TestParameter({
            "BUILD", "Å", "K", "Ａ", "ａ", "０", " 𝐀", "𝐴", "𝒜", "Ⅳ", "Ⓑ", "ẞ", "ß", "Ä", "İ", "ı"
          })
          String entry)
      throws Exception {
    var normalizedEntry =
        Normalizer.normalize(entry, Normalizer.Form.NFC)
            .toUpperCase(Locale.ROOT)
            .toLowerCase(Locale.ROOT);
    validateGetTypeConsistency(scratchRoot, entry, normalizedEntry);
    validateGetTypeConsistency(scratchRoot, normalizedEntry, entry);
  }

  private void validateGetTypeConsistency(Path baseDir, String entryToCreate, String entryToCheck)
      throws IOException {
    baseDir.createDirectoryAndParents();
    var dir = baseDir.createTempDirectory("readdir_cache-");
    var pathToCreate = dir.getChild(StringEncoding.unicodeToInternal(entryToCreate));
    FileSystemUtils.createEmptyFile(pathToCreate);

    var syscallCache = DefaultSyscallCache.newBuilder().build();
    // Prime the cache by reading the parent directory.
    syscallCache.readdir(dir);
    assertWithMessage("expecting entry %s to exist", entryToCreate)
        .that(syscallCache.getType(pathToCreate, Symlinks.FOLLOW))
        .isNotNull();

    var pathToCheck = dir.getChild(StringEncoding.unicodeToInternal(entryToCheck));
    var existsWithCache = syscallCache.getType(pathToCheck, Symlinks.FOLLOW) != null;
    var existsWithoutCache = pathToCheck.statIfFound() != null;
    assertWithMessage("created : %s", entryToCreate)
        .withMessage("checking: %s", entryToCheck)
        .withMessage("with cache: %s", existsWithCache)
        .withMessage("w/o cache : %s", existsWithoutCache)
        .that(existsWithCache)
        .isEqualTo(existsWithoutCache);
  }
}
