// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixFileSystem;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

/**
 * Tests for {@code LinuxSandboxedStrategy}.
 *
 * <p>The general idea for each test is to provide a file tree consisting of symlinks, directories
 * and empty files and then handing that together with an arbitrary number of input files (what
 * would be specified in the "srcs" attribute, for example) to the LinuxSandboxedStrategy.
 *
 * <p>The algorithm that processes the mounts must then always find (and thus mount) the expected
 * tree of files given only the set of input files.
 */
@RunWith(JUnit4.class)
public class LinuxSandboxedStrategyTest {
  private FileSystem testFS;
  private Path workingDir;
  private Path fakeSandboxDir;

  @Before
  public void setUp() throws Exception {
    testFS = new UnixFileSystem();
    workingDir = testFS.getPath(new File(TestUtils.tmpDir()).getCanonicalPath());
    fakeSandboxDir = workingDir.getRelative("sandbox");
    fakeSandboxDir.createDirectory();
  }

  @After
  public void tearDown() throws Exception {
    FileSystemUtils.deleteTreesBelow(workingDir);
  }

  private Path getSandboxPath(Path entry) {
    return fakeSandboxDir.getRelative(entry.asFragment().relativeTo("/"));
  }

  /**
   * Strips the working directory (which can be very long) from the file names in the input map, to
   * make assertion failures easier to read.
   */
  private ImmutableSetMultimap<String, String> userFriendlyMap(Multimap<Path, Path> input) {
    ImmutableSetMultimap.Builder<String, String> userFriendlyMap = ImmutableSetMultimap.builder();
    for (Entry<Path, Path> entry : input.entries()) {
      String key = entry.getKey().getPathString().replace(workingDir.getPathString(), "");
      String value = entry.getValue().getPathString().replace(workingDir.getPathString(), "");
      userFriendlyMap.put(key, value);
    }
    return userFriendlyMap.build();
  }

  private void createTreeStructure(Multimap<String, String> linksAndFiles) throws IOException {
    for (Entry<String, String> entry : linksAndFiles.entries()) {
      Path filePath = workingDir.getRelative(entry.getKey());
      String linkTarget = entry.getValue();

      FileSystemUtils.createDirectoryAndParents(filePath.getParentDirectory());

      if (!linkTarget.isEmpty()) {
        filePath.createSymbolicLink(new PathFragment(linkTarget));
      } else if (filePath.getPathString().endsWith("/")) {
        filePath.createDirectory();
      } else {
        FileSystemUtils.createEmptyFile(filePath);
      }
    }
  }

  /**
   * Takes a Multimap of file specifications, creates the necessary files / symlinks / dirs,
   * mounts files listed in customMount at their canonical location in the sandbox and returns the
   * output of {@code LinuxSandboxedStrategy#fixMounts} for it.
   */
  private ImmutableSetMultimap<Path, Path> mounts(
      Multimap<String, String> linksAndFiles, List<String> customMounts) throws IOException {
    createTreeStructure(linksAndFiles);

    ImmutableSetMultimap.Builder<Path, Path> mounts = ImmutableSetMultimap.builder();
    for (String customMount : customMounts) {
      Path customMountPath = workingDir.getRelative(customMount);
      mounts.put(customMountPath, getSandboxPath(customMountPath));
    }
    return LinuxSandboxedStrategy.validateMounts(
        fakeSandboxDir,
        LinuxSandboxedStrategy.withResolvedSymlinks(
            fakeSandboxDir, LinuxSandboxedStrategy.withRecursedDirs(mounts.build())));
  }

  private ImmutableSetMultimap<String, String> userFriendlyMounts(
      Multimap<String, String> linksAndFiles, List<String> customMounts) throws IOException {
    return userFriendlyMap(mounts(linksAndFiles, customMounts));
  }

  /**
   * Takes a Multimap of file specifications, creates the necessary files / symlinks / dirs,
   * mounts the first file of the specification at its canonical location in the sandbox and returns
   * the output of {@code LinuxSandboxedStrategy#fixMounts} for it.
   */
  private ImmutableSetMultimap<Path, Path> mounts(Multimap<String, String> linksAndFiles)
      throws IOException {
    return mounts(linksAndFiles, ImmutableList.of(Iterables.getFirst(linksAndFiles.keys(), null)));
  }

  private ImmutableSetMultimap<String, String> userFriendlyMounts(
      Multimap<String, String> linksAndFiles) throws IOException {
    return userFriendlyMap(mounts(linksAndFiles));
  }

  /**
   * Returns a Multimap of mount entries for a list files, which can be used to assert that all
   * expected mounts have been made by the LinuxSandboxedStrategy.
   */
  private ImmutableSetMultimap<Path, Path> asserts(List<String> asserts) {
    ImmutableSetMultimap.Builder<Path, Path> pathifiedAsserts = ImmutableSetMultimap.builder();
    for (String fileName : asserts) {
      Path inputPath = workingDir.getRelative(fileName);
      pathifiedAsserts.put(inputPath, getSandboxPath(inputPath));
    }
    return pathifiedAsserts.build();
  }

  private ImmutableSetMultimap<String, String> userFriendlyAsserts(List<String> asserts) {
    return userFriendlyMap(asserts(asserts));
  }

  @Test
  public void resolvesRelativeFileToFileSymlinkInSameDir() throws IOException {
    Multimap<String, String> testFiles = LinkedHashMultimap.create();
    testFiles.put("symlink.txt", "goal.txt");
    testFiles.put("goal.txt", "");

    List<String> assertMounts = new ArrayList<>();
    assertMounts.add("symlink.txt");
    assertMounts.add("goal.txt");

    assertThat(userFriendlyMounts(testFiles)).containsExactly(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void resolvesRelativeFileToFileSymlinkInSubDir() throws IOException {
    Multimap<String, String> testFiles =
        ImmutableSetMultimap.of(
            "symlink.txt", "x/goal.txt",
            "x/goal.txt", "");

    List<String> assertMounts = ImmutableList.of("symlink.txt", "x/goal.txt");
    assertThat(userFriendlyMounts(testFiles)).containsExactly(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void resolvesRelativeFileToFileSymlinkInParentDir() throws IOException {
    Multimap<String, String> testFiles =
        ImmutableSetMultimap.of(
            "x/symlink.txt", "../goal.txt",
            "goal.txt", "");

    List<String> assertMounts = ImmutableList.of("x/symlink.txt", "goal.txt");

    assertThat(userFriendlyMounts(testFiles)).containsExactly(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void recursesSubDirs() throws IOException {
    ImmutableList<String> inputFile = ImmutableList.of("a/b");

    Multimap<String, String> testFiles =
        ImmutableSetMultimap.of(
            "a/b/x.txt", "",
            "a/b/y.txt", "z.txt",
            "a/b/z.txt", "");

    List<String> assertMounts = ImmutableList.of("a/b/x.txt", "a/b/y.txt", "a/b/z.txt");

    assertThat(userFriendlyMounts(testFiles, inputFile))
        .containsExactly(userFriendlyAsserts(assertMounts));
  }

  /**
   * Test that the algorithm correctly identifies and refuses symlink loops.
   */
  @Test
  public void catchesSymlinkLoop() throws IOException {
    try {
      mounts(
          ImmutableSetMultimap.of(
              "a", "b",
              "b", "a"));
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessage(
              String.format(
                  "%s (Too many levels of symbolic links)",
                  workingDir.getRelative("a").getPathString()));
    }
  }

  /**
   * Test that the algorithm correctly detects and refuses symlinks whose subcomponents are not all
   * directories (e.g. "a -> dir/file/file").
   */
  @Test
  public void catchesIllegalSymlink() throws IOException {
    try {
      mounts(
          ImmutableSetMultimap.of(
              "b", "a/c",
              "a", ""));
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessage(
              String.format("%s (Not a directory)", workingDir.getRelative("a/c").getPathString()));
    }
  }
}
