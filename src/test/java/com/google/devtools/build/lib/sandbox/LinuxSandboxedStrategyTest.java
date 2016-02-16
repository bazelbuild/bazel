// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
public class LinuxSandboxedStrategyTest extends LinuxSandboxedStrategyTestCase {
  /**
   * Strips the working directory (which can be very long) from the file names in the input map, to
   * make assertion failures easier to read.
   */
  private ImmutableMap<String, String> userFriendlyMap(Map<Path, Path> input) {
    ImmutableMap.Builder<String, String> userFriendlyMap = ImmutableMap.builder();
    for (Entry<Path, Path> entry : input.entrySet()) {
      String key = entry.getKey().getPathString().replace(workspaceDir.getPathString(), "");
      String value = entry.getValue().getPathString().replace(workspaceDir.getPathString(), "");
      userFriendlyMap.put(key, value);
    }
    return userFriendlyMap.build();
  }

  /**
   * Takes a map of file specifications, creates the necessary files / symlinks / dirs,
   * mounts files listed in customMount at their canonical location in the sandbox and returns the
   * output of {@code LinuxSandboxedStrategy#fixMounts} for it.
   */
  private ImmutableMap<String, String> userFriendlyMounts(
      Map<String, String> linksAndFiles, List<String> customMounts) throws Exception {
    return userFriendlyMap(mounts(linksAndFiles, customMounts));
  }

  private ImmutableMap<Path, Path> mounts(
      Map<String, String> linksAndFiles, List<String> customMounts) throws Exception {
    createTreeStructure(linksAndFiles);

    ImmutableMap.Builder<Path, Path> mounts = ImmutableMap.builder();
    for (String customMount : customMounts) {
      Path customMountPath = workspaceDir.getRelative(customMount);
      mounts.put(customMountPath, customMountPath);
    }
    return ImmutableMap.copyOf(LinuxSandboxedStrategy.finalizeMounts(mounts.build()));
  }

  /**
   * Takes a map of file specifications, creates the necessary files / symlinks / dirs,
   * mounts the first file of the specification at its canonical location in the sandbox and returns
   * the output of {@code LinuxSandboxedStrategy#fixMounts} for it.
   */
  private Map<String, String> userFriendlyMounts(Map<String, String> linksAndFiles)
      throws Exception {
    return userFriendlyMap(mounts(linksAndFiles));
  }

  private Map<Path, Path> mounts(Map<String, String> linksAndFiles) throws Exception {
    return mounts(
        linksAndFiles, ImmutableList.of(Iterables.getFirst(linksAndFiles.keySet(), null)));
  }

  /**
   * Returns a map of mount entries for a list files, which can be used to assert that all
   * expected mounts have been made by the LinuxSandboxedStrategy.
   */
  private ImmutableMap<String, String> userFriendlyAsserts(List<String> asserts) {
    return userFriendlyMap(asserts(asserts));
  }

  private ImmutableMap<Path, Path> asserts(List<String> asserts) {
    ImmutableMap.Builder<Path, Path> pathifiedAsserts = ImmutableMap.builder();
    for (String fileName : asserts) {
      Path inputPath = workspaceDir.getRelative(fileName);
      pathifiedAsserts.put(inputPath, inputPath);
    }
    return pathifiedAsserts.build();
  }

  private void createTreeStructure(Map<String, String> linksAndFiles) throws Exception {
    for (Entry<String, String> entry : linksAndFiles.entrySet()) {
      Path filePath = workspaceDir.getRelative(entry.getKey());
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

  @Test
  public void testResolvesRelativeFileToFileSymlinkInSameDir() throws Exception {
    Map<String, String> testFiles = new LinkedHashMap<>();
    testFiles.put("symlink.txt", "goal.txt");
    testFiles.put("goal.txt", "");

    List<String> assertMounts = new ArrayList<>();
    assertMounts.add("symlink.txt");
    assertMounts.add("goal.txt");

    assertThat(userFriendlyMounts(testFiles)).isEqualTo(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void testResolvesRelativeFileToFileSymlinkInSubDir() throws Exception {
    Map<String, String> testFiles =
        ImmutableMap.of(
            "symlink.txt", "x/goal.txt",
            "x/goal.txt", "");

    List<String> assertMounts = ImmutableList.of("symlink.txt", "x/goal.txt");
    assertThat(userFriendlyMounts(testFiles)).isEqualTo(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void testResolvesRelativeFileToFileSymlinkInParentDir() throws Exception {
    Map<String, String> testFiles =
        ImmutableMap.of(
            "x/symlink.txt", "../goal.txt",
            "goal.txt", "");

    List<String> assertMounts = ImmutableList.of("x/symlink.txt", "goal.txt");

    assertThat(userFriendlyMounts(testFiles)).isEqualTo(userFriendlyAsserts(assertMounts));
  }

  @Test
  public void testRecursesSubDirs() throws Exception {
    ImmutableList<String> inputFile = ImmutableList.of("a/b");

    Map<String, String> testFiles =
        ImmutableMap.of(
            "a/b/x.txt", "",
            "a/b/y.txt", "z.txt",
            "a/b/z.txt", "");

    List<String> assertMounts = ImmutableList.of("a/b/x.txt", "a/b/y.txt", "a/b/z.txt");

    assertThat(userFriendlyMounts(testFiles, inputFile))
        .isEqualTo(userFriendlyAsserts(assertMounts));
  }

  /**
   * Test that the algorithm correctly identifies and refuses symlink loops.
   */
  @Test
  public void testCatchesSymlinkLoop() throws Exception {
    try {
      mounts(
          ImmutableMap.of(
              "a", "b",
              "b", "a"));
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessage(
              String.format(
                  "%s (Too many levels of symbolic links)",
                  workspaceDir.getRelative("a").getPathString()));
    }
  }

  /**
   * Test that the algorithm correctly detects and refuses symlinks whose subcomponents are not all
   * directories (e.g. "a -> dir/file/file").
   */
  @Test
  public void testCatchesIllegalSymlink() throws Exception {
    try {
      mounts(
          ImmutableMap.of(
              "b", "a/c",
              "a", ""));
      fail();
    } catch (IOException e) {
      assertThat(e)
          .hasMessage(
              String.format(
                  "%s (Not a directory)", workspaceDir.getRelative("a/c").getPathString()));
    }
  }

  @Test
  public void testParseManifestFile() throws Exception {
    Path targetDir = workspaceDir.getRelative("runfiles");
    targetDir.createDirectory();

    Path testFile = workspaceDir.getRelative("testfile");
    FileSystemUtils.createEmptyFile(testFile);

    Path manifestFile = workspaceDir.getRelative("MANIFEST");
    FileSystemUtils.writeContent(
        manifestFile,
        Charset.defaultCharset(),
        String.format("x/testfile %s\nx/emptyfile \n", testFile.getPathString()));

    Map mounts =
        LinuxSandboxedStrategy.parseManifestFile(targetDir, manifestFile.getPathFile(), false, "");

    assertThat(userFriendlyMap(mounts))
        .isEqualTo(
            userFriendlyMap(
                ImmutableMap.of(
                    fileSystem.getPath("/runfiles/x/testfile"),
                    testFile,
                    fileSystem.getPath("/runfiles/x/emptyfile"),
                    fileSystem.getPath("/dev/null"))));
  }

  @Test
  public void testParseFilesetManifestFile() throws Exception {
    Path targetDir = workspaceDir.getRelative("fileset");
    targetDir.createDirectory();

    Path testFile = workspaceDir.getRelative("testfile");
    FileSystemUtils.createEmptyFile(testFile);

    Path manifestFile = workspaceDir.getRelative("MANIFEST");
    FileSystemUtils.writeContent(
        manifestFile,
        Charset.defaultCharset(),
        String.format("workspace/x/testfile %s\n0\n", testFile.getPathString()));

    Map mounts =
        LinuxSandboxedStrategy.parseManifestFile(
            targetDir, manifestFile.getPathFile(), true, "workspace");

    assertThat(userFriendlyMap(mounts))
        .isEqualTo(
            userFriendlyMap(ImmutableMap.of(fileSystem.getPath("/fileset/x/testfile"), testFile)));
  }
}
