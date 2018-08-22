// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.framework;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.StringUtilities;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link PathUtils}. */
@RunWith(JUnit4.class)
public class PathUtilsTest {
  @Test
  public void testDeleteTree() throws IOException {
    Path directory = Files.createTempDirectory("test");

    Path subDir = subDir(directory, "subdir");
    write(subDir, "file1.txt", "Hello!");
    write(subDir, "file2.txt", "I am here!");

    Path innerDir = subDir(subDir, "inner");
    write(innerDir, "inner.txt", "Deep");

    subDir(subDir, "empty");

    PathUtils.deleteTree(directory);

    assertThat(Files.exists(directory)).isFalse();
  }

  @Test
  public void testCopyTree() throws IOException {
    Path source = Files.createTempDirectory("source");
    Path target = Files.createTempDirectory("target");

    try {
      Path subDir = subDir(source, "subdir");
      write(subDir, "file1.txt", "Hello!");
      write(subDir, "file2.txt", "I am here!");

      Path innerDir = subDir(subDir, "inner");
      write(innerDir, "inner.txt", "Deep");

      subDir(subDir, "empty");

      PathUtils.copyTree(source, target);

      Path targetSubdir = resolveDirectory(target, "subdir");
      Path targetInner = resolveDirectory(targetSubdir, "inner");
      resolveDirectory(targetSubdir, "empty");

      assertFileExists(targetSubdir, "file1.txt", "Hello!");
      assertFileExists(targetSubdir, "file2.txt", "I am here!");

      assertFileExists(targetInner, "inner.txt", "Deep");
    } finally {
      PathUtils.deleteTree(source);
      PathUtils.deleteTree(target);
    }
  }

  @Test
  public void testResolve() throws IOException {
    Path directory = Files.createTempDirectory("test");
    try {
      Path expected = directory.resolve("subdir").resolve("inner").resolve("file.txt");
      Path resolved = PathUtils.resolve(directory, "subdir", "inner", "file.txt");
      // can not use assertThat here, because Path implements Iterable and there is ambiguity
      // in overloaded methods resolution between assertThat(T) and assertThat(Iterable<T>)
      Assert.assertEquals(expected, resolved);
      assertThat(resolved.getFileName().toString()).isEqualTo("file.txt");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testCreateFile() throws IOException {
    Path directory = Files.createTempDirectory("test");
    try {
      Path resolved = PathUtils.resolve(directory, "a", "b", "c");
      assertThat(Files.exists(resolved)).isFalse();

      PathUtils.createFile(resolved);

      assertThat(Files.exists(resolved)).isTrue();
      assertThat(Files.isRegularFile(resolved)).isTrue();
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testCreateFileChained() throws IOException {
    Path directory = Files.createTempDirectory("test");
    try {
      Path resolved = PathUtils.resolve(directory, "a", "b", "c");
      assertThat(Files.exists(resolved)).isFalse();

      Path created = PathUtils.createFile(directory, "a/b/c");
      assertThat(Files.exists(created)).isTrue();
      assertThat(Files.isRegularFile(created)).isTrue();
      // can not use assertThat here, because Path implements Iterable and there is ambiguity
      // in overloaded methods resolution between assertThat(T) and assertThat(Iterable<T>)
      Assert.assertEquals(resolved, created);
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testRewriteExistingFile() throws IOException {
    Path directory = Files.createTempDirectory("test");
    try {
      Path file = PathUtils.createFile(directory, "file.txt");
      assertThat(Files.exists(file)).isTrue();
      assertThat(Files.isRegularFile(file)).isTrue();

      PathUtils.writeFile(file, "Variant1");
      List<String> contents1 = PathUtils.readFile(file);
      assertThat(contents1).hasSize(1);
      assertThat(contents1.get(0)).isEqualTo("Variant1");

      PathUtils.writeFile(file, "Variant2");
      List<String> contents2 = PathUtils.readFile(file);
      assertThat(contents2).hasSize(1);
      assertThat(contents2.get(0)).isEqualTo("Variant2");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testReadWriteAppend() throws IOException {
    Path directory = Files.createTempDirectory("test");
    try {
      Path file = PathUtils.createFile(directory, "file.txt");
      PathUtils.writeFile(file, "line 1");
      assertThat(Files.exists(file)).isTrue();

      List<String> linesBefore = PathUtils.readFile(file);
      assertThat(linesBefore).hasSize(1);
      assertThat(linesBefore.get(0)).isEqualTo("line 1");

      PathUtils.append(file, "line 2");

      List<String> lines = PathUtils.readFile(file);
      assertThat(lines).hasSize(2);
      assertThat(lines.get(0)).isEqualTo("line 1");
      assertThat(lines.get(1)).isEqualTo("line 2");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  @Test
  public void testReplaceWithSymlinkContents() throws IOException {
    // do not run the test for windows
    if (!OS.isPosixCompatible()) {
      return;
    }

    Path directory = Files.createTempDirectory("test");
    try {
      Path target = PathUtils.createFile(directory, "source.txt");
      PathUtils.writeFile(target, "Target contents");
      Path link = directory.resolve("link.txt");
      Files.createSymbolicLink(link, target);

      assertThat(Files.exists(link)).isTrue();
      assertThat(Files.isSymbolicLink(link)).isTrue();

      PathUtils.replaceWithSymlinkContents(link);
      assertThat(Files.isSymbolicLink(link)).isFalse();

      List<String> lines = PathUtils.readFile(link);
      assertThat(lines).hasSize(1);
      assertThat(lines.get(0)).isEqualTo("Target contents");
    } finally {
      PathUtils.deleteTree(directory);
    }
  }

  private static void assertFileExists(Path directory, String fileName, String text)
      throws IOException {
    Path file = directory.resolve(fileName);
    assertThat(Files.exists(file)).isTrue();
    assertThat(Files.isRegularFile(file)).isTrue();
    assertThat(StringUtilities.joinLines(Files.readAllLines(file))).isEqualTo(text);
  }

  private static Path resolveDirectory(Path directory, String directoryName) {
    Path subDir = directory.resolve(directoryName);
    assertThat(Files.exists(subDir)).isTrue();
    assertThat(Files.isDirectory(subDir)).isTrue();
    return subDir;
  }

  private static Path subDir(Path directory, String directoryName) throws IOException {
    Path subDir = directory.resolve(directoryName);
    Files.createDirectories(subDir);
    subDir.toFile().deleteOnExit();
    return subDir;
  }

  private static void write(Path directory, String fileName, String text) throws IOException {
    Path file = directory.resolve(fileName);
    write(file, text);
  }

  private static void write(Path path, String text) throws IOException {
    assertThat(path.toFile().createNewFile()).isTrue();
    Files.write(path, Collections.singleton(text));
    path.toFile().deleteOnExit();
  }
}
