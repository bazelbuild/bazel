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

import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

/**
 * Tests for the {@link JavaIoFileSystem}. That file system by itself is not capable of creating
 * symlinks; use the unix one to create them, so that the test can check that the file system
 * handles their existence correctly.
 */
public class JavaIoFileSystemTest extends SymlinkAwareFileSystemTest {

  private ManualClock clock;

  @Override
  public FileSystem getFreshFileSystem(DigestHashFunction digestHashFunction) {
    clock = new ManualClock();
    return new JavaIoFileSystem(clock, digestHashFunction);
  }

  // Tests are inherited from the FileSystemTest

  // JavaIoFileSystem incorrectly throws a FileNotFoundException for all IO errors. This means that
  // statIfFound incorrectly suppresses those errors.
  @Override
  @Test
  public void testBadPermissionsThrowsExceptionOnStatIfFound() {}

  @Test
  public void testSetLastModifiedTime() throws Exception {
    Path file = xEmptyDirectory.getChild("new-file");
    FileSystemUtils.createEmptyFile(file);

    file.setLastModifiedTime(1000L);
    assertThat(file.getLastModifiedTime()).isEqualTo(1000L);
    file.setLastModifiedTime(0L);
    assertThat(file.getLastModifiedTime()).isEqualTo(0L);

    clock.advanceMillis(42000L);
    file.setLastModifiedTime(-1L);
    assertThat(file.getLastModifiedTime()).isEqualTo(42000L);
  }

  @Override
  protected boolean isHardLinked(Path a, Path b) throws IOException {
    return Files.readAttributes(
            Paths.get(a.toString()), BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS)
        .fileKey()
        .equals(
            Files.readAttributes(
                    Paths.get(b.toString()), BasicFileAttributes.class, LinkOption.NOFOLLOW_LINKS)
                .fileKey());
  }

  /**
   * This test has a large number of threads racing to create the same subdirectories.
   *
   * <p>We create N number of distinct directory trees, eg. the tree "0-0/0-1/0-2/0-3/0-4" followed
   * by the tree "1-0/1-1/1-2/1-3/1-4" etc. If there is race we should quickly get a deadlock.
   *
   * <p>A timeout of this test is likely because of a deadlock.
   */
  @Test
  public void testCreateDirectoriesThreadSafety() throws Exception {
    int threadCount = 200;
    int directoryCreationCount = 500; // We create this many sets of directories
    int subDirectoryCount = 5; // Each directory tree is this deep
    ListeningExecutorService executor =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(threadCount));
    List<ListenableFuture<IOException>> futures = new ArrayList<>();
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
      futures.add(
          executor.submit(
              () -> {
                try {
                  for (int loopi = 0; loopi < directoryCreationCount; ++loopi) {
                    List<Path> subDirs =
                        getSubDirectories(xEmptyDirectory, loopi, subDirectoryCount);
                    Path lastDir = Iterables.getLast(subDirs);
                    FileSystemUtils.createDirectoryAndParents(lastDir);
                  }
                } catch (IOException e) {
                  return e;
                }
                return null;
              }));
    }
    ListenableFuture<List<IOException>> all = Futures.allAsList(futures);
    // If the test times out here then there's likely to be a deadlock
    List<IOException> exceptions =
        all.get(TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS);
    Optional<IOException> error = exceptions.stream().filter(Objects::nonNull).findFirst();
    if (error.isPresent()) {
      throw error.get();
    }
  }

  private static List<Path> getSubDirectories(Path base, int loopi, int subDirectoryCount) {
    Path path = base;
    List<Path> subDirs = new ArrayList<>();
    for (int subDirIndex = 0; subDirIndex < subDirectoryCount; ++subDirIndex) {
      path = path.getChild(String.format("%d-%d", loopi, subDirIndex));
      subDirs.add(path);
    }
    return subDirs;
  }
}
