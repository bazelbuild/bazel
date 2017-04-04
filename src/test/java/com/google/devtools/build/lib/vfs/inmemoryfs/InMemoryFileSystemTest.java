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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.ScopeEscapableFileSystemTest;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tests for {@link InMemoryFileSystem}. Note that most tests are inherited
 * from {@link ScopeEscapableFileSystemTest} and ancestors. This specific
 * file focuses only on concurrency tests.
 *
 */
@RunWith(JUnit4.class)
public class InMemoryFileSystemTest extends ScopeEscapableFileSystemTest {

  @Override
  public FileSystem getFreshFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance(), SCOPE_ROOT);
  }

  @Override
  public void destroyFileSystem(FileSystem fileSystem) {
    // Nothing.
  }

  private static final int NUM_THREADS_FOR_CONCURRENCY_TESTS = 10;
  private static final String TEST_FILE_DATA = "data";

  /**
   * Writes the given data to the given file.
   */
  private static void writeToFile(Path path, String data) throws IOException {
    OutputStream out = path.getOutputStream();
    out.write(data.getBytes(Charset.defaultCharset()));
    out.close();
  }

  /**
   * Tests concurrent creation of a substantial tree hierarchy including
   * files, directories, symlinks, file contents, and permissions.
   */
  @Test
  public void testConcurrentTreeConstruction() throws Exception {
    final int NUM_TO_WRITE = 10000;
    final AtomicInteger baseSelector = new AtomicInteger();

    // 1) Define the intended path structure.
    class PathCreator extends TestThread {
      @Override
      public void runTest() throws Exception {
        Path base = testFS.getPath("/base" + baseSelector.getAndIncrement());
        base.createDirectory();

        for (int i = 0; i < NUM_TO_WRITE; i++) {
          Path subdir1 = base.getRelative("subdir1_" + i);
          subdir1.createDirectory();
          Path subdir2 = base.getRelative("subdir2_" + i);
          subdir2.createDirectory();

          Path file = base.getRelative("somefile" + i);
          writeToFile(file, TEST_FILE_DATA);

          subdir1.setReadable(true);
          subdir2.setReadable(false);
          file.setReadable(true);

          subdir1.setWritable(false);
          subdir2.setWritable(true);
          file.setWritable(false);

          subdir1.setExecutable(false);
          subdir2.setExecutable(true);
          file.setExecutable(false);

          subdir1.setLastModifiedTime(100);
          subdir2.setLastModifiedTime(200);
          file.setLastModifiedTime(300);

          Path symlink = base.getRelative("symlink" + i);
          symlink.createSymbolicLink(file);
        }
      }
    }

    // 2) Construct the tree.
    Collection<TestThread> threads =
        Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new PathCreator();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }

    // 3) Define the validation logic.
    class PathValidator extends TestThread {
      @Override
      public void runTest() throws Exception {
        Path base = testFS.getPath("/base" + baseSelector.getAndIncrement());
        assertTrue(base.exists());
        assertFalse(base.getRelative("notreal").exists());

        for (int i = 0; i < NUM_TO_WRITE; i++) {
          Path subdir1 = base.getRelative("subdir1_" + i);
          assertTrue(subdir1.exists());
          assertTrue(subdir1.isDirectory());
          assertTrue(subdir1.isReadable());
          assertFalse(subdir1.isWritable());
          assertFalse(subdir1.isExecutable());
          assertEquals(100, subdir1.getLastModifiedTime());

          Path subdir2 = base.getRelative("subdir2_" + i);
          assertTrue(subdir2.exists());
          assertTrue(subdir2.isDirectory());
          assertFalse(subdir2.isReadable());
          assertTrue(subdir2.isWritable());
          assertTrue(subdir2.isExecutable());
          assertEquals(200, subdir2.getLastModifiedTime());

          Path file = base.getRelative("somefile" + i);
          assertTrue(file.exists());
          assertTrue(file.isFile());
          assertTrue(file.isReadable());
          assertFalse(file.isWritable());
          assertFalse(file.isExecutable());
          assertEquals(300, file.getLastModifiedTime());
          BufferedReader reader = new BufferedReader(
              new InputStreamReader(file.getInputStream(), Charset.defaultCharset()));
          assertEquals(TEST_FILE_DATA, reader.readLine());
          assertNull(reader.readLine());

          Path symlink = base.getRelative("symlink" + i);
          assertTrue(symlink.exists());
          assertTrue(symlink.isSymbolicLink());
          assertEquals(file.asFragment(), symlink.readSymbolicLink());
        }
      }
    }

    // 4) Validate the results.
    baseSelector.set(0);
    threads = Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new PathValidator();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }
  }

  /**
   * Tests concurrent creation of many files, all within the same directory.
   */
  @Test
  public void testConcurrentDirectoryConstruction() throws Exception {
   final int NUM_TO_WRITE = 10000;
    final AtomicInteger baseSelector = new AtomicInteger();

    // 1) Define the intended path structure.
    class PathCreator extends TestThread {
      @Override
      public void runTest() throws Exception {
        final int threadId = baseSelector.getAndIncrement();
        Path base = testFS.getPath("/common_dir");
        base.createDirectory();

        for (int i = 0; i < NUM_TO_WRITE; i++) {
          Path file = base.getRelative("somefile_" + threadId + "_" + i);
          writeToFile(file, TEST_FILE_DATA);
          file.setReadable(i % 2 == 0);
          file.setWritable(i % 3 == 0);
          file.setExecutable(i % 4 == 0);
          file.setLastModifiedTime(i);
          Path symlink = base.getRelative("symlink_" + threadId + "_" + i);
          symlink.createSymbolicLink(file);
        }
      }
    }

    // 2) Create the files.
    Collection<TestThread> threads =
        Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new PathCreator();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }

    // 3) Define the validation logic.
    class PathValidator extends TestThread {
      @Override
      public void runTest() throws Exception {
        final int threadId = baseSelector.getAndIncrement();
        Path base = testFS.getPath("/common_dir");
        assertTrue(base.exists());

        for (int i = 0; i < NUM_TO_WRITE; i++) {
          Path file = base.getRelative("somefile_" + threadId + "_" + i);
          assertTrue(file.exists());
          assertTrue(file.isFile());
          assertEquals(i % 2 == 0, file.isReadable());
          assertEquals(i % 3 == 0, file.isWritable());
          assertEquals(i % 4 == 0, file.isExecutable());
          assertEquals(i, file.getLastModifiedTime());
          if (file.isReadable()) {
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(file.getInputStream(), Charset.defaultCharset()));
            assertEquals(TEST_FILE_DATA, reader.readLine());
            assertNull(reader.readLine());
          }

          Path symlink = base.getRelative("symlink_" + threadId + "_" + i);
          assertTrue(symlink.exists());
          assertTrue(symlink.isSymbolicLink());
          assertEquals(file.asFragment(), symlink.readSymbolicLink());
        }
      }
    }

    // 4) Validate the results.
    baseSelector.set(0);
    threads = Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new PathValidator();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }
  }

  /**
   * Tests concurrent file deletion.
   */
  @Test
  public void testConcurrentDeletion() throws Exception {
    final int NUM_TO_WRITE = 10000;
    final AtomicInteger baseSelector = new AtomicInteger();

    final Path base = testFS.getPath("/base");
    base.createDirectory();

    // 1) Create a bunch of files.
    for (int i = 0; i < NUM_TO_WRITE; i++) {
      writeToFile(base.getRelative("file" + i), TEST_FILE_DATA);
    }

    // 2) Define our deletion strategy.
    class FileDeleter extends TestThread {
      @Override
      public void runTest() throws Exception {
        for (int i = 0; i < NUM_TO_WRITE / NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
          int whichFile = baseSelector.getAndIncrement();
          Path file = base.getRelative("file" + whichFile);
          if (whichFile % 25 != 0) {
            assertTrue(file.delete());
          } else {
            // Throw another concurrent access point into the mix.
            file.setExecutable(whichFile % 2 == 0);
          }
          assertFalse(base.getRelative("doesnotexist" + whichFile).delete());
        }
      }
    }

    // 3) Delete some files.
    Collection<TestThread> threads =
        Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new FileDeleter();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }

    // 4) Check the results.
    for (int i = 0; i < NUM_TO_WRITE; i++) {
      Path file = base.getRelative("file" + i);
      if (i % 25 != 0) {
        assertFalse(file.exists());
      } else {
        assertTrue(file.exists());
        assertEquals(i % 2 == 0, file.isExecutable());
      }
    }
  }

  /**
   * Tests concurrent file renaming.
   */
  @Test
  public void testConcurrentRenaming() throws Exception {
    final int NUM_TO_WRITE = 10000;
    final AtomicInteger baseSelector = new AtomicInteger();

    final Path base = testFS.getPath("/base");
    base.createDirectory();

    // 1) Create a bunch of files.
    for (int i = 0; i < NUM_TO_WRITE; i++) {
      writeToFile(base.getRelative("file" + i), TEST_FILE_DATA);
    }

    // 2) Define our renaming strategy.
    class FileDeleter extends TestThread {
      @Override
      public void runTest() throws Exception {
        for (int i = 0; i < NUM_TO_WRITE / NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
          int whichFile = baseSelector.getAndIncrement();
          Path file = base.getRelative("file" + whichFile);
          if (whichFile % 25 != 0) {
            Path newName = base.getRelative("newname" + whichFile);
            file.renameTo(newName);
          } else {
            // Throw another concurrent access point into the mix.
            file.setExecutable(whichFile % 2 == 0);
          }
          assertFalse(base.getRelative("doesnotexist" + whichFile).delete());
        }
      }
    }

    // 3) Rename some files.
    Collection<TestThread> threads =
        Lists.newArrayListWithCapacity(NUM_THREADS_FOR_CONCURRENCY_TESTS);
    for (int i = 0; i < NUM_THREADS_FOR_CONCURRENCY_TESTS; i++) {
      TestThread thread = new FileDeleter();
      thread.start();
      threads.add(thread);
    }
    for (TestThread thread : threads) {
      thread.joinAndAssertState(0);
    }

    // 4) Check the results.
    for (int i = 0; i < NUM_TO_WRITE; i++) {
      Path file = base.getRelative("file" + i);
      if (i % 25 != 0) {
        assertFalse(file.exists());
        assertTrue(base.getRelative("newname" + i).exists());
      } else {
        assertTrue(file.exists());
        assertEquals(i % 2 == 0, file.isExecutable());
      }
    }
  }

  @Test
  public void testEloop() throws Exception {
    Path a = testFS.getPath("/a");
    Path b = testFS.getPath("/b");
    a.createSymbolicLink(PathFragment.create("b"));
    b.createSymbolicLink(PathFragment.create("a"));
    try {
      a.stat();
    } catch (IOException e) {
      assertThat(e).hasMessage("/a (Too many levels of symbolic links)");
    }
  }

  @Test
  public void testEloopSelf() throws Exception {
    Path a = testFS.getPath("/a");
    a.createSymbolicLink(PathFragment.create("a"));
    try {
      a.stat();
    } catch (IOException e) {
      assertThat(e).hasMessage("/a (Too many levels of symbolic links)");
    }
  }
}
