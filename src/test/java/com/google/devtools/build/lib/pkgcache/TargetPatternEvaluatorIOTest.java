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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableSet;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryContentInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** TargetPatternEvaluator tests that require a custom filesystem. */
@RunWith(JUnit4.class)
public class TargetPatternEvaluatorIOTest extends AbstractTargetPatternEvaluatorTest {
  private static final String FS_ROOT = "/fsg";

  private static class Transformer {
    @SuppressWarnings("unused")
    @Nullable
    public FileStatus stat(FileStatus stat, Path path, boolean followSymlinks) throws IOException {
      return stat;
    }

    @SuppressWarnings("unused")
    @Nullable
    public Collection<Dirent> readdir(Collection<Dirent> readdir, Path path, boolean followSymlinks)
        throws IOException {
      return readdir;
    }
  }

  private Transformer transformer = new Transformer();

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance()) {
      @Override
      public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
        FileStatus defaultResult = super.stat(path, followSymlinks);
        return transformer.stat(defaultResult, path, followSymlinks);
      }

      @Nullable
      @Override
      public FileStatus statIfFound(Path path, boolean followSymlinks) {
        return statNullable(path, followSymlinks);
      }

      @Nullable
      @Override
      public FileStatus statNullable(Path path, boolean followSymlinks) {
        FileStatus defaultResult = super.statNullable(path, followSymlinks);
        try {
          return transformer.stat(defaultResult, path, followSymlinks);
        } catch (IOException e) {
          return null;
        }
      }

      @Override
      protected Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
        Collection<Dirent> defaultResult = super.readdir(path, followSymlinks);
        return transformer.readdir(defaultResult, path, followSymlinks);
      }
    };
  }

  /**
   * Test that a child with an inconsistent stat (first a directory, then not) does not prevent
   * evaluation of the remaining packages beneath a directory and the return of a partial result.
   */
  @Test
  public void testBadStatKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    getSkyframeExecutor().turnOffSyscallCacheForTesting();
    // Given a package, "parent",
    Path parent = scratch.file("parent/BUILD", "sh_library(name = 'parent')").getParentDirectory();
    // And a child, "badstat",
    FileSystemUtils.createDirectoryAndParents(parent.getRelative("badstat"));

    // Such that badstat first reports that it is a directory, and then reports that it isn't,
    this.transformer = createInconsistentFileStateTransformer("parent/badstat");

    // When we find all the targets beneath parent in keep_going mode, we get the valid target
    // parent:parent, even though processing badstat threw an InconsistentFilesystemException,
    Truth.assertThat(parseListKeepGoing("//parent/...").getFirst())
        .containsExactlyElementsIn(labels("//parent:parent"));

    // And the TargetPatternEvaluator reported the expected ERROR event to the handler.
    assertContainsEvent(
        "Failed to get information about path, for parent/badstat, skipping: Inconsistent "
            + "filesystem operations",
        ImmutableSet.of(EventKind.ERROR));
  }

  /**
   * Test that a package subdirectory that throws an IOException when it is listed via readdir
   * does not prevent evaluation of the remaining packages beneath a directory and the return of
   * a partial result.
   */
  @Test
  public void testBadReaddirKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    skyframeExecutor.turnOffSyscallCacheForTesting();
    // Given a package, "parent",
    Path parent = scratch.file("parent/BUILD", "sh_library(name = 'parent')").getParentDirectory();
    // And a child, "badstat",
    FileSystemUtils.createDirectoryAndParents(parent.getRelative("badstat"));

    // Such that badstat reports that it is a directory, but throws an error when its Dirents are
    // collected,
    this.transformer = createBadDirectoryListingTransformer("parent/badstat");

    // When we find all the targets beneath parent in keep_going mode, we get the valid target
    // parent:parent, even though processing badstat threw an IOException,
    Truth.assertThat(parseListKeepGoing("//parent/...").getFirst())
        .containsExactlyElementsIn(labels("//parent:parent"));

    // And the TargetPatternEvaluator reported the expected ERROR event to the handler.
    assertContainsEvent(
        "Failed to list directory contents, for parent/badstat, skipping: Path ended in "
            + "parent/badstat, so readdir failed",
        ImmutableSet.of(EventKind.ERROR));
  }

  private Transformer createInconsistentFileStateTransformer(final String badPathSuffix) {
    final AtomicBoolean isDirectory = new AtomicBoolean(true);
    return new Transformer() {
      @Nullable
      @Override
      public FileStatus stat(final FileStatus stat, Path path, boolean followSymlinks)
          throws IOException {
        if (path.getPathString().endsWith(badPathSuffix)) {
          return new InMemoryContentInfo(BlazeClock.instance()) {
            @Override
            public boolean isDirectory() {
              // Trigger inconsistent filesystem exception.
              return isDirectory.getAndSet(false);
            }

            @Override
            public boolean isFile() {
              return stat.isFile();
            }

            @Override
            public boolean isSpecialFile() {
              return stat.isSpecialFile();
            }

            @Override
            public boolean isSymbolicLink() {
              return stat.isSymbolicLink();
            }

            @Override
            public long getSize()  {
              try {
                return stat.getSize();
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
            }

            @Override
            public synchronized long getLastModifiedTime() {
              try {
                return stat.getLastModifiedTime();
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
            }

            @Override
            public synchronized long getLastChangeTime() {
              try {
                return stat.getLastChangeTime();
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
            }

            @Override
            public long getNodeId() {
              try {
                return stat.getNodeId();
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
            }
          };
        }
        return stat;
      }
    };
  }

  private Transformer createBadDirectoryListingTransformer(final String badPathSuffix) {
    return new Transformer() {
      @Nullable
      @Override
      public Collection<Dirent> readdir(Collection<Dirent> readdir, Path path,
          boolean followSymlinks) throws IOException {
        if (path.getPathString().endsWith(badPathSuffix)) {
          throw new IOException("Path ended in " + badPathSuffix + ", so readdir failed.");
        }
        return readdir;
      }
    };
  }
}
