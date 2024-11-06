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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryContentInfo;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** TargetPatternEvaluator tests that require a custom filesystem. */
@RunWith(TestParameterInjector.class)
public class TargetPatternEvaluatorIOTest extends AbstractTargetPatternEvaluatorTest {
  private static class Transformer {
    @SuppressWarnings("unused")
    @Nullable
    public FileStatus stat(FileStatus stat, PathFragment path, boolean followSymlinks)
        throws IOException {
      return stat;
    }

    @SuppressWarnings("unused")
    @Nullable
    public Collection<Dirent> readdir(
        Collection<Dirent> readdir, PathFragment path, boolean followSymlinks) throws IOException {
      return readdir;
    }
  }

  private Transformer transformer = new Transformer();

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(DigestHashFunction.SHA256) {
      @Override
      public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
        FileStatus defaultResult = super.stat(path, followSymlinks);
        return transformer.stat(defaultResult, path, followSymlinks);
      }

      @Nullable
      @Override
      public FileStatus statIfFound(PathFragment path, boolean followSymlinks) {
        return statNullable(path, followSymlinks);
      }

      @Nullable
      @Override
      public FileStatus statNullable(PathFragment path, boolean followSymlinks) {
        FileStatus defaultResult = super.statNullable(path, followSymlinks);
        try {
          return transformer.stat(defaultResult, path, followSymlinks);
        } catch (IOException e) {
          return null;
        }
      }

      @Override
      protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
          throws IOException {
        Collection<Dirent> defaultResult = super.readdir(path, followSymlinks);
        return transformer.readdir(defaultResult, path, followSymlinks);
      }
    };
  }

  /**
   * Tests that a child with an inconsistent stat (first a directory, then not) is handled properly.
   * Even keep-going mode aborts eagerly in the face of inconsistent stats.
   */
  @Test
  public void testBadStat(@TestParameter boolean keepGoing) throws Exception {
    reporter.removeHandler(failFastHandler);
    // Given a package, "parent",
    Path parent = scratch.file("parent/BUILD", "sh_library(name = 'parent')").getParentDirectory();
    // And a child, "badstat",
    parent.getRelative("badstat").createDirectoryAndParents();

    // Such that badstat first reports that it is a directory, and then reports that it isn't,
    this.transformer = createInconsistentFileStateTransformer("parent/badstat");

    TargetParsingException e =
        assertThrows(
            TargetParsingException.class,
            () ->
                parseTargetPatternList(
                    parser, reporter, ImmutableList.of("//parent/..."), keepGoing));
    assertThat(e).hasMessageThat().contains("Inconsistent filesystem operations");
    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(FailureDetails.PackageLoading.Code.TRANSIENT_INCONSISTENT_FILESYSTEM_ERROR);
  }

  /**
   * Tests that a child with an inconsistent stat (first a directory, then not) is handled properly
   * when given a path-as-target. Even keep-going mode aborts eagerly in the face of inconsistent
   * stats.
   */
  @Test
  public void testBadStatPathAsTarget(@TestParameter boolean keepGoing) throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("parent/BUILD", "sh_library(name = 'parent')").getParentDirectory();
    delegatingSyscallCache.setDelegate(TestUtils.makeDisappearingFileCache("parent/BUILD"));
    TargetParsingException e =
        assertThrows(
            TargetParsingException.class,
            () -> parseTargetPatternList(parser, reporter, ImmutableList.of("parent"), keepGoing));
    assertThat(e).hasMessageThat().contains("Inconsistent filesystem operations");
    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(FailureDetails.PackageLoading.Code.TRANSIENT_INCONSISTENT_FILESYSTEM_ERROR);
  }

  /**
   * Tests that a package subdirectory that throws an IOException when it is listed via readdir does
   * not prevent evaluation of the remaining packages beneath a directory and the return of a
   * partial result.
   */
  @Test
  public void testBadReaddirKeepGoing() throws Exception {
    reporter.removeHandler(failFastHandler);
    // Given a package, "parent",
    Path parent = scratch.file("parent/BUILD", "filegroup(name = 'parent')").getParentDirectory();
    // And a child, "badstat",
    parent.getRelative("badstat").createDirectoryAndParents();

    // Such that badstat reports that it is a directory, but throws an error when its Dirents are
    // collected,
    this.transformer = createBadDirectoryListingTransformer("parent/badstat");

    // When we find all the targets beneath parent in keep_going mode, we get the valid target
    // parent:parent, even though processing badstat threw an IOException,
    assertThat(parseListKeepGoing("//parent/...").getFirst())
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
      public FileStatus stat(final FileStatus stat, PathFragment path, boolean followSymlinks) {
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
            public long getSize() {
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
      public Collection<Dirent> readdir(
          Collection<Dirent> readdir, PathFragment path, boolean followSymlinks)
          throws IOException {
        if (path.getPathString().endsWith(badPathSuffix)) {
          throw new IOException("Path ended in " + badPathSuffix + ", so readdir failed.");
        }
        return readdir;
      }
    };
  }
}
