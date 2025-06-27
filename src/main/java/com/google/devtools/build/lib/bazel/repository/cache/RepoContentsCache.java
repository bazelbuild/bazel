// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.cache;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.server.IdleTask;
import com.google.devtools.build.lib.server.IdleTaskException;
import com.google.devtools.build.lib.util.FileSystemLock;
import com.google.devtools.build.lib.util.FileSystemLock.LockMode;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * A cache directory that stores the contents of fetched repos across different workspaces.
 *
 * <p>The repo contents cache is laid out in two layers. The first layer is a lookup by "predeclared
 * inputs hash", which is defined as the hash of all predeclared inputs of a repo (such as
 * transitive bzl digest, repo attrs, starlark semantics, etc). Each distinct predeclared inputs
 * hash is its own entry directory in the first layer.
 *
 * <p>Inside each entry directory are pairs of directories and files {@code <N, N.recorded_inputs>}
 * where {@code N} is an integer. The file {@code N.recorded_inputs} contains the recorded inputs
 * and their values of a cached repo, and the directory {@code N} contains the cached repo contents.
 * There is also a file named {@code counter} that stores the next available {@code N} for this
 * entry directory, and a file named {@code lock} to ensure exclusive access to the {@code counter}
 * file.
 *
 * <p>On a cache hit (that is, the predeclared inputs hash matches, and recorded inputs are
 * up-to-date), the recorded inputs file has its mtime updated. Cached repos whose recorded inputs
 * file is older than {@code --repo_contents_cache_gc_max_age} are garbage collected.
 */
public final class RepoContentsCache {
  public static final String RECORDED_INPUTS_SUFFIX = ".recorded_inputs";

  /**
   * The path to a "lock" file, relative to the root of the repo contents cache. While a shared lock
   * is held, no garbage collection should happen. While an exclusive lock is held, no reads should
   * happen.
   */
  public static final String LOCK_PATH = "gc_lock";

  /**
   * The path to a trash directory relative to the root of the repo contents cache.
   *
   * <p>Since deleting entire directories could take a bit of time, we create a trash directory
   * where we move the garbage directories to (which should be very fast). Then we can delete this
   * trash directory altogether at the end. This makes the GC process safe against being interrupted
   * in the middle (any undeleted trash will get deleted by the next GC). Also be sure to name this
   * trashDir something that couldn't ever be a predeclared inputs hash (starting with an underscore
   * should suffice).
   */
  public static final String TRASH_PATH = "_trash";

  @Nullable private Path path;
  @Nullable private FileSystemLock sharedLock;

  public void setPath(@Nullable Path path) {
    this.path = path;
  }

  @Nullable
  public Path getPath() {
    return path;
  }

  public boolean isEnabled() {
    return path != null;
  }

  /** A candidate repo in the repo contents cache for one predeclared input hash. */
  public record CandidateRepo(Path recordedInputsFile, Path contentsDir) {
    private static CandidateRepo fromRecordedInputsFile(Path recordedInputsFile) {
      String recordedInputsFileBaseName = recordedInputsFile.getBaseName();
      String contentsDirBaseName =
          recordedInputsFileBaseName.substring(
              0, recordedInputsFileBaseName.length() - RECORDED_INPUTS_SUFFIX.length());
      return new CandidateRepo(
          recordedInputsFile, recordedInputsFile.replaceName(contentsDirBaseName));
    }

    /** Updates the mtime of the recorded inputs file, to delay GC for this entry. */
    public void touch() {
      try {
        recordedInputsFile.setLastModifiedTime(Path.NOW_SENTINEL_TIME);
      } catch (IOException e) {
        // swallow the exception. it's not a huge deal.
      }
    }
  }

  /** Returns the list of candidate repos for the given predeclared input hash. */
  public ImmutableList<CandidateRepo> getCandidateRepos(String predeclaredInputHash) {
    Preconditions.checkState(path != null);
    Path entryDir = path.getRelative(predeclaredInputHash);
    try {
      return entryDir.getDirectoryEntries().stream()
          .filter(path -> path.getBaseName().endsWith(RECORDED_INPUTS_SUFFIX))
          // Prefer newer cache entries over older ones. They're more likely to be up-to-date; plus,
          // if a repo is force-fetched, we want to use the new repo instead of always being stuck
          // with the old one.
          // To "prefer newer cache entries", we sort the entry file names by length DESC and then
          // lexicographically DESC. This approximates sorting by converting to int and then DESC,
          // but is defensive against non-numerically named entries.
          .sorted(
              Comparator.comparing((Path path) -> path.getBaseName().length())
                  .thenComparing(Path::getBaseName)
                  .reversed())
          .map(CandidateRepo::fromRecordedInputsFile)
          .collect(toImmutableList());
    } catch (IOException e) {
      // This should only happen if `entryDir` doesn't exist yet or is not a directory. Either way,
      // don't outright fail; just treat it as if the cache is empty.
      return ImmutableList.of();
    }
  }

  private Path ensureTrashDir() throws IOException {
    Preconditions.checkState(path != null);
    Path trashDir = path.getChild(TRASH_PATH);
    trashDir.createDirectoryAndParents();
    return trashDir;
  }

  /**
   * Moves a freshly fetched repo into the contents cache.
   *
   * @return the repo dir in the contents cache.
   */
  public Path moveToCache(
      Path fetchedRepoDir, Path fetchedRepoMarkerFile, String predeclaredInputHash)
      throws IOException {
    Preconditions.checkState(path != null);

    Path entryDir = path.getRelative(predeclaredInputHash);
    if (!entryDir.isDirectory()) {
      entryDir.delete();
    }
    String counter = getNextCounterInDir(entryDir);
    Path cacheRecordedInputsFile = entryDir.getChild(counter + RECORDED_INPUTS_SUFFIX);
    Path cacheRepoDir = entryDir.getChild(counter);

    cacheRepoDir.deleteTree();
    cacheRepoDir.getParentDirectory().createDirectoryAndParents();
    // Move the fetched marker file to a temp location, so that if following operations fail, both
    // the fetched repo and the cache locations are considered out-of-date.
    Path temporaryMarker = ensureTrashDir().getChild(UUID.randomUUID().toString());
    FileSystemUtils.moveFile(fetchedRepoMarkerFile, temporaryMarker);
    // Now perform the move, and afterwards, restore the marker file.
    try {
      fetchedRepoDir.renameTo(cacheRepoDir);
    } catch (IOException e) {
      cacheRepoDir.createDirectoryAndParents();
      FileSystemUtils.moveTreesBelow(fetchedRepoDir, cacheRepoDir);
    }
    temporaryMarker.renameTo(cacheRecordedInputsFile);
    // Set up a symlink at the original fetched repo dir path.
    fetchedRepoDir.deleteTree();
    FileSystemUtils.ensureSymbolicLink(fetchedRepoDir, cacheRepoDir);
    return cacheRepoDir;
  }

  private static String getNextCounterInDir(Path entryDir) throws IOException {
    Path counterFile = entryDir.getRelative("counter");
    try (var lock = FileSystemLock.get(entryDir.getRelative("lock"), LockMode.EXCLUSIVE)) {
      int c = 0;
      if (counterFile.exists()) {
        try {
          c = Integer.parseInt(FileSystemUtils.readContent(counterFile, StandardCharsets.UTF_8));
        } catch (NumberFormatException e) {
          // ignored
        }
      }
      String counter = Integer.toString(c + 1);
      FileSystemUtils.writeContent(counterFile, StandardCharsets.UTF_8, counter);
      return counter;
    }
  }

  public void acquireSharedLock() throws IOException {
    Preconditions.checkState(path != null);
    Preconditions.checkState(sharedLock == null, "this process already has the shared lock");
    sharedLock = FileSystemLock.get(path.getRelative(LOCK_PATH), LockMode.SHARED);
  }

  public void releaseSharedLock() throws IOException {
    Preconditions.checkState(sharedLock != null);
    sharedLock.close();
    sharedLock = null;
  }

  /**
   * Creates a garbage collection {@link IdleTask} that deletes cached repos who are last accessed
   * more than {@code maxAge} ago, with an idle delay of {@code idleDelay}.
   */
  public IdleTask createGcIdleTask(Duration maxAge, Duration idleDelay) {
    Preconditions.checkState(path != null);
    return new IdleTask() {
      @Override
      public String displayName() {
        return "Repo contents cache garbage collection";
      }

      @Override
      public Duration delay() {
        return idleDelay;
      }

      @Override
      public void run() throws InterruptedException, IdleTaskException {
        try {
          Preconditions.checkState(path != null);
          // If we can't grab the lock, abort GC. Someone will come along later.
          try (var lock = FileSystemLock.tryGet(path.getRelative(LOCK_PATH), LockMode.EXCLUSIVE)) {
            runGc(maxAge);
          }
          // Empty the trash dir outside the lock. No one is reading from these files, so it should
          // be safe. At worst, multiple servers performing GC will try to delete the same files,
          // but whatever.
          path.getChild(TRASH_PATH).deleteTreesBelow();
        } catch (IOException e) {
          throw new IdleTaskException(e);
        }
      }
    };
  }

  private void runGc(Duration maxAge) throws InterruptedException, IOException {
    path.setLastModifiedTime(Path.NOW_SENTINEL_TIME);
    Instant cutoff = Instant.ofEpochMilli(path.getLastModifiedTime()).minus(maxAge);
    Path trashDir = ensureTrashDir();

    for (Dirent dirent : path.readdir(Symlinks.NOFOLLOW)) {
      if (dirent.getType() != Dirent.Type.DIRECTORY || dirent.getName().equals(TRASH_PATH)) {
        continue;
      }
      for (Path recordedInputsFile : path.getChild(dirent.getName()).getDirectoryEntries()) {
        if (!recordedInputsFile.getBaseName().endsWith(RECORDED_INPUTS_SUFFIX)) {
          continue;
        }
        if (Thread.interrupted()) {
          throw new InterruptedException();
        }

        if (Instant.ofEpochMilli(recordedInputsFile.getLastModifiedTime()).isBefore(cutoff)) {
          // Sorry buddy, you're out.
          recordedInputsFile.delete();
          var repoDir = CandidateRepo.fromRecordedInputsFile(recordedInputsFile).contentsDir;
          // Use a UUID to avoid clashes.
          repoDir.renameTo(trashDir.getChild(UUID.randomUUID().toString()));
        }
      }
    }
  }
}
