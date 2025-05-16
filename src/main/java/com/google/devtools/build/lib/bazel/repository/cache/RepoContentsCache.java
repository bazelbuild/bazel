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
import com.google.devtools.build.lib.util.FileSystemLock;
import com.google.devtools.build.lib.util.FileSystemLock.LockMode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import javax.annotation.Nullable;

/** A cache directory that stores the contents of fetched repos across different workspaces. */
public class RepoContentsCache {
  public static final String RECORDED_INPUTS_SUFFIX = ".recorded_inputs";

  @Nullable private Path path;

  // TODO: wyv@ - implement garbage collection

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
  }

  /** Returns the list of candidate repos for the given predeclared input hash. */
  public ImmutableList<CandidateRepo> getCandidateRepos(String predeclaredInputHash) {
    Preconditions.checkState(path != null);
    Path entryDir = path.getRelative(predeclaredInputHash);
    try {
      return entryDir.getDirectoryEntries().stream()
          .filter(path -> path.getBaseName().endsWith(RECORDED_INPUTS_SUFFIX))
          // We're just sorting for consistency here. (Note that "10.recorded_inputs" would sort
          // before "1.recorded_inputs".) If necessary, we could use some sort of heuristics here
          // to speed up the subsequent up-to-date-ness checking.
          .sorted(Comparator.comparing(Path::getBaseName))
          .map(CandidateRepo::fromRecordedInputsFile)
          .collect(toImmutableList());
    } catch (IOException e) {
      // This should only happen if `entryDir` doesn't exist yet or is not a directory. Either way,
      // don't outright fail; just treat it as if the cache is empty.
      return ImmutableList.of();
    }
  }

  /** Moves a freshly fetched repo into the contents cache. */
  public void moveToCache(
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
    cacheRepoDir.createDirectoryAndParents();
    // Move the fetched marker file to a temp location, so that if following operations fail, both
    // the fetched repo and the cache locations are considered out-of-date.
    Path temporaryMarker = entryDir.getChild(counter + ".temp_recorded_inputs");
    FileSystemUtils.moveFile(fetchedRepoMarkerFile, temporaryMarker);
    // Now perform the move, and afterwards, restore the marker file.
    try {
      fetchedRepoDir.renameTo(cacheRepoDir);
    } catch (IOException e) {
      FileSystemUtils.moveTreesBelow(fetchedRepoDir, cacheRepoDir);
    }
    temporaryMarker.renameTo(cacheRecordedInputsFile);
    // Set up a symlink at the original fetched repo dir path.
    fetchedRepoDir.deleteTree();
    FileSystemUtils.ensureSymbolicLink(fetchedRepoDir, cacheRepoDir);
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
}
