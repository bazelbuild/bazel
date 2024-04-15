// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.producers;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionUniquenessFunction;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.producers.GlobComputationProducer.GlobDetail;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.ArrayList;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * {@link PatternWithWildcardProducer} is a sub-{@link StateMachine} created by {@link
 * FragmentProducer}. It handles glob pattern fragment which contains wildcard characters ({@code *}
 * or {@code **}).
 *
 * <p>Since wildcard is present, all dirents can be a possible pattern fragment match. So we need to
 * query the {@link DirectoryListingValue} and match all {@link Dirent}s to the glob pattern
 * fragment.
 *
 * <p>Handling symlink dirents requires special consideration. We query {@link FileValue}s for all
 * symlink dirents in a batch. The results are put in the {@link #symlinks} container. The {@link
 * #processSymlinks} method is invoked only once to handle all symlinks.
 *
 * <p>All matching dirents are handled by creating the {@link DirectoryDirentProducer}s for each one
 * of them.
 */
final class PatternWithWildcardProducer
    implements StateMachine, Consumer<SkyValue>, SymlinkProducer.ResultSink {

  // -------------------- Input --------------------
  private final GlobDetail globDetail;

  /** The {@link PathFragment} of the directory prefixed by the package fragments. */
  private final PathFragment base;

  private final int fragmentIndex;

  // -------------------- Internal State --------------------
  private DirectoryListingValue directoryListingValue = null;

  /** Holds both symlink path and target path for all symlink type dirents. */
  private ArrayList<Pair<FileValue.Key, FileValue>> symlinks = null;

  private int symlinksCount = 0;
  @Nullable private final Set<Pair<PathFragment, Integer>> visitedGlobSubTasks;

  // -------------------- Output --------------------
  private final FragmentProducer.ResultSink resultSink;

  PatternWithWildcardProducer(
      GlobDetail globDetail,
      PathFragment base,
      int fragmentIndex,
      FragmentProducer.ResultSink resultSink,
      @Nullable Set<Pair<PathFragment, Integer>> visitedGlobSubTasks) {
    this.globDetail = globDetail;
    this.base = base;
    this.fragmentIndex = fragmentIndex;
    this.resultSink = resultSink;
    this.visitedGlobSubTasks = visitedGlobSubTasks;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(
        DirectoryListingValue.key(RootedPath.toRootedPath(globDetail.packageRoot(), base)),
        (Consumer<SkyValue>) this);
    return this::processDirectoryListingValue;
  }

  @Override
  public void accept(SkyValue skyValue) {
    directoryListingValue = (DirectoryListingValue) skyValue;
  }

  private StateMachine processDirectoryListingValue(Tasks tasks) {
    Preconditions.checkNotNull(directoryListingValue);
    String patternFragment = globDetail.patternFragments().get(fragmentIndex);
    for (Dirent dirent : directoryListingValue.getDirents()) {
      if (dirent.getType() == Dirent.Type.UNKNOWN) {
        continue;
      }

      String direntName = dirent.getName();

      if (!UnixGlob.matches(patternFragment, direntName, globDetail.regexPatternCache())) {
        continue;
      }

      // At this point, we know that the dirent matches current pattern fragment but we don't yet
      // know if it belongs in the result. Delay creating the full PathFragment until we actually
      // need it.

      if (dirent.getType() == Dirent.Type.SYMLINK) {
        tasks.enqueue(
            new SymlinkProducer(
                FileValue.key(
                    RootedPath.toRootedPath(globDetail.packageRoot(), base.getChild(direntName))),
                (SymlinkProducer.ResultSink) this));
        ++symlinksCount;
      } else if (dirent.getType() == Dirent.Type.DIRECTORY) {
        tasks.enqueue(
            new DirectoryDirentProducer(
                globDetail,
                base.getChild(direntName),
                fragmentIndex,
                resultSink,
                visitedGlobSubTasks));
      } else {
        if (FragmentProducer.shouldAddFileMatchingToResult(fragmentIndex, globDetail)) {
          resultSink.acceptPathFragmentWithPackageFragment(base.getChild(direntName));
        }
      }
    }

    if (symlinksCount > 0) {
      // When there are multiple symlinks under the sub-directory, we want to put all symlink
      // `FileValue`s into a container and handle all of them in a single `processSymlinks`
      // execution.
      // At this point, we already knew number symlinks under the sub-directory, so allocate the
      // same size for the symlinks array in advance.
      symlinks = Lists.newArrayListWithCapacity(symlinksCount);
      return this::processSymlinks;
    }
    return DONE;
  }

  @Override
  public void acceptSymlinkFileValue(FileValue symlinkValue, FileValue.Key symlinkKey) {
    symlinks.add(Pair.of(symlinkKey, symlinkValue));
  }

  @Override
  public void acceptInconsistentFilesystemException(InconsistentFilesystemException exception) {
    resultSink.acceptGlobError(GlobError.of(exception));
  }

  private StateMachine processSymlinks(Tasks tasks) {
    if (symlinks.isEmpty() || symlinks.size() < symlinksCount) {
      // It is possible that some symlinks cannot be accepted due to inconsistent filesystem error.
      // In this case, since the `InconsistentFilesystemException` is accepted and glob function
      // computation will error out, it is unnecessary to proceed.
      return DONE;
    }

    for (Pair<FileValue.Key, FileValue> symlink : symlinks) {
      FileValue.Key symlinkKey = symlink.first;
      FileValue symlinkValue = symlink.second;

      if (!symlinkValue.exists()) {
        // Tolerate when the symlink is pointing to a non-existing path.
        continue;
      }

      // This check is more strict than necessary: we raise an error if globbing traverses into
      // a directory for any reason, even though it's only necessary if that reason was the
      // resolution of a recursive glob ("**"). Fixing this would require plumbing the ancestor
      // symlink information through DirectoryListingValue.
      if (symlinkValue.isDirectory()
          && symlinkValue.unboundedAncestorSymlinkExpansionChain() != null) {
        tasks.lookUp(
            FileSymlinkInfiniteExpansionUniquenessFunction.key(
                symlinkValue.unboundedAncestorSymlinkExpansionChain()),
            v -> {});
        resultSink.acceptGlobError(
            GlobError.of(
                new FileSymlinkInfiniteExpansionException(
                    symlinkValue.pathToUnboundedAncestorSymlinkExpansionChain(),
                    symlinkValue.unboundedAncestorSymlinkExpansionChain())));
        return DONE;
      }

      // Use the symlink path instead of the target path.
      PathFragment direntPath = symlinkKey.argument().getRootRelativePath();
      if (symlinkValue.isDirectory()) {
        tasks.enqueue(
            new DirectoryDirentProducer(
                globDetail, direntPath, fragmentIndex, resultSink, visitedGlobSubTasks));
      } else {
        if (FragmentProducer.shouldAddFileMatchingToResult(fragmentIndex, globDetail)) {
          resultSink.acceptPathFragmentWithPackageFragment(direntPath);
        }
      }
    }

    // After all symlinks of dirents are processed, `symlinks` array list is useless and should be
    // garbage collected.
    symlinks = null;
    return DONE;
  }
}
