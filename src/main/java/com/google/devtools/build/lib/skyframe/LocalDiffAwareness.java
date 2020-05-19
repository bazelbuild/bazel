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

package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Set;

/**
 * File system watcher for local filesystems. It's able to provide a list of changed files between
 * two consecutive calls. On Linux, uses the standard Java WatchService, which uses 'inotify' and,
 * on OS X, uses {@link MacOSXFsEventsDiffAwareness}, which use FSEvents.
 *
 * <p>
 * This is an abstract class, specialized by {@link MacOSXFsEventsDiffAwareness} and
 * {@link WatchServiceDiffAwareness}.
 */
public abstract class LocalDiffAwareness implements DiffAwareness {
  /**
   * Option to enable / disable local diff awareness.
   */
  public static final class Options extends OptionsBase {
    @Option(
        name = "watchfs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "On Linux/macOS: If true, %{product} tries to use the operating system's file watch "
                + "service for local changes instead of scanning every file for a change. On "
                + "Windows: this flag currently is a non-op but can be enabled in conjunction "
                + "with --experimental_windows_watchfs.")
    public boolean watchFS;

    @Option(
        name = "experimental_windows_watchfs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "If true, experimental Windows support for --watchfs is enabled. Otherwise --watchfs"
                + "is a non-op on Windows. Make sure to also enable --watchfs.")
    public boolean windowsWatchFS;
  }

  /** Factory for creating {@link LocalDiffAwareness} instances. */
  public static class Factory implements DiffAwareness.Factory {
    private final ImmutableList<String> prefixBlacklist;

    /**
     * Creates a new factory; the file system watcher may not work on all file systems, particularly
     * for network file systems. The prefix blacklist can be used to blacklist known paths that
     * point to network file systems.
     */
    public Factory(ImmutableList<String> prefixBlacklist) {
      this.prefixBlacklist = prefixBlacklist;
    }

    @Override
    public DiffAwareness maybeCreate(Root pathEntry) {
      com.google.devtools.build.lib.vfs.Path resolvedPathEntry;
      try {
        resolvedPathEntry = pathEntry.asPath().resolveSymbolicLinks();
      } catch (IOException e) {
        return null;
      }
      PathFragment resolvedPathEntryFragment = resolvedPathEntry.asFragment();
      // There's no good way to automatically detect network file systems. We rely on a blacklist
      // for now (and maybe add a command-line option in the future?).
      for (String prefix : prefixBlacklist) {
        if (resolvedPathEntryFragment.startsWith(PathFragment.create(prefix))) {
          return null;
        }
      }
      // On OSX uses FsEvents due to https://bugs.openjdk.java.net/browse/JDK-7133447
      if (OS.getCurrent() == OS.DARWIN) {
        return new MacOSXFsEventsDiffAwareness(resolvedPathEntryFragment.toString());
      }

      return new WatchServiceDiffAwareness(resolvedPathEntryFragment.toString());
    }
  }

  /**
   * A view that results in any subsequent getDiff calls returning
   * {@link ModifiedFileSet#EVERYTHING_MODIFIED}. Use this if --watchFs is disabled.
   *
   * <p>The position is set to -2 in order for {@link #areInSequence} below to always return false
   * if this view is passed to it. Any negative number would work; we don't use -1 as the other
   * view may have a position of 0.
   */
  protected static final View EVERYTHING_MODIFIED =
      new SequentialView(/*owner=*/null, /*position=*/-2, ImmutableSet.<Path>of());

  public static boolean areInSequence(SequentialView oldView, SequentialView newView) {
    // Keep this in sync with the EVERYTHING_MODIFIED View above.
    return oldView.owner == newView.owner && (oldView.position + 1) == newView.position;
  }

  private int numGetCurrentViewCalls = 0;

  /** Root directory to watch. This is an absolute path. */
  protected final Path watchRootPath;

  protected LocalDiffAwareness(String watchRoot) {
    this.watchRootPath = FileSystems.getDefault().getPath(watchRoot);
  }

  /**
   * The WatchService is inherently sequential and side-effectful, so we enforce this by only
   * supporting {@link #getDiff} calls that happen to be sequential.
   */
  @VisibleForTesting
  static class SequentialView implements DiffAwareness.View {
    private final LocalDiffAwareness owner;
    private final int position;
    private final Set<Path> modifiedAbsolutePaths;

    public SequentialView(LocalDiffAwareness owner, int position, Set<Path> modifiedAbsolutePaths) {
      this.owner = owner;
      this.position = position;
      this.modifiedAbsolutePaths = modifiedAbsolutePaths;
    }

    @Override
    public String toString() {
      return String.format("SequentialView[owner=%s, position=%d, modifiedAbsolutePaths=%s]", owner,
          position, modifiedAbsolutePaths);
    }
  }

  /**
   * Returns true on any call before first call to {@link #newView}.
   */
  protected boolean isFirstCall() {
    return numGetCurrentViewCalls == 0;
  }

  /**
   * Create a new views using a list of modified absolute paths. This will increase the view
   * counter.
   */
  protected SequentialView newView(Set<Path> modifiedAbsolutePaths) {
    numGetCurrentViewCalls++;
    return new SequentialView(this, numGetCurrentViewCalls, modifiedAbsolutePaths);
  }

  @Override
  public ModifiedFileSet getDiff(View oldView, View newView)
      throws IncompatibleViewException, BrokenDiffAwarenessException {
    SequentialView oldSequentialView;
    SequentialView newSequentialView;
    try {
      oldSequentialView = (SequentialView) oldView;
      newSequentialView = (SequentialView) newView;
    } catch (ClassCastException e) {
      throw new IncompatibleViewException("Given views are not from LocalDiffAwareness");
    }
    if (!areInSequence(oldSequentialView, newSequentialView)) {
      return ModifiedFileSet.EVERYTHING_MODIFIED;
    }

    ModifiedFileSet.Builder resultBuilder = ModifiedFileSet.builder();
    for (Path modifiedPath : newSequentialView.modifiedAbsolutePaths) {
      if (!modifiedPath.startsWith(watchRootPath)) {
        throw new BrokenDiffAwarenessException(
            String.format("%s is not under %s", modifiedPath, watchRootPath));
      }
      PathFragment relativePath =
          PathFragment.create(watchRootPath.relativize(modifiedPath).toString());
      if (!relativePath.isEmpty()) {
        resultBuilder.modify(relativePath);
      }
    }
    return resultBuilder.build();
  }

  @Override
  public String name() {
    return "local";
  }
}
