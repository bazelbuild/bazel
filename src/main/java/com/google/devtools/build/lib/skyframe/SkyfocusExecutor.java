// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.actions.FileStateValue.NONEXISTENT_FILE_STATE_NODE;
import static com.google.devtools.build.lib.skyframe.SkyfocusState.ActiveDirectoriesType.DERIVED;
import static com.google.devtools.build.lib.skyframe.SkyfocusState.ActiveDirectoriesType.USER_DEFINED;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.PathFragmentPrefixTrie;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.SkyframeFocuser.FocusResult;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** A class that prepares the active directories to run the core SkyframeFocuser algorithm. */
public class SkyfocusExecutor {

  private SkyfocusExecutor() {}

  /**
   * Prepares the active directories to run the core SkyframeFocuser algorithm.
   *
   * <p>This method will update the active directories if the user has requested new active
   * directories from the command line, or if the user has not requested new active directories,
   * automatically derive it using the source state.
   *
   * @return an optional of a SkyfocusState. If the value is present, the active directories has
   *     been updated.
   */
  public static Optional<SkyfocusState> prepareActiveDirectories(
      Collection<Label> topLevelTargetLabels,
      Optional<PathFragmentPrefixTrie> activeDirectoriesMatcher,
      InMemoryMemoizingEvaluator evaluator,
      SkyfocusState skyfocusState,
      PackageManager packageManager,
      PathPackageLocator pkgLocator,
      ExtendedEventHandler eventHandler) {
    Preconditions.checkState(
        !topLevelTargetLabels.isEmpty(),
        "Cannot prepare active directories without top level targets to focus on.");

    SkyfocusState.Builder newSkyfocusStateBuilder =
        skyfocusState.toBuilder()
            .focusedTargetLabels(
                ImmutableSet.<Label>builder()
                    // Persist previous focused labels.
                    .addAll(skyfocusState.focusedTargetLabels())
                    .addAll(topLevelTargetLabels)
                    .build());

    Set<FileStateKey> newActiveDirectories = Sets.newConcurrentHashSet();

    if (skyfocusState.options().activeDirectories.isEmpty()
        && skyfocusState.activeDirectoriesType().equals(DERIVED)) {
      // If the user hasn't defined a new active directories from the command line and there
      // isn't an active user-defined active directories in use, automatically derive one using the
      // targets being built.
      try (SilentCloseable c = Profiler.instance().profile("Skyfocus derive active directories")) {
        eventHandler.handle(Event.info("Skyfocus: automatically deriving active directories."));

        ImmutableSet<PathFragment> topLevelTargetPackages =
            topLevelTargetLabels.stream().map(Label::getPackageFragment).collect(toImmutableSet());

        // For each FSK, add to the active directories if the FSK's parent dir shares the same
        // package as one of the top level targets.
        evaluator
            .getInMemoryGraph()
            .parallelForEach(
                node -> {
                  if (node.getKey() instanceof FileStateKey fileStateKey) {
                    Preconditions.checkState(
                        node.isDone(),
                        "FileState node is not done. This is an internal inconsistency.");
                    if (node.getValue().equals(NONEXISTENT_FILE_STATE_NODE)) {
                      return;
                    }

                    if (activeDirectoriesMatcher.isPresent()) {
                      // Check if the file belongs to the given active directories prefixes.
                      if (activeDirectoriesMatcher
                          .get()
                          .includes(fileStateKey.argument().getRootRelativePath())) {
                        newActiveDirectories.add(fileStateKey.argument());
                      }
                      return;
                    }

                    // If project directories are not defined, check if the file belongs to the
                    // package of a top level target being built.
                    PathFragment currPath = fileStateKey.argument().getRootRelativePath();
                    while (currPath != null) {
                      try {
                        if (packageManager.isPackage(
                            eventHandler,
                            PackageIdentifier.create(RepositoryName.MAIN, currPath))) {
                          if (topLevelTargetPackages.contains(currPath)) {
                            newActiveDirectories.add(fileStateKey.argument());
                          }
                          break;
                        }
                      } catch (InconsistentFilesystemException e) {
                        throw new IllegalStateException(e);
                      } catch (InterruptedException e) {
                        // Swallow interrupted exceptions at this level, since this is probably from
                        // the main thread, and so there's not much else to do here.
                        //
                        // If this is a stray SIGINT, then we can't do much here either.
                      }

                      // traverse up the path until we find a valid package
                      currPath = currPath.getParentDirectory();
                    }
                  }
                });

        if (!skyfocusState.forcedRerun()
            && skyfocusState.activeDirectories().containsAll(newActiveDirectories)
            && skyfocusState.focusedTargetLabels().containsAll(topLevelTargetLabels)) {
          // Already focused on a superset of the active directories, no need to do anything.
          return Optional.empty();
        }

        newSkyfocusStateBuilder
            .activeDirectoriesType(DERIVED)
            .activeDirectories(
                ImmutableSet.<FileStateKey>builder()
                    .addAll(
                        // Only persist previously derived active directoriess. If they were
                        // user defined, overwrite them.
                        skyfocusState.activeDirectoriesType().equals(DERIVED)
                            ? skyfocusState.activeDirectories()
                            : ImmutableSet.of())
                    .addAll(newActiveDirectories)
                    .build());
      }
    } else {
      if (skyfocusState.options().activeDirectories.isEmpty() && !skyfocusState.forcedRerun()) {
        // No command line request to update the active directories; return early.
        return Optional.empty();
      }

      // User is setting a new explicit active directories from the command line option.
      // This will override any previously defined active directories.

      ImmutableSet<RootedPath> activeDirectoriesRootedPaths =
          Stream.concat(
                  Stream.concat(
                          skyfocusState.options().activeDirectories.stream(),
                          // The Bzlmod lockfile can be created after a build without having existed
                          // before and must always be kept in the active directories if it is used.
                          Stream.of(LabelConstants.MODULE_LOCKFILE_NAME.toString()))
                      .map(k -> toFileStateKey(pkgLocator, k)),
                  Stream.of(
                      RootedPath.toRootedPath(
                          Root.fromPath(pkgLocator.getOutputBase()),
                          LabelConstants.MODULE_LOCKFILE_NAME)))
              .collect(toImmutableSet());
      evaluator
          .getInMemoryGraph()
          .parallelForEach(
              node -> {
                if (node.getKey() instanceof FileStateKey fileStateKey) {
                  RootedPath rootedPath = fileStateKey.argument();
                  if (activeDirectoriesRootedPaths.contains(rootedPath)) {
                    newActiveDirectories.add(fileStateKey);
                  }
                }
              });

      int missingCount = activeDirectoriesRootedPaths.size() - newActiveDirectories.size();
      if (missingCount > 0) {
        eventHandler.handle(
            Event.warn(
                missingCount
                    + " files were not found in the transitive closure, and so they are not"
                    + " included in the active directories. They are: "
                    + activeDirectoriesRootedPaths.stream()
                        .filter(not(newActiveDirectories::contains))
                        .map(r -> r.getRootRelativePath().toString())
                        .collect(joining(", "))));
      }

      if ((skyfocusState.options().activeDirectories.isEmpty()
              || skyfocusState.activeDirectories().equals(newActiveDirectories))
          && skyfocusState.focusedTargetLabels().containsAll(topLevelTargetLabels)) {
        if (skyfocusState.forcedRerun()) {
          newActiveDirectories.addAll(skyfocusState.activeDirectories());
        } else {
          return Optional.empty();
        }
      }

      newSkyfocusStateBuilder
          .activeDirectoriesType(USER_DEFINED)
          .activeDirectories(ImmutableSet.copyOf(newActiveDirectories));
    }

    eventHandler.handle(Event.info("Updated active directories successfully."));
    return Optional.of(newSkyfocusStateBuilder.build());
  }

  public static FocusResult execute(
      ImmutableSet<FileStateKey> activeDirectories,
      InMemoryMemoizingEvaluator evaluator,
      ExtendedEventHandler eventHandler,
      @Nullable ActionCache actionCache)
      throws InterruptedException {

    Set<SkyKey> roots = evaluator.getLatestTopLevelEvaluations();
    checkState(
        roots != null && !roots.isEmpty(), "Skyfocus needs roots, so it can't be null or empty.");

    ImmutableSet<SkyKey> leafs =
        ImmutableSet.<SkyKey>builder()
            // TODO: b/312819241 - BUILD_ID is necessary for build correctness of volatile actions,
            // like stamping, but retains a lot of memory (100MB of retained heap for a 9+GB build).
            // Figure out a way to not include it.
            .add(PrecomputedValue.BUILD_ID.getKey())
            .addAll(activeDirectories)
            .build();

    eventHandler.handle(
        Event.info(
            String.format(
                "Focusing on %d roots, %d leafs... (use --experimental_skyfocus_dump_keys to show"
                    + " them)",
                roots.size(), leafs.size())));

    FocusResult focusResult;

    try (SilentCloseable c = Profiler.instance().profile("SkyframeFocuser")) {
      focusResult = SkyframeFocuser.focus(evaluator.getInMemoryGraph(), actionCache, roots, leafs);
    }

    return focusResult;
  }

  /** Turns a root relative path string into a RootedPath object. */
  static RootedPath toFileStateKey(PathPackageLocator pkgLocator, String rootRelativePathFragment) {
    // For simplicity's sake, use the first --package_path as the root. This
    // may be an issue with packages from a different package_path root.
    // TODO: b/312819241  - handle multiple package_path roots.
    Root packageRoot = pkgLocator.getPathEntries().get(0);
    return RootedPath.toRootedPath(packageRoot, PathFragment.create(rootRelativePathFragment));
  }
}
