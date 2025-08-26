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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction.LOCKFILE_MODE;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

/**
 * Module collecting Bazel module and module extensions resolution results and updating the
 * lockfile.
 */
public class BazelLockFileModule extends BlazeModule {

  private SkyframeExecutor executor;
  private Path workspaceRoot;
  private Path outputBase;
  private LockfileMode optionsLockfileMode;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final ImmutableSet<LockfileMode> ENABLED_IN_MODES =
      Sets.immutableEnumSet(LockfileMode.UPDATE, LockfileMode.REFRESH);

  @Override
  public void beforeCommand(CommandEnvironment env) {
    executor = env.getSkyframeExecutor();
    workspaceRoot = env.getWorkspace();
    outputBase = env.getOutputBase();
    optionsLockfileMode = env.getOptions().getOptions(RepositoryOptions.class).lockfileMode;
  }

  @Override
  @SuppressWarnings("AllowVirtualThreads")
  public void afterCommand() {
    MemoizingEvaluator evaluator = executor.getEvaluator();
    BazelModuleResolutionValue moduleResolutionValue;
    BazelDepGraphValue depGraphValue;
    BazelLockFileValue oldLockfile;
    BazelLockFileValue oldHiddenLockfile;
    try {
      PrecomputedValue lockfileModeValue =
          (PrecomputedValue) evaluator.getExistingValue(LOCKFILE_MODE.getKey());
      if (lockfileModeValue == null) {
        // No command run on this server has triggered module resolution yet.
        return;
      }
      // Check the Skyframe value in addition to the option since some commands (e.g. shutdown)
      // don't propagate the options to Skyframe, but we can only operate on Skyframe values that
      // were generated in UPDATE mode.
      LockfileMode skyframeLockfileMode = (LockfileMode) lockfileModeValue.get();
      if (!(ENABLED_IN_MODES.contains(optionsLockfileMode)
          && ENABLED_IN_MODES.contains(skyframeLockfileMode))) {
        return;
      }
      moduleResolutionValue =
          (BazelModuleResolutionValue) evaluator.getExistingValue(BazelModuleResolutionValue.KEY);
      depGraphValue = (BazelDepGraphValue) evaluator.getExistingValue(BazelDepGraphValue.KEY);
      oldLockfile = (BazelLockFileValue) evaluator.getExistingValue(BazelLockFileValue.KEY);
      oldHiddenLockfile =
          (BazelLockFileValue) evaluator.getExistingValue(BazelLockFileValue.HIDDEN_KEY);
    } catch (InterruptedException e) {
      // Not thrown in Bazel.
      throw new IllegalStateException(e);
    }
    if (moduleResolutionValue == null || depGraphValue == null || oldLockfile == null) {
      // Since these values are required to compute the main repo mapping, which happens in every
      // build, an error must have occurred that prevented the evaluation of these values and that
      // has already been reported at this point.
      return;
    }
    if (oldHiddenLockfile == null) {
      oldHiddenLockfile = BazelLockFileValue.EMPTY_LOCKFILE;
    }

    // All nodes corresponding to module extensions that have been evaluated in the current build
    // are done at this point. Look up entries by eval keys to record results even if validation
    // later fails due to invalid imports.
    // Note: This also picks up up-to-date results from previous builds that are not in the
    // transitive closure of the current build. Since extensions are potentially costly to evaluate,
    // this is seen as an advantage. Full reproducibility can be ensured by running 'bazel shutdown'
    // first if needed.
    int maxNumExtensions = depGraphValue.getExtensionUsagesTable().rowMap().size();
    // Presize conservatively to avoid blocking for resizing.
    Map<ModuleExtensionId, LockFileModuleExtension.WithFactors> newExtensionInfos =
        new ConcurrentHashMap<>(maxNumExtensions);
    executor
        .getEvaluator()
        .getInMemoryGraph()
        .parallelForEach(
            entry -> {
              if (entry.isDone()
                  && entry.getKey() instanceof SingleExtensionValue.EvalKey key
                  // entry.getValue() can be null if the extension evaluation failed.
                  && entry.getValue() instanceof SingleExtensionValue value
                  // The innate extensions are implemented in Java and don't benefit from a lockfile
                  // entry.
                  && !key.argument().isInnate()) {
                newExtensionInfos.put(key.argument(), value.lockFileInfo().get());
              }
            });

    Thread updateLockfile =
        Thread.startVirtualThread(
            () -> {
              var notReproducibleExtensionInfos =
                  combineModuleExtensions(
                      oldLockfile.getModuleExtensions(),
                      newExtensionInfos,
                      /* hasUsages= */ depGraphValue.getExtensionUsagesTable()::containsRow,
                      /* reproducible= */ false);

              // Create an updated version of the lockfile, keeping only the extension results from
              // the old lockfile that are still up-to-date and adding the newly resolved
              // extension results, as long as any of them are not known to be reproducible.
              BazelLockFileValue newLockfile =
                  BazelLockFileValue.builder()
                      .setRegistryFileHashes(
                          ImmutableSortedMap.copyOf(moduleResolutionValue.getRegistryFileHashes()))
                      .setSelectedYankedVersions(moduleResolutionValue.getSelectedYankedVersions())
                      .setModuleExtensions(notReproducibleExtensionInfos)
                      .build();

              // Write the new values to the files, but only if needed. This is not just a
              // performance optimization: whenever the lockfile is updated, most Skyframe nodes
              // will be marked as dirty on the next build, which breaks commands such as `bazel
              // config` that rely on
              // com.google.devtools.build.skyframe.MemoizingEvaluator#getDoneValues.
              if (!newLockfile.equals(oldLockfile)) {
                updateLockfile(workspaceRoot, newLockfile);
              }
            });

    final BazelLockFileValue oldHiddenLockfileFinal = oldHiddenLockfile;
    Thread updateHiddenLockfile =
        Thread.startVirtualThread(
            () -> {
              // Results of reproducible extensions do not need to be stored for reproducibility,
              // but avoiding reevaluations on server startups helps cold build performance.
              var reproducibleExtensionInfos =
                  combineModuleExtensions(
                      oldHiddenLockfileFinal.getModuleExtensions(),
                      newExtensionInfos,
                      /* hasUsages= */ depGraphValue.getExtensionUsagesTable()::containsRow,
                      /* reproducible= */ true);
              BazelLockFileValue newHiddenLockfile =
                  BazelLockFileValue.builder()
                      .setSelectedYankedVersions(ImmutableMap.of())
                      .setModuleExtensions(reproducibleExtensionInfos)
                      .build();

              if (!newHiddenLockfile.equals(oldHiddenLockfileFinal)) {
                updateLockfile(outputBase, newHiddenLockfile);
              }
            });

    try {
      updateLockfile.join();
      updateHiddenLockfile.join();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      logger.atSevere().withCause(e).log(
          "Interrupted while updating MODULE.bazel.lock file: %s", e.getMessage());
    }
  }

  /**
   * Combines the old extensions stored in the lockfile -if they are still used and have the same
   * dependence on os/arch - with the new extensions from the events (if any)
   */
  @VisibleForTesting
  static ImmutableMap<
          ModuleExtensionId, ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
      combineModuleExtensions(
          Map<ModuleExtensionId, ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
              oldExtensionInfos,
          Map<ModuleExtensionId, LockFileModuleExtension.WithFactors> newExtensionInfos,
          Predicate<ModuleExtensionId> hasUsages,
          boolean reproducible) {
    Map<ModuleExtensionId, ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
        updatedExtensionMap = new HashMap<>();

    // Keep those per factor extension results that are still used according to the static
    // information given in the extension declaration (dependence on os and arch).
    // Other information such as transitive .bzl hash and usages hash are *not* checked here.
    for (var entry : oldExtensionInfos.entrySet()) {
      var moduleExtensionId = entry.getKey();
      if (!hasUsages.test(moduleExtensionId)) {
        // Extensions without any usages are not needed anymore.
        continue;
      }
      var newExtensionInfo = newExtensionInfos.get(moduleExtensionId);
      if (newExtensionInfo == null) {
        // No information based on which we could invalidate old entries, keep all of them.
        updatedExtensionMap.put(moduleExtensionId, entry.getValue());
        continue;
      }
      var newFactors = newExtensionInfo.extensionFactors();
      // Prefer the new result for its particular set of factors.
      var perFactorsResultsToKeep =
          ImmutableSortedMap.copyOf(
              Maps.filterKeys(
                  entry.getValue(),
                  oldFactors ->
                      oldFactors.hasSameDependenciesAs(newFactors)
                          && !oldFactors.equals(newFactors)));
      if (perFactorsResultsToKeep.isEmpty()) {
        continue;
      }
      updatedExtensionMap.put(moduleExtensionId, perFactorsResultsToKeep);
    }

    // Add the new resolved extensions
    for (var extensionIdAndInfo : newExtensionInfos.entrySet()) {
      LockFileModuleExtension extension = extensionIdAndInfo.getValue().moduleExtension();
      if (extension.isReproducible() != reproducible) {
        continue;
      }

      var oldExtensionEntries = updatedExtensionMap.get(extensionIdAndInfo.getKey());
      ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension> extensionEntries;
      var factors = extensionIdAndInfo.getValue().extensionFactors();
      if (oldExtensionEntries != null) {
        // extension exists, add the new entry to the existing map
        extensionEntries =
            ImmutableSortedMap.copyOf(
                ImmutableMap.<ModuleExtensionEvalFactors, LockFileModuleExtension>builder()
                    .putAll(oldExtensionEntries)
                    .put(factors, extension)
                    .buildKeepingLast());
      } else {
        // new extension
        extensionEntries = ImmutableMap.of(factors, extension);
      }
      updatedExtensionMap.put(extensionIdAndInfo.getKey(), extensionEntries);
    }

    // The order in which extensions are added to extensionResolutionEvents depends on the order
    // in which their Skyframe evaluations finish, which is non-deterministic. We ensure a
    // deterministic lockfile by sorting.
    return ImmutableSortedMap.copyOf(
        updatedExtensionMap, ModuleExtensionId.LEXICOGRAPHIC_COMPARATOR);
  }

  /**
   * Updates the data stored in the lockfile (MODULE.bazel.lock)
   *
   * @param lockfileRoot Root under which the lockfile is located
   * @param updatedLockfile The updated lockfile data to save
   */
  private static void updateLockfile(Path lockfileRoot, BazelLockFileValue updatedLockfile) {
    RootedPath lockfilePath =
        RootedPath.toRootedPath(Root.fromPath(lockfileRoot), LabelConstants.MODULE_LOCKFILE_NAME);
    try {
      FileSystemUtils.writeContent(
          lockfilePath.asPath(),
          UTF_8,
          GsonTypeAdapterUtil.LOCKFILE_GSON.toJson(updatedLockfile) + "\n");
    } catch (IOException e) {
      logger.atSevere().withCause(e).log(
          "Error while updating MODULE.bazel.lock file: %s", e.getMessage());
    }
  }
}
