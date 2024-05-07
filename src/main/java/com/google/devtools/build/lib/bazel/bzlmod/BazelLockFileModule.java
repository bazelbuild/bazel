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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Module collecting Bazel module and module extensions resolution results and updating the
 * lockfile.
 */
public class BazelLockFileModule extends BlazeModule {

  private SkyframeExecutor executor;
  private Path workspaceRoot;
  private boolean enabled;
  @Nullable private BazelModuleResolutionEvent moduleResolutionEvent;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @Override
  public void beforeCommand(CommandEnvironment env) {
    executor = env.getSkyframeExecutor();
    workspaceRoot = env.getWorkspace();

    enabled =
        env.getOptions().getOptions(RepositoryOptions.class).lockfileMode == LockfileMode.UPDATE;
    moduleResolutionEvent = null;
    env.getEventBus().register(this);
  }

  @Override
  public void afterCommand() {
    if (!enabled || moduleResolutionEvent == null) {
      // Command does not use Bazel modules or the lockfile mode is not update.
      // Since Skyframe caches events, they are replayed even when nothing has changed.
      return;
    }

    BazelDepGraphValue depGraphValue;
    BazelModuleResolutionValue moduleResolutionValue;
    try {
      depGraphValue =
          (BazelDepGraphValue) executor.getEvaluator().getExistingValue(BazelDepGraphValue.KEY);
      moduleResolutionValue =
          (BazelModuleResolutionValue)
              executor.getEvaluator().getExistingValue(BazelModuleResolutionValue.KEY);
    } catch (InterruptedException e) {
      // Not thrown in Bazel.
      throw new IllegalStateException(e);
    }

    BazelLockFileValue oldLockfile = moduleResolutionEvent.getOnDiskLockfileValue();
    ImmutableMap<String, Optional<Checksum>> fileHashes;
    if (moduleResolutionValue == null) {
      // BazelDepGraphFunction took the dep graph from the lockfile and didn't cause evaluation of
      // BazelModuleResolutionFunction. The file hashes in the lockfile are still up-to-date.
      fileHashes = oldLockfile.getRegistryFileHashes();
    } else {
      fileHashes = ImmutableSortedMap.copyOf(moduleResolutionValue.getRegistryFileHashes());
    }

    // All nodes corresponding to module extensions that have been evaluated in the current build
    // are done at this point. Look up entries by eval keys to record results even if validation
    // later fails due to invalid imports.
    // Note: This also picks up up-to-date result from previous builds that are not in the
    // transitive closure of the current build. Since extension are potentially costly to evaluate,
    // this is seen as an advantage. Full reproducibility can be ensured by running 'bazel shutdown'
    // first if needed.
    Map<ModuleExtensionId, LockFileModuleExtension.WithFactors> newExtensionInfos =
        new ConcurrentHashMap<>();
    executor
        .getEvaluator()
        .getInMemoryGraph()
        .parallelForEach(
            entry -> {
              if (entry.isDone()
                  && entry.getKey() instanceof SingleExtensionValue.EvalKey key
                  // entry.getValue() can be null if the extension evaluation failed.
                  && entry.getValue() instanceof SingleExtensionValue value) {
                newExtensionInfos.put(key.argument(), value.getLockFileInfo().get());
              }
            });
    var combinedExtensionInfos =
        combineModuleExtensions(
            oldLockfile.getModuleExtensions(), newExtensionInfos, depGraphValue);

    // Create an updated version of the lockfile, keeping only the extension results from the old
    // lockfile that are still up-to-date and adding the newly resolved extension results.
    BazelLockFileValue newLockfile =
        moduleResolutionEvent.getResolutionOnlyLockfileValue().toBuilder()
            .setRegistryFileHashes(fileHashes)
            .setModuleExtensions(combinedExtensionInfos)
            .build();

    // Write the new value to the file, but only if needed. This is not just a performance
    // optimization: whenever the lockfile is updated, most Skyframe nodes will be marked as dirty
    // on the next build, which breaks commands such as `bazel config` that rely on
    // com.google.devtools.build.skyframe.MemoizingEvaluator#getDoneValues.
    if (!newLockfile.equals(oldLockfile)) {
      updateLockfile(workspaceRoot, newLockfile);
    }
  }

  /**
   * Combines the old extensions stored in the lockfile -if they are still valid- with the new
   * extensions from the events (if any)
   */
  private ImmutableMap<
          ModuleExtensionId, ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
      combineModuleExtensions(
          ImmutableMap<
                  ModuleExtensionId,
                  ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
              oldExtensionInfos,
          Map<ModuleExtensionId, LockFileModuleExtension.WithFactors> newExtensionInfos,
          BazelDepGraphValue depGraphValue) {
    Map<ModuleExtensionId, ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension>>
        updatedExtensionMap = new HashMap<>();

    // Keep old extensions if they are still valid.
    for (var entry : oldExtensionInfos.entrySet()) {
      var moduleExtensionId = entry.getKey();
      var factorToLockedExtension = entry.getValue();
      ModuleExtensionEvalFactors firstEntryFactors =
          factorToLockedExtension.keySet().iterator().next();
      // All entries for a single extension share the same usages digest, so it suffices to check
      // the first entry.
      if (shouldKeepExtension(
          moduleExtensionId,
          firstEntryFactors,
          newExtensionInfos.get(moduleExtensionId),
          depGraphValue)) {
        updatedExtensionMap.put(moduleExtensionId, factorToLockedExtension);
      }
    }

    // Add the new resolved extensions
    for (var extensionIdAndInfo : newExtensionInfos.entrySet()) {
      LockFileModuleExtension extension = extensionIdAndInfo.getValue().moduleExtension();
      if (!extension.shouldLockExtension()) {
        continue;
      }

      var oldExtensionEntries = updatedExtensionMap.get(extensionIdAndInfo.getKey());
      ImmutableMap<ModuleExtensionEvalFactors, LockFileModuleExtension> extensionEntries;
      var factors = extensionIdAndInfo.getValue().extensionFactors();
      if (oldExtensionEntries != null) {
        // extension exists, add the new entry to the existing map
        extensionEntries =
            new ImmutableMap.Builder<ModuleExtensionEvalFactors, LockFileModuleExtension>()
                .putAll(oldExtensionEntries)
                .put(factors, extension)
                .buildKeepingLast();
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
   * Keep an extension if and only if:
   *
   * <ol>
   *   <li>it still has usages
   *   <li>its dependency on os & arch didn't change
   *   <li>it hasn't become reproducible
   * </ol>
   *
   * We do not check for changes in the usage hash of the extension or e.g. hashes of files accessed
   * during the evaluation. These values are checked in SingleExtensionEvalFunction.
   *
   * @param lockedExtensionKey object holding the old extension id and state of os and arch
   * @param newExtensionInfo the in-memory lockfile entry produced by the most recent up-to-date
   *     evaluation of this extension (if any)
   * @param depGraphValue the dep graph value
   * @return True if this extension should still be in lockfile, false otherwise
   */
  private boolean shouldKeepExtension(
      ModuleExtensionId moduleExtensionId,
      ModuleExtensionEvalFactors lockedExtensionKey,
      @Nullable LockFileModuleExtension.WithFactors newExtensionInfo,
      BazelDepGraphValue depGraphValue) {
    if (!depGraphValue.getExtensionUsagesTable().containsRow(moduleExtensionId)) {
      return false;
    }

    // If there is no new extension info, the properties of the existing extension entry can't have
    // changed.
    if (newExtensionInfo == null) {
      return true;
    }

    boolean doNotLockExtension = !newExtensionInfo.moduleExtension().shouldLockExtension();
    boolean dependencyOnOsChanged =
        lockedExtensionKey.getOs().isEmpty()
            != newExtensionInfo.extensionFactors().getOs().isEmpty();
    boolean dependencyOnArchChanged =
        lockedExtensionKey.getArch().isEmpty()
            != newExtensionInfo.extensionFactors().getArch().isEmpty();
    return !doNotLockExtension && !dependencyOnOsChanged && !dependencyOnArchChanged;
  }

  /**
   * Updates the data stored in the lockfile (MODULE.bazel.lock)
   *
   * @param workspaceRoot Root of the workspace where the lockfile is located
   * @param updatedLockfile The updated lockfile data to save
   */
  public static void updateLockfile(Path workspaceRoot, BazelLockFileValue updatedLockfile) {
    RootedPath lockfilePath =
        RootedPath.toRootedPath(Root.fromPath(workspaceRoot), LabelConstants.MODULE_LOCKFILE_NAME);
    try {
      FileSystemUtils.writeContent(
          lockfilePath.asPath(),
          UTF_8,
          GsonTypeAdapterUtil.createLockFileGson(
                      lockfilePath
                          .asPath()
                          .getParentDirectory()
                          .getRelative(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME),
                      workspaceRoot)
                  .toJson(updatedLockfile)
              + "\n");
    } catch (IOException e) {
      logger.atSevere().withCause(e).log(
          "Error while updating MODULE.bazel.lock file: %s", e.getMessage());
    }
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void bazelModuleResolved(BazelModuleResolutionEvent moduleResolutionEvent) {
    // Latest event wins, which is relevant in the case of `bazel mod tidy`, where a new event is
    // sent after the command has modified the module file.
    this.moduleResolutionEvent = moduleResolutionEvent;
  }
}
