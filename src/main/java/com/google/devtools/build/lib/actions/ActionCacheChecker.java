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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.ActionCache.Entry.SerializableTreeArtifactValue;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.cache.Protos.ActionCacheStatistics.MissReason;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.ArchivedRepresentation;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Checks whether an {@link Action} needs to be executed, or whether it has not changed since it was
 * last stored in the action cache. Must be informed of the new Action data after execution as well.
 *
 * <p>The fingerprint, input files names, and metadata (either mtimes or MD5sums) of each action are
 * cached in the action cache to avoid unnecessary rebuilds.
 *
 * <p>While instances of this class hold references to action and metadata cache instances, they are
 * otherwise lightweight, and should be constructed anew and discarded for each build request.
 */
public class ActionCacheChecker {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ActionKeyContext actionKeyContext;
  private final Predicate<? super Action> executionFilter;
  private final ArtifactResolver artifactResolver;
  private final ProxyMetadataFactory proxyMetadataFactory;
  private final CacheConfig cacheConfig;

  @Nullable private final ActionCache actionCache; // Null when not enabled.

  /** Cache config parameters for ActionCacheChecker. */
  @AutoValue
  public abstract static class CacheConfig {
    abstract boolean enabled();

    abstract boolean storeOutputMetadata();

    public static Builder builder() {
      return new AutoValue_ActionCacheChecker_CacheConfig.Builder();
    }

    /** Builder for ActionCacheChecker.CacheConfig. */
    @AutoValue.Builder
    public abstract static class Builder {
      public abstract Builder setEnabled(boolean value);

      public abstract Builder setStoreOutputMetadata(boolean value);

      public abstract CacheConfig build();
    }
  }

  public ActionCacheChecker(
      @Nullable ActionCache actionCache,
      ArtifactResolver artifactResolver,
      ActionKeyContext actionKeyContext,
      Predicate<? super Action> executionFilter,
      ProxyMetadataFactory proxyMetadataFactory,
      @Nullable CacheConfig cacheConfig) {
    this.executionFilter = executionFilter;
    this.actionKeyContext = actionKeyContext;
    this.artifactResolver = artifactResolver;
    this.proxyMetadataFactory = proxyMetadataFactory;
    this.cacheConfig =
        cacheConfig != null
            ? cacheConfig
            : CacheConfig.builder().setEnabled(true).setStoreOutputMetadata(false).build();
    if (this.cacheConfig.enabled()) {
      this.actionCache = Preconditions.checkNotNull(actionCache);
    } else {
      this.actionCache = null;
    }
  }

  public boolean isActionExecutionProhibited(Action action) {
    return !executionFilter.apply(action);
  }

  /** Whether the action cache is enabled. */
  public boolean enabled() {
    return cacheConfig.enabled();
  }

  /**
   * Checks whether one of existing output paths is already used as a key. If yes, returns it -
   * otherwise uses first output file as a key
   */
  @Nullable
  private ActionCache.Entry getCacheEntry(Action action) {
    if (!cacheConfig.enabled()) {
      return null; // ignore existing cache when disabled.
    }
    return ActionCacheUtils.getCacheEntry(actionCache, action);
  }

  public void removeCacheEntry(Action action) {
    checkState(enabled(), "Action cache disabled");
    ActionCacheUtils.removeCacheEntry(actionCache, action);
  }

  @Nullable
  private static FileArtifactValue getCachedMetadata(
      @Nullable CachedOutputMetadata cachedOutputMetadata, Artifact artifact) {
    checkArgument(!artifact.isTreeArtifact());

    if (cachedOutputMetadata == null) {
      return null;
    }

    return cachedOutputMetadata.fileMetadata.get(artifact);
  }

  @Nullable
  private static TreeArtifactValue getCachedTreeMetadata(
      @Nullable CachedOutputMetadata cachedOutputMetadata, Artifact artifact) {
    checkArgument(artifact.isTreeArtifact());

    if (cachedOutputMetadata == null) {
      return null;
    }

    return cachedOutputMetadata.treeMetadata.get((SpecialArtifact) artifact);
  }

  /**
   * Returns whether an action cache entry is up to date.
   *
   * @param entry action cache entry
   * @param action action to be validated.
   * @param actionKey the action key previously obtained from action.getKey()
   * @param actionInputs the action inputs; usually action.getInputs(), but might be a previously
   *     cached set of discovered inputs for actions that discover them.
   * @param outputMetadataStore metadata provider for action outputs.
   * @param cachedOutputMetadata cached metadata that should be used instead of {@code
   *     outputMetadataStore}.
   * @param outputChecker used to check whether remote metadata should be trusted.
   * @param effectiveEnvironment the effective client environment for the action.
   * @param effectiveExecProperties the effective exec properties for the action.
   * @param outputPermissions the requested output permissions
   * @param useArchivedTreeArtifacts whether archived tree artifacts are enabled.
   * @return whether the action cache entry is valid.
   */
  private static boolean isUpToDate(
      ActionCache.Entry entry,
      Action action,
      String actionKey,
      NestedSet<Artifact> actionInputs,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore,
      @Nullable CachedOutputMetadata cachedOutputMetadata,
      @Nullable OutputChecker outputChecker,
      ImmutableMap<String, String> effectiveEnvironment,
      ImmutableMap<String, String> effectiveExecProperties,
      OutputPermissions outputPermissions,
      boolean useArchivedTreeArtifacts)
      throws InterruptedException {
    var builder =
        new ActionCache.Entry.Builder(
            actionKey,
            action.discoversInputs(),
            effectiveEnvironment,
            effectiveExecProperties,
            outputPermissions,
            useArchivedTreeArtifacts);

    for (Artifact artifact : action.getOutputs()) {
      if (artifact.isTreeArtifact()) {
        TreeArtifactValue treeMetadata = getCachedTreeMetadata(cachedOutputMetadata, artifact);
        if (treeMetadata == null) {
          treeMetadata = getOutputTreeMetadataMaybe(outputMetadataStore, artifact);
        }
        if (treeMetadata != null
            && shouldTrustTreeMetadata(artifact, treeMetadata, outputChecker)) {
          builder.addOutputTree((SpecialArtifact) artifact, treeMetadata);
        } else {
          return false;
        }

      } else {
        FileArtifactValue metadata = getCachedMetadata(cachedOutputMetadata, artifact);
        if (metadata == null) {
          metadata = getOutputMetadataMaybe(outputMetadataStore, artifact);
        }
        if (metadata != null && shouldTrustMetadata(artifact, metadata, outputChecker)) {
          builder.addOutputFile(artifact, metadata);
        } else {
          return false;
        }
      }
    }
    for (Artifact artifact : actionInputs.toList()) {
      FileArtifactValue inputMetadata = getInputMetadataMaybe(inputMetadataProvider, artifact);
      builder.addInputFile(artifact, inputMetadata);
    }
    return Arrays.equals(entry.getDigest(), builder.build().getDigest());
  }

  private static boolean shouldTrustMetadata(
      Artifact artifact, FileArtifactValue metadata, @Nullable OutputChecker outputChecker) {
    checkArgument(!artifact.isTreeArtifact());
    if (outputChecker == null) {
      return true;
    }
    return outputChecker.shouldTrustMetadata(artifact, metadata);
  }

  private static boolean shouldTrustTreeMetadata(
      Artifact artifact, TreeArtifactValue treeMetadata, @Nullable OutputChecker outputChecker) {
    checkArgument(artifact.isTreeArtifact());
    if (outputChecker == null) {
      return true;
    }
    if (treeMetadata.getArchivedRepresentation().isPresent()) {
      ArchivedTreeArtifact archivedArtifact =
          treeMetadata
              .getArchivedRepresentation()
              .map(ArchivedRepresentation::archivedTreeFileArtifact)
              .orElseThrow();
      FileArtifactValue archivedMetadata =
          treeMetadata
              .getArchivedRepresentation()
              .map(ArchivedRepresentation::archivedFileValue)
              .orElseThrow();
      if (!outputChecker.shouldTrustMetadata(archivedArtifact, archivedMetadata)) {
        return false;
      }
    }
    for (Map.Entry<TreeFileArtifact, FileArtifactValue> entry :
        treeMetadata.getChildValues().entrySet()) {
      TreeFileArtifact child = entry.getKey();
      FileArtifactValue childMetadata = entry.getValue();
      if (!outputChecker.shouldTrustMetadata(child, childMetadata)) {
        return false;
      }
    }
    return true;
  }

  private boolean unconditionalExecution(Action action) {
    return !isActionExecutionProhibited(action) && action.executeUnconditionally();
  }

  private static ImmutableMap<String, String> computeEffectiveExecProperties(
      Action action, ImmutableMap<String, String> defaultExecProperties) {
    return action.getExecProperties().isEmpty()
        ? defaultExecProperties
        : action.getExecProperties();
  }

  private static ImmutableMap<String, String> computeEffectiveEnvironment(
      Action action, Map<String, String> clientEnv) {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    for (String var : action.getClientEnvironmentVariables()) {
      String value = clientEnv.get(var);
      if (value != null) {
        builder.put(var, value);
      }
    }
    return builder.buildKeepingLast();
  }

  /**
   * The currently cached outputs when output metadata is stored (i.e., {@code
   * CacheConfig#shouldStoreOutputMetadata}).
   *
   * <p>Metadata retrieved from the filesystem overrides the cached metadata. This way, an action
   * will not be rerun if the cached metadata is still valid, unless the filesystem state needs to
   * be updated.
   */
  private record CachedOutputMetadata(
      ImmutableMap<Artifact, FileArtifactValue> fileMetadata,
      ImmutableMap<SpecialArtifact, TreeArtifactValue> treeMetadata) {}

  private CachedOutputMetadata loadCachedOutputMetadata(
      Action action, ActionCache.Entry entry, OutputMetadataStore outputMetadataStore)
      throws InterruptedException {
    ImmutableMap.Builder<Artifact, FileArtifactValue> mergedFileMetadata = ImmutableMap.builder();
    ImmutableMap.Builder<SpecialArtifact, TreeArtifactValue> mergedTreeMetadata =
        ImmutableMap.builder();
    ImmutableSet<String> proxyOutputs = ImmutableSet.copyOf(entry.getProxyOutputs());

    for (Artifact artifact : action.getOutputs()) {
      if (artifact.isTreeArtifact()) {
        SpecialArtifact parent = (SpecialArtifact) artifact;
        SerializableTreeArtifactValue cachedTreeMetadata = entry.getOutputTree(parent);
        if (cachedTreeMetadata == null) {
          continue;
        }

        Map<TreeFileArtifact, FileArtifactValue> childValues = new HashMap<>();
        // Load remote child file metadata from cache.
        cachedTreeMetadata
            .childValues()
            .forEach(
                (key, value) ->
                    childValues.put(TreeFileArtifact.createTreeOutput(parent, key), value));

        Optional<ArchivedRepresentation> archivedRepresentation =
            cachedTreeMetadata
                .archivedFileValue()
                .map(
                    fileArtifactValue ->
                        ArchivedRepresentation.create(
                            ArchivedTreeArtifact.createForTree(parent), fileArtifactValue));

        TreeArtifactValue filesystemTreeMetadata;
        try {
          filesystemTreeMetadata = outputMetadataStore.getTreeArtifactValue(parent);
        } catch (FileNotFoundException ignored) {
          filesystemTreeMetadata = null;
        } catch (IOException e) {
          // Ignore the cached metadata if we encountered an error when loading its counterpart from
          // the filesystem.
          logger.atWarning().withCause(e).log("Failed to load metadata for %s", parent);
          continue;
        }

        if (filesystemTreeMetadata != null) {
          // Filesystem metadata overrides the cached metadata.
          childValues.putAll(filesystemTreeMetadata.getChildValues());
          if (filesystemTreeMetadata.getArchivedRepresentation().isPresent()) {
            archivedRepresentation = filesystemTreeMetadata.getArchivedRepresentation();
          }
        }

        TreeArtifactValue.Builder merged = TreeArtifactValue.newBuilder(parent);
        childValues.forEach(merged::putChild);
        archivedRepresentation.ifPresent(merged::setArchivedRepresentation);

        mergedTreeMetadata.put(parent, merged.build());
      } else {
        FileArtifactValue cachedMetadata;
        if (proxyOutputs.contains(artifact.getExecPathString())) {
          try {
            cachedMetadata = proxyMetadataFactory.createProxyMetadata(artifact);
          } catch (IOException e) {
            cachedMetadata = null;
          }
        } else {
          cachedMetadata = entry.getOutputFile(artifact);
        }

        if (cachedMetadata == null) {
          continue;
        }

        FileArtifactValue filesystemMetadata;
        try {
          filesystemMetadata = getOutputMetadataOrConstant(outputMetadataStore, artifact);
        } catch (FileNotFoundException ignored) {
          filesystemMetadata = null;
        } catch (IOException e) {
          // Ignore the cached metadata if we encountered an error when loading its counterpart from
          // the filesystem.
          logger.atWarning().withCause(e).log("Failed to load metadata for %s", artifact);
          continue;
        }

        // Filesystem metadata overrides the cached metadata.
        mergedFileMetadata.put(
            artifact, filesystemMetadata != null ? filesystemMetadata : cachedMetadata);
      }
    }

    return new CachedOutputMetadata(
        mergedFileMetadata.buildOrThrow(), mergedTreeMetadata.buildOrThrow());
  }

  /**
   * Checks whether {@code action} needs to be executed and returns a non-null {@link Token} if so.
   *
   * <p>The method checks if any of the action's inputs or outputs have changed. Returns a non-null
   * {@link Token} if the action needs to be executed, and null otherwise.
   *
   * <p>If this method returns non-null, indicating that the action will be executed, the {@code
   * outputMetadataStore} must have any cached metadata cleared so that it does not serve stale
   * metadata for the action's outputs after the action is executed.
   */
  // Note: the handler should only be used for DEPCHECKER events; there's no
  // guarantee it will be available for other events.
  @Nullable
  public Token getTokenIfNeedToExecute(
      Action action,
      List<Artifact> resolvedCacheArtifacts,
      Map<String, String> clientEnv,
      OutputPermissions outputPermissions,
      EventHandler handler,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore,
      ImmutableMap<String, String> remoteDefaultPlatformProperties,
      @Nullable OutputChecker outputChecker,
      boolean useArchivedTreeArtifacts)
      throws InterruptedException {
    // TODO(bazel-team): (2010) For RunfilesAction/SymlinkAction and similar actions that
    // produce only symlinks we should not check whether inputs are valid at all - all that matters
    // that inputs and outputs are still exist (and new inputs have not appeared). All other checks
    // are unnecessary. In other words, the only metadata we should check for them is file existence
    // itself.

    if (!cacheConfig.enabled()) {
      return new Token(action);
    }
    NestedSet<Artifact> actionInputs = action.getInputs();
    // Resolve action inputs from cache, if necessary.
    boolean inputsKnown = action.inputsKnown();
    if (!inputsKnown && resolvedCacheArtifacts != null) {
      // The action doesn't know its inputs, but the caller has a good idea of what they are.
      checkState(
          action.discoversInputs(),
          "Actions that don't know their inputs must discover them: %s",
          action);
      if (action instanceof ActionCacheAwareAction actionCacheAwareAction
          && actionCacheAwareAction.storeInputsExecPathsInActionCache()) {
        actionInputs = NestedSetBuilder.wrap(Order.STABLE_ORDER, resolvedCacheArtifacts);
      } else {
        actionInputs =
            NestedSetBuilder.<Artifact>stableOrder()
                .addTransitive(action.getMandatoryInputs())
                .addAll(resolvedCacheArtifacts)
                .build();
      }
    }
    ActionCache.Entry entry = getCacheEntry(action);
    CachedOutputMetadata cachedOutputMetadata = null;
    if (entry != null && !entry.isCorrupted() && cacheConfig.storeOutputMetadata()) {
      cachedOutputMetadata = loadCachedOutputMetadata(action, entry, outputMetadataStore);
    }

    Token token = new Token(action);
    if (mustExecute(
        action,
        entry,
        token,
        handler,
        inputMetadataProvider,
        outputMetadataStore,
        actionInputs,
        clientEnv,
        outputPermissions,
        remoteDefaultPlatformProperties,
        cachedOutputMetadata,
        outputChecker,
        useArchivedTreeArtifacts)) {
      if (entry != null) {
        removeCacheEntry(action);
      }
      return token;
    }

    if (!inputsKnown) {
      action.updateInputs(actionInputs);
    }

    // Inject cached output metadata if we have an action cache hit.
    if (cachedOutputMetadata != null) {
      cachedOutputMetadata.fileMetadata.forEach(outputMetadataStore::injectFile);
      cachedOutputMetadata.treeMetadata.forEach(outputMetadataStore::injectTree);
    }

    return null;
  }

  private boolean mustExecute(
      Action action,
      @Nullable ActionCache.Entry entry,
      Token token,
      EventHandler handler,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore,
      NestedSet<Artifact> actionInputs,
      Map<String, String> clientEnv,
      OutputPermissions outputPermissions,
      ImmutableMap<String, String> remoteDefaultPlatformProperties,
      @Nullable CachedOutputMetadata cachedOutputMetadata,
      @Nullable OutputChecker outputChecker,
      boolean useArchivedTreeArtifacts)
      throws InterruptedException {
    // Unconditional execution can be applied only for actions that are allowed to be executed.
    if (unconditionalExecution(action)) {
      checkState(action.isVolatile());
      reportUnconditionalExecution(handler, action);
      actionCache.accountMiss(MissReason.UNCONDITIONAL_EXECUTION);
      return true;
    }

    if (entry == null) {
      reportNewAction(handler, action);
      actionCache.accountMiss(MissReason.NOT_CACHED);
      return true;
    }

    if (entry.isCorrupted()) {
      reportCorruptedCacheEntry(handler, action);
      actionCache.accountMiss(MissReason.CORRUPTED_CACHE_ENTRY);
      return true;
    }

    String actionKey = action.getKey(actionKeyContext, inputMetadataProvider);
    token.actionKey = actionKey; // Save the action key for reuse in updateActionCache().

    ImmutableMap<String, String> effectiveEnvironment =
        computeEffectiveEnvironment(action, clientEnv);
    ImmutableMap<String, String> effectiveExecProperties =
        computeEffectiveExecProperties(action, remoteDefaultPlatformProperties);

    if (!isUpToDate(
        entry,
        action,
        actionKey,
        actionInputs,
        inputMetadataProvider,
        outputMetadataStore,
        cachedOutputMetadata,
        outputChecker,
        effectiveEnvironment,
        effectiveExecProperties,
        outputPermissions,
        useArchivedTreeArtifacts)) {
      reportDigestMismatch(handler, action);
      actionCache.accountMiss(MissReason.DIGEST_MISMATCH);
      return true;
    }

    actionCache.accountHit();
    return false;
  }

  private static FileArtifactValue getInputMetadataOrConstant(
      InputMetadataProvider inputMetadataProvider, Artifact artifact) throws IOException {
    FileArtifactValue metadata = inputMetadataProvider.getInputMetadata(artifact);
    return (metadata != null && artifact.isConstantMetadata())
        ? FileArtifactValue.ConstantMetadataValue.INSTANCE
        : metadata;
  }

  private static FileArtifactValue getOutputMetadataOrConstant(
      OutputMetadataStore outputMetadataStore, Artifact artifact)
      throws IOException, InterruptedException {
    FileArtifactValue metadata = outputMetadataStore.getOutputMetadata(artifact);
    return (metadata != null && artifact.isConstantMetadata())
        ? FileArtifactValue.ConstantMetadataValue.INSTANCE
        : metadata;
  }

  // TODO(ulfjack): It's unclear to me why we're ignoring all IOExceptions. In some cases, we want
  // to trigger a re-execution, so we should catch the IOException explicitly there. In others, we
  // should propagate the exception, because it is unexpected (e.g., bad file system state).
  @Nullable
  private static FileArtifactValue getInputMetadataMaybe(
      InputMetadataProvider inputMetadataProvider, Artifact artifact) {
    try {
      return getInputMetadataOrConstant(inputMetadataProvider, artifact);
    } catch (IOException e) {
      return null;
    }
  }

  // TODO(ulfjack): It's unclear to me why we're ignoring all IOExceptions. In some cases, we want
  // to trigger a re-execution, so we should catch the IOException explicitly there. In others, we
  // should propagate the exception, because it is unexpected (e.g., bad file system state).
  @Nullable
  private static FileArtifactValue getOutputMetadataMaybe(
      OutputMetadataStore outputMetadataStore, Artifact artifact) throws InterruptedException {
    checkArgument(!artifact.isTreeArtifact());
    try {
      return getOutputMetadataOrConstant(outputMetadataStore, artifact);
    } catch (IOException e) {
      return null;
    }
  }

  @Nullable
  private static TreeArtifactValue getOutputTreeMetadataMaybe(
      OutputMetadataStore outputMetadataStore, Artifact artifact) throws InterruptedException {
    checkArgument(artifact.isTreeArtifact());
    try {
      return outputMetadataStore.getTreeArtifactValue((SpecialArtifact) artifact);
    } catch (IOException e) {
      return null;
    }
  }

  public void updateActionCache(
      Action action,
      Token token,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore,
      Map<String, String> clientEnv,
      OutputPermissions outputPermissions,
      ImmutableMap<String, String> remoteDefaultPlatformProperties,
      boolean useArchivedTreeArtifacts)
      throws IOException, InterruptedException {
    checkState(cacheConfig.enabled(), "cache unexpectedly disabled, action: %s", action);
    Preconditions.checkArgument(token != null, "token unexpectedly null, action: %s", action);
    String key = token.cacheKey;
    if (actionCache.get(key) != null) {
      // This cache entry has already been updated by a shared action. We don't need to do it again.
      return;
    }
    ImmutableMap<String, String> effectiveEnvironment =
        computeEffectiveEnvironment(action, clientEnv);
    ImmutableMap<String, String> effectiveExecProperties =
        computeEffectiveExecProperties(action, remoteDefaultPlatformProperties);

    // We may already have the action key stored in the token if there was a previous (but out of
    // date) cache entry for this action. If not, there's no need to store the action key in the
    // token since we won't need it again.
    String actionKey = token.actionKey;
    if (actionKey == null) {
      actionKey = action.getKey(actionKeyContext, inputMetadataProvider);
    }

    var builder =
        new ActionCache.Entry.Builder(
            actionKey,
            action.discoversInputs(),
            effectiveEnvironment,
            effectiveExecProperties,
            outputPermissions,
            useArchivedTreeArtifacts);

    for (Artifact output : action.getOutputs()) {
      // Remove old records from the cache if they used different key.
      String execPath = output.getExecPathString();
      if (!key.equals(execPath)) {
        actionCache.remove(execPath);
      }
      if (!outputMetadataStore.artifactOmitted(output)) {
        if (output.isTreeArtifact()) {
          SpecialArtifact parent = (SpecialArtifact) output;
          TreeArtifactValue metadata = outputMetadataStore.getTreeArtifactValue(parent);
          builder.addOutputTree(parent, metadata, cacheConfig.storeOutputMetadata());
        } else {
          // Output files *must* exist and be accessible after successful action execution. We use
          // the 'constant' metadata for the volatile workspace status output. The volatile output
          // contains information such as timestamps, and even when --stamp is enabled, we don't
          // want to rebuild everything if only that file changes.
          FileArtifactValue metadata = getOutputMetadataOrConstant(outputMetadataStore, output);
          checkState(metadata != null);
          builder.addOutputFile(output, metadata, cacheConfig.storeOutputMetadata());
        }
      }
    }

    boolean storeAllInputsInActionCache =
        action instanceof ActionCacheAwareAction actionCacheAwareAction
            && actionCacheAwareAction.storeInputsExecPathsInActionCache();
    ImmutableSet<Artifact> excludePathsFromActionCache =
        !storeAllInputsInActionCache && action.discoversInputs()
            ? action.getMandatoryInputs().toSet()
            : ImmutableSet.of();

    for (Artifact input : action.getInputs().toList()) {
      builder.addInputFile(
          input,
          getInputMetadataMaybe(inputMetadataProvider, input),
          /* saveExecPath= */ !excludePathsFromActionCache.contains(input));
    }

    actionCache.put(key, builder.build());
  }

  @Nullable
  public List<Artifact> getCachedInputs(Action action, PackageRootResolver resolver)
      throws PackageRootResolver.PackageRootException, InterruptedException {
    ActionCache.Entry entry = getCacheEntry(action);
    if (entry == null || entry.isCorrupted()) {
      return ImmutableList.of();
    }

    List<PathFragment> outputs = new ArrayList<>();
    for (Artifact output : action.getOutputs()) {
      outputs.add(output.getExecPath());
    }
    List<PathFragment> inputExecPaths = new ArrayList<>();
    if (entry.discoversInputs()) {
      for (String path : entry.getDiscoveredInputPaths()) {
        PathFragment execPath = PathFragment.create(path);
        // Code assumes that action has only 1-2 outputs and ArrayList.contains() will be most
        // efficient.
        if (!outputs.contains(execPath)) {
          inputExecPaths.add(execPath);
        }
      }
    }

    // Note that this method may trigger a violation of the desirable invariant that getInputs()
    // is a superset of getMandatoryInputs(). See bug about an "action not in canonical form"
    // error message and the integration test test_crosstool_change_and_failure().
    Map<PathFragment, Artifact> allowedDerivedInputsMap = new HashMap<>();
    for (Artifact derivedInput : action.getAllowedDerivedInputs().toList()) {
      if (!derivedInput.isSourceArtifact()) {
        allowedDerivedInputsMap.put(derivedInput.getExecPath(), derivedInput);
      }
    }

    ImmutableList.Builder<Artifact> inputArtifactsBuilder = ImmutableList.builder();
    List<PathFragment> unresolvedPaths = new ArrayList<>();
    for (PathFragment execPath : inputExecPaths) {
      Artifact artifact = allowedDerivedInputsMap.get(execPath);
      if (artifact != null) {
        inputArtifactsBuilder.add(artifact);
      } else {
        // Remember this execPath, we will try to resolve it as a source artifact.
        unresolvedPaths.add(execPath);
      }
    }

    Map<PathFragment, SourceArtifact> resolvedArtifacts =
        artifactResolver.resolveSourceArtifacts(unresolvedPaths, resolver);
    if (resolvedArtifacts == null) {
      // We are missing some dependencies. We need to rerun this update later.
      return null;
    }

    for (PathFragment execPath : unresolvedPaths) {
      Artifact artifact = resolvedArtifacts.get(execPath);
      // If PathFragment cannot be resolved into the artifact, ignore it. This could happen if the
      // rule has changed and the action no longer depends on, e.g., an additional source file in a
      // separate package and that package is no longer referenced anywhere else. It is safe to
      // ignore such paths because dependency checker would identify changes in inputs (ignored path
      // was used before) and will force action execution.
      if (artifact != null) {
        inputArtifactsBuilder.add(artifact);
      }
    }
    return inputArtifactsBuilder.build();
  }

  /**
   * Only call if action requires execution because there was a failure to record action cache hit
   */
  public Token getTokenUnconditionallyAfterFailureToRecordActionCacheHit(
      Action action,
      List<Artifact> resolvedCacheArtifacts,
      Map<String, String> clientEnv,
      OutputPermissions outputPermissions,
      EventHandler handler,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore,
      ImmutableMap<String, String> remoteDefaultPlatformProperties,
      @Nullable OutputChecker outputChecker,
      boolean useArchivedTreeArtifacts)
      throws InterruptedException {
    if (action != null) {
      removeCacheEntry(action);
    }
    return getTokenIfNeedToExecute(
        action,
        resolvedCacheArtifacts,
        clientEnv,
        outputPermissions,
        handler,
        inputMetadataProvider,
        outputMetadataStore,
        remoteDefaultPlatformProperties,
        outputChecker,
        useArchivedTreeArtifacts);
  }

  /**
   * In most cases, this method should not be called directly - reportXXX() methods should be used
   * instead. This is done to avoid cost associated with building the message.
   */
  private static void reportRebuild(@Nullable EventHandler handler, Action action, String message) {
    // For RunfilesTreeAction, do not report rebuild.
    if (handler != null) {
      handler.handle(
          Event.of(
              EventKind.DEPCHECKER,
              null,
              "Executing " + action.prettyPrint() + ": " + message + "."));
    }
  }

  private static void reportUnconditionalExecution(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "unconditional execution is requested");
  }

  private static void reportDigestMismatch(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "action changed since cached execution");
  }

  private static void reportNewAction(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "no entry in the cache (action is new)");
  }

  private static void reportCorruptedCacheEntry(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "cache entry is corrupted");
  }

  /** Wrapper for all context needed by the ActionCacheChecker to handle a single action. */
  public static final class Token {

    /** The primary output's path, used as the key for {@link ActionCache} . */
    private final String cacheKey;

    /** The result of calling {@link Action#getKey}, or {@code null} if it was not called. */
    @Nullable private String actionKey;

    private Token(Action action) {
      this.cacheKey = action.getPrimaryOutput().getExecPathString();
    }
  }

}
