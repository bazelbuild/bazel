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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.ActionCache.Entry;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Checks whether an {@link Action} needs to be executed, or whether it has not changed since it was
 * last stored in the action cache. Must be informed of the new Action data after execution as well.
 *
 * <p>The fingerprint, input files names, and metadata (either mtimes or MD5sums) of each action are
 * cached in the action cache to avoid unnecessary rebuilds. Middleman artifacts are handled
 * specially, avoiding the need to create actual files corresponding to the middleman artifacts.
 * Instead of that, results of MiddlemanAction dependency checks are cached internally and then
 * reused whenever an input middleman artifact is encountered.
 *
 * <p>While instances of this class hold references to action and metadata cache instances, they are
 * otherwise lightweight, and should be constructed anew and discarded for each build request.
 */
public class ActionCacheChecker {
  private final ActionCache actionCache;
  private final Predicate<? super Action> executionFilter;
  private final ArtifactResolver artifactResolver;
  // True iff --verbose_explanations flag is set.
  private final boolean verboseExplanations;

  public ActionCacheChecker(ActionCache actionCache, ArtifactResolver artifactResolver,
      Predicate<? super Action> executionFilter, boolean verboseExplanations) {
    this.actionCache = actionCache;
    this.executionFilter = executionFilter;
    this.artifactResolver = artifactResolver;
    this.verboseExplanations = verboseExplanations;
  }

  public boolean isActionExecutionProhibited(Action action) {
    return !executionFilter.apply(action);
  }

  /**
   * Checks whether one of existing output paths is already used as a key.
   * If yes, returns it - otherwise uses first output file as a key
   */
  private ActionCache.Entry getCacheEntry(Action action) {
    for (Artifact output : action.getOutputs()) {
      ActionCache.Entry entry = actionCache.get(output.getExecPathString());
      if (entry != null) {
        return entry;
      }
    }
    return null;
  }

  private void removeCacheEntry(Action action) {
    for (Artifact output : action.getOutputs()) {
      actionCache.remove(output.getExecPathString());
    }
  }

  /**
   * Validate metadata state for action input or output artifacts.
   *
   * @param entry cached action information.
   * @param action action to be validated.
   * @param actionInputs the inputs of the action. Normally just the result of action.getInputs(),
   * but if this action doesn't yet know its inputs, we check the inputs from the cache.
   * @param metadataHandler provider of metadata for the artifacts this action interacts with.
   * @param checkOutput true to validate output artifacts, Otherwise, just
   *                    validate inputs.
   *
   * @return true if at least one artifact has changed, false - otherwise.
   */
  private boolean validateArtifacts(Entry entry, Action action,
      Iterable<Artifact> actionInputs, MetadataHandler metadataHandler, boolean checkOutput) {
    Iterable<Artifact> artifacts = checkOutput
        ? Iterables.concat(action.getOutputs(), actionInputs)
        : actionInputs;
    Map<String, Metadata> mdMap = new HashMap<>();
    for (Artifact artifact : artifacts) {
      mdMap.put(artifact.getExecPathString(), metadataHandler.getMetadataMaybe(artifact));
    }
    return !Digest.fromMetadata(mdMap).equals(entry.getFileDigest());
  }

  private void reportCommand(EventHandler handler, Action action) {
    if (handler != null) {
      if (verboseExplanations) {
        String keyDescription = action.describeKey();
        reportRebuild(handler, action, keyDescription == null
            ? "action command has changed"
            : "action command has changed.\nNew action: " + keyDescription);
      } else {
        reportRebuild(handler, action,
            "action command has changed (try --verbose_explanations for more info)");
      }
    }
  }

  protected boolean unconditionalExecution(Action action) {
    return !isActionExecutionProhibited(action) && action.executeUnconditionally();
  }

  /**
   * Checks whether {@code action} needs to be executed and returns a non-null Token if so.
   *
   * <p>The method checks if any of the action's inputs or outputs have changed. Returns a non-null
   * {@link Token} if the action needs to be executed, and null otherwise.
   *
   * <p>If this method returns non-null, indicating that the action will be executed, the
   * metadataHandler's {@link MetadataHandler#discardOutputMetadata} method must be called, so that
   * it does not serve stale metadata for the action's outputs after the action is executed.
   */
  // Note: the handler should only be used for DEPCHECKER events; there's no
  // guarantee it will be available for other events.
  public Token getTokenIfNeedToExecute(Action action, Iterable<Artifact> resolvedCacheArtifacts,
      EventHandler handler, MetadataHandler metadataHandler) {
    // TODO(bazel-team): (2010) For RunfilesAction/SymlinkAction and similar actions that
    // produce only symlinks we should not check whether inputs are valid at all - all that matters
    // that inputs and outputs are still exist (and new inputs have not appeared). All other checks
    // are unnecessary. In other words, the only metadata we should check for them is file existence
    // itself.

    MiddlemanType middlemanType = action.getActionType();
    if (middlemanType.isMiddleman()) {
      // Some types of middlemen are not checked because they should not
      // propagate invalidation of their inputs.
      if (middlemanType != MiddlemanType.ERROR_PROPAGATING_MIDDLEMAN) {
        checkMiddlemanAction(action, handler, metadataHandler);
      }
      return null;
    }
    Iterable<Artifact> actionInputs = action.getInputs();
    // Resolve action inputs from cache, if necessary.
    boolean inputsKnown = action.inputsKnown();
    if (!inputsKnown && resolvedCacheArtifacts != null) {
      // The action doesn't know its inputs, but the caller has a good idea of what they are.
      Preconditions.checkState(action.discoversInputs(),
          "Actions that don't know their inputs must discover them: %s", action);
      actionInputs = resolvedCacheArtifacts;
    }
    ActionCache.Entry entry = getCacheEntry(action);
    if (mustExecute(action, entry, handler, metadataHandler, actionInputs)) {
      if (entry != null) {
        removeCacheEntry(action);
      }
      return new Token(getKeyString(action));
    }

    if (!inputsKnown) {
      action.updateInputs(actionInputs);
    }
    return null;
  }

  protected boolean mustExecute(Action action, @Nullable ActionCache.Entry entry,
      EventHandler handler, MetadataHandler metadataHandler, Iterable<Artifact> actionInputs) {
    // Unconditional execution can be applied only for actions that are allowed to be executed.
    if (unconditionalExecution(action)) {
      Preconditions.checkState(action.isVolatile());
      reportUnconditionalExecution(handler, action);
      return true; // must execute - unconditional execution is requested.
    }
    if (entry == null) {
      reportNewAction(handler, action);
      return true; // must execute -- no cache entry (e.g. first build)
    }

    if (entry.isCorrupted()) {
      reportCorruptedCacheEntry(handler, action);
      return true; // cache entry is corrupted - must execute
    } else if (validateArtifacts(entry, action, actionInputs, metadataHandler, true)) {
      reportChanged(handler, action);
      return true; // files have changed
    } else if (!entry.getActionKey().equals(action.getKey())) {
      reportCommand(handler, action);
      return true; // must execute -- action key is different
    }

    entry.getFileDigest();
    return false; // cache hit
  }

  public void afterExecution(Action action, Token token, MetadataHandler metadataHandler)
      throws IOException {
    Preconditions.checkArgument(token != null);
    String key = token.cacheKey;
    if (actionCache.get(key) != null) {
      // This cache entry has already been updated by a shared action. We don't need to do it again.
      return;
    }
    ActionCache.Entry entry =
        actionCache.createEntry(action.getKey(), action.discoversInputs());
    for (Artifact output : action.getOutputs()) {
      // Remove old records from the cache if they used different key.
      String execPath = output.getExecPathString();
      if (!key.equals(execPath)) {
        actionCache.remove(execPath);
      }
      if (!metadataHandler.artifactOmitted(output)) {
        // Output files *must* exist and be accessible after successful action execution.
        Metadata metadata = metadataHandler.getMetadata(output);
        Preconditions.checkState(metadata != null);
        entry.addFile(output.getExecPath(), metadata);
      }
    }
    for (Artifact input : action.getInputs()) {
      entry.addFile(input.getExecPath(), metadataHandler.getMetadataMaybe(input));
    }
    entry.getFileDigest();
    actionCache.put(key, entry);
  }

  @Nullable
  public Iterable<Artifact> getCachedInputs(Action action, PackageRootResolver resolver)
      throws PackageRootResolutionException {
    ActionCache.Entry entry = getCacheEntry(action);
    if (entry == null || entry.isCorrupted()) {
      return ImmutableList.of();
    }

    List<PathFragment> outputs = new ArrayList<>();
    for (Artifact output : action.getOutputs()) {
      outputs.add(output.getExecPath());
    }
    List<PathFragment> inputs = new ArrayList<>();
    for (String path : entry.getPaths()) {
      PathFragment execPath = new PathFragment(path);
      // Code assumes that action has only 1-2 outputs and ArrayList.contains() will be
      // most efficient.
      if (!outputs.contains(execPath)) {
        inputs.add(execPath);
      }
    }
    return action.resolveInputsFromCache(artifactResolver, resolver, inputs);
  }

  /**
   * Special handling for the MiddlemanAction. Since MiddlemanAction output
   * artifacts are purely fictional and used only to stay within dependency
   * graph model limitations (action has to depend on artifacts, not on other
   * actions), we do not need to validate metadata for the outputs - only for
   * inputs. We also do not need to validate MiddlemanAction key, since action
   * cache entry key already incorporates that information for the middlemen
   * and we will experience a cache miss when it is different. Whenever it
   * encounters middleman artifacts as input artifacts for other actions, it
   * consults with the aggregated middleman digest computed here.
   */
  protected void checkMiddlemanAction(Action action, EventHandler handler,
      MetadataHandler metadataHandler) {
    Artifact middleman = action.getPrimaryOutput();
    String cacheKey = middleman.getExecPathString();
    ActionCache.Entry entry = actionCache.get(cacheKey);
    boolean changed = false;
    if (entry != null) {
      if (entry.isCorrupted()) {
        reportCorruptedCacheEntry(handler, action);
        changed = true;
      } else if (validateArtifacts(entry, action, action.getInputs(), metadataHandler, false)) {
        reportChanged(handler, action);
        changed = true;
      }
    } else {
      reportChangedDeps(handler, action);
      changed = true;
    }
    if (changed) {
      // Compute the aggregated middleman digest.
      // Since we never validate action key for middlemen, we should not store
      // it in the cache entry and just use empty string instead.
      entry = actionCache.createEntry("", false);
      for (Artifact input : action.getInputs()) {
        entry.addFile(input.getExecPath(), metadataHandler.getMetadataMaybe(input));
      }
    }

    metadataHandler.setDigestForVirtualArtifact(middleman, entry.getFileDigest());
    if (changed) {
      actionCache.put(cacheKey, entry);
    }
  }

  /**
   * Returns an action key. It is always set to the first output exec path string.
   */
  private static String getKeyString(Action action) {
    Preconditions.checkState(!action.getOutputs().isEmpty());
    return action.getOutputs().iterator().next().getExecPathString();
  }


  /**
   * In most cases, this method should not be called directly - reportXXX() methods
   * should be used instead. This is done to avoid cost associated with building
   * the message.
   */
  private static void reportRebuild(@Nullable EventHandler handler, Action action, String message) {
    // For MiddlemanAction, do not report rebuild.
    if (handler != null && !action.getActionType().isMiddleman()) {
      handler.handle(new Event(
          EventKind.DEPCHECKER, null, "Executing " + action.prettyPrint() + ": " + message + "."));
    }
  }

  // Called by IncrementalDependencyChecker.
  protected static void reportUnconditionalExecution(
      @Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "unconditional execution is requested");
  }

  private static void reportChanged(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "One of the files has changed");
  }

  private static void reportChangedDeps(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "the set of files on which this action depends has changed");
  }

  private static void reportNewAction(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "no entry in the cache (action is new)");
  }

  private static void reportCorruptedCacheEntry(@Nullable EventHandler handler, Action action) {
    reportRebuild(handler, action, "cache entry is corrupted");
  }

  /** Wrapper for all context needed by the ActionCacheChecker to handle a single action. */
  public static final class Token {
    private final String cacheKey;

    private Token(String cacheKey) {
      this.cacheKey = Preconditions.checkNotNull(cacheKey);
    }
  }
}
