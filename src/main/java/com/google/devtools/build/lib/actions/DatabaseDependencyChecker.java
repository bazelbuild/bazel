// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.ArtifactMetadataCache;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Collection;
import java.util.Set;

/**
 * An implementation of {@link DependencyChecker} based on the {@link ActionCache}.
 */
@ThreadSafe
public class DatabaseDependencyChecker extends ActionCacheChecker implements DependencyChecker {

  private final ArtifactMetadataCache artifactMetadataCache;

  @VisibleForTesting
  protected DependentActionGraph dependencyGraph;

  protected ActionGraph actionGraph;

  /**
   * Some inputs might be allowed to be missing.
   */
  private final Predicate<PathFragment> allowedMissingInputs;

  /**
   * Creates dependency checker instance.
   */
  public DatabaseDependencyChecker(ActionCache actionCache,
      ArtifactResolver artifactResolver,
      ArtifactMetadataCache artifactMetadataCache,
      PackageUpToDateChecker packageUpToDateChecker,
      Predicate<? super Action> executionFilter,
      boolean verboseExplanations,
      Predicate<PathFragment> allowedMissingInputs) {
    super(actionCache, artifactResolver, packageUpToDateChecker, executionFilter,
        verboseExplanations);
    this.artifactMetadataCache = artifactMetadataCache;
    this.dependencyGraph = null;
    this.actionGraph = null;
    this.allowedMissingInputs = allowedMissingInputs;
  }

  @Override
  @SuppressWarnings("unused") // Derived classes' implementation may throw.
  public void init(Set<Artifact> topLevelArtifacts, Set<Artifact> builtArtifacts,
      DependentActionGraph forwardGraph, Executor executor, ModifiedFileSet modified,
      EventHandler eventHandler) throws InterruptedException {
    Preconditions.checkState(builtArtifacts == null || builtArtifacts.isEmpty());
    this.dependencyGraph = Preconditions.checkNotNull(forwardGraph);
    this.actionGraph = Preconditions.checkNotNull(forwardGraph.getActionGraph());
  }

  @Override
  public DependentActionGraph getActionGraphForBuild() {
    return dependencyGraph;
  }

  @Override
  public long getWorkSavedByDependencyChecker() {
    return 0;
  }

  // Note: the listener should only be used for DEPCHECKER events; there's no
  // guarantee it will be available for other events.
  @Override
  public Token needToExecute(Action action, EventHandler handler) {
    // Set inputs known flag now, since it may change after we have checked the action cache.
    boolean inputsKnown = action.inputsKnown();
    Token key = super.needToExecute(action, handler, artifactMetadataCache);
    if (key != null) {
      // The action "knew" its inputs, but might "relearn" them during execution.
      if (inputsKnown && action.discoversInputs()) {
        dependencyGraph.beforeChange(action);
      }

      // Immediately store input metadata. This is needed for correct rebuilds when something
      // modifies its input files during action execution.
      artifactMetadataCache.cacheMetadata(action.getInputs());
      return key;
    } else if (!action.inputsKnown()) {
      // This situation should never occur now since calculating the exact list of inputs is
      // dependent on the outputs, so instead of reaching this point user should get:
      // a) no entry in the cache (first run)
      // b) output file (usually the *.d file) is new, changed or missing
      throw new IllegalStateException(
          action.prettyPrint() + " does not have a known list "
          + "of inputs yet all other validations have passed.");
    }
    return null;
  }


  @Override
  public void afterExecution(Action action, Token token) throws IOException {
    super.afterExecution(action, token, getMetadataHandler());
  }

  @Override
  protected boolean updateActionInputs(Action action, ActionCache.Entry entry) {
    dependencyGraph.beforeChange(action);
    if (super.updateActionInputs(action, entry)) {
      artifactMetadataCache.cacheMetadata(action.getInputs());
      return true;
    }
    return false;
  }

  @Override
  public Collection<Artifact> getMissingInputs(Action action) {
    artifactMetadataCache.beforeRetrieval(action.getInputs());

    // Since this is executed per-action and is on the critical path,
    // don't undergo the cost of object creation unless missing artifacts exist.
    Collection<Artifact> missingInputs = null;
    for (Artifact input : action.getMandatoryInputs()) {
      // For generated inputs we assume that they exist (and are up-to-date)
      // since we validated that earlier in the build. Some inputs are allowed to
      // be missing, see ConfiguredRuleClassProvider.setAllowedMissingInputs().
      if (input.isSourceArtifact() && !artifactExists(input) &&
          !allowedMissingInputs.apply(input.getExecPath())) {
        if (missingInputs == null) {
          missingInputs = Lists.newArrayList();
        }
        missingInputs.add(input);
      }
    }
    return missingInputs == null ? ImmutableList.<Artifact>of() : missingInputs;
  }

  @Override
  public boolean artifactExists(Artifact artifact) {
    return artifactMetadataCache.artifactExists(artifact);
  }

  @Override
  public boolean isInjected(Artifact artifact) throws IOException {
    return artifactMetadataCache.isInjected(artifact);
  }

  @Override
  public MetadataHandler getMetadataHandler() {
    return artifactMetadataCache;
  }
}
