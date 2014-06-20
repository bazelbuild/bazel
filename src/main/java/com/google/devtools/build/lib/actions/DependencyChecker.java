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

import com.google.devtools.build.lib.actions.ActionCacheChecker.DepcheckerListener;
import com.google.devtools.build.lib.actions.ActionCacheChecker.Token;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;

import java.io.IOException;
import java.util.Collection;
import java.util.Set;

/**
 * A DependencyChecker encapsulates a particular dependency-checking policy.
 */
@ThreadCompatible // Note: NOT ThreadSafe
public interface DependencyChecker {

  /**
   * Called at the beginning of the build.
   *
   * @param topLevelArtifacts the top-level artifacts to build.
   * @param builtArtifacts the set of successfully built artifacts, expected
   *     to be empty (and possibly partly filled here).
   * @param forwardGraph the graph corresponding to topLevelArtifacts.
   * @param executor an opaque application-specific value that will be
   *     passed down to the execute() method of any Action executed during
   *     this build.
   * @param modified the modified source files.
   * @param listener the error listener.
   */
  void init(Set<Artifact> topLevelArtifacts, Set<Artifact> builtArtifacts,
      DependentActionGraph forwardGraph, Executor executor, ModifiedFileSet modified,
      ErrorEventListener listener) throws InterruptedException;

  /**
   * Returns a forward action graph suitable for builder execution. This may
   * or may not be the same as the graph passed to {@link #init}.
   */
  DependentActionGraph getActionGraphForBuild();

  /** Returns the metadata handler used internally by the dependency checker. */
  MetadataHandler getMetadataHandler();

  /**
   * Returns an estimate of the workload saved for this build as a result
   * of culling from the dependency checker.
   * @return work saved, as an aggregate of {@link Action#estimateWorkload()}
   *     values for saved actions.
   */
  long getWorkSavedByDependencyChecker();

  /**
   * Returns a collection of missing mandatory input files.
   *
   * <p>The returned files should already exist (at the point of the build where this method is
   * called), either because they are source files or because their generating actions have
   * already been executed.
   *
   * @param action the Action to check.
   * @return a collection of missing mandatory input files, possibly empty and possibly immutable.
   */
  Collection<Artifact> getMissingInputs(Action action);

  /**
   * Check if the action needs to be executed.
   *
   * <p>Precondition: we must have already built all of the input files for this
   * action.
   *
   * @param action the Action for which dependency checking is required.
   * @param listener an optional DepcheckerListener via which the DependencyChecker may
   *   report DEPCHECKER events.
   * @return a non-null value iff the action needs to be executed, or null
   *   otherwise.  This value is opaque to the caller, and must be passed to
   *   the subsequent calls to afterExecution for this action.
   */
  Token needToExecute(Action action, DepcheckerListener listener) throws IOException;

  /**
   * This method is called immediately after the action has been executed.
   * It can for example be used to record the file system metadata
   * associated with the output files (e.g. their timestamps).
   * The specified token must have been returned by the previous call to
   * needToExecute for this action.
   *
   * @param action the action that has just been executed.
   * @param token the token returned by needToExecute(action) immediately prior
   *   to this action's execution.
   */
  void afterExecution(Action action, Token token) throws IOException;

  /**
   * Equivalent of the artifact.getPath().exists() except it might utilize
   * internal metadata caches to query artifact existence.
   *
   * @return true iff artifact does exist.
   */
  boolean artifactExists(Artifact artifact);

  /**
   * @return Whether the artifact's data was injected (from remote execution).
   * @throws IOException if statting artifact threw an exception.
   */
  boolean isInjected(Artifact artifact) throws IOException;

  /**
   * @return true if action execution is prohibited by the execution filter,
   *         false - otherwise.
   */
  boolean isActionExecutionProhibited(Action action);
}
