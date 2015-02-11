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

import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.profiler.Describable;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Collection;

import javax.annotation.Nullable;

/**
 * An Action represents a function from Artifacts to Artifacts executed as an
 * atomic build step.  Examples include compilation of a single C++ source
 * file, or linking a single library.
 */
public interface Action extends ActionMetadata, Describable {

  /**
   * Prepares for executing this action; called by the Builder prior to
   * executing the Action itself. This method should prepare the file system, so
   * that the execution of the Action can write the output files. At a minimum
   * any pre-existing and write protected output files should be removed or the
   * permissions should be changed, so that they can be safely overwritten by
   * the action.
   *
   * @throws IOException if there is an error deleting the outputs.
   */
  void prepare(Path execRoot) throws IOException;

  /**
   * Executes this action; called by the Builder when all of this Action's
   * inputs have been successfully created.  (Behaviour is undefined if the
   * prerequisites are not up to date.)  This method <i>actually does the work
   * of the Action, unconditionally</i>; in other words, it is invoked by the
   * Builder only when dependency analysis has deemed it necessary.</p>
   *
   * <p>The framework guarantees that the output directory for each file in
   * <code>getOutputs()</code> has already been created, and will check to
   * ensure that each of those files is indeed created.</p>
   *
   * <p>Implementations of this method should try to honour the {@link
   * java.lang.Thread#interrupted} contract: if an interrupt is delivered to
   * the thread in which execution occurs, the action should detect this on a
   * best-effort basis and terminate as quickly as possible by throwing an
   * ActionExecutionException.
   *
   * <p>Action execution must be ThreadCompatible in order to be safely used
   * with a concurrent Builder implementation such as ParallelBuilder.
   *
   * @param actionExecutionContext Services in the scope of the action, like the output and error
   *   streams to use for messages arising during action execution.
   * @throws ActionExecutionException if execution fails for any reason.
   * @throws InterruptedException
   */
  @ConditionallyThreadCompatible
  void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /**
   * Returns true iff action must be executed regardless of its current state.
   * Default implementation can be overridden by some actions that might be
   * executed unconditionally under certain circumstances - e.g., if caching of
   * test results is not requested, this method could be used to force test
   * execution even if all dependencies are up-to-date.
   *
   * <p>Note, it is <b>very</b> important not to abuse this method, since it
   * completely overrides dependency checking. Any use of this method must
   * be carefully reviewed and proved to be necessary.
   *
   * <p>Note that the definition of {@link #isVolatile} depends on the
   * definition of this method, so be sure to consider both methods together
   * when making changes.
   */
  boolean executeUnconditionally();

  /**
   * Returns true if it's ever possible that {@link #executeUnconditionally}
   * could evaluate to true during the lifetime of this instance, false
   * otherwise.
   */
  boolean isVolatile();

  /**
   * Method used to find inputs before execution for an action that
   * {@link ActionMetadata#discoversInputs}.
   */
  void discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /**
   * Method used to update action inputs based on the information contained in
   * the action cache. It will be called iff inputsKnown() is false for the
   * given action instance and there is a related cache entry in the action
   * cache.
   *
   * Method must be redefined for any action that may return
   * inputsKnown() == false. It also expects that implementation will ensure
   * that inputsKnown() returns true after call to this method.
   *
   * @param artifactResolver the artifact factory that can be used to manufacture artifacts
   * @param inputPaths List of relative (to the execution root) input paths
   * @param resolver object which helps to resolve some of the artifacts
   * @return false if some dependencies are missing and we need to update again later,
   * otherwise true.
   */
  boolean updateInputsFromCache(
      ArtifactResolver artifactResolver, PackageRootResolver resolver,
      Collection<PathFragment> inputPaths);

  /**
   * Return a best-guess estimate of the operation's resource consumption on the
   * local host itself for use in scheduling.
   *
   * @param executor the application-specific value passed to the
   *   executor parameter of the top-level call to
   *   Builder.buildArtifacts().
   */
  @Nullable ResourceSet estimateResourceConsumption(Executor executor);

  /**
   * @return true iff path prefix conflict (conflict where two actions generate
   *         two output artifacts with one of the artifact's path being the
   *         prefix for another) between this action and another action should
   *         be reported.
   */
  boolean shouldReportPathPrefixConflict(Action action);

  /**
   * Returns true if the output should bypass output filtering. This is used for test actions.
   */
  boolean showsOutputUnconditionally();

  /**
   * Called by {@link com.google.devtools.build.lib.rules.extra.ExtraAction} at execution time to
   * extract information from this action into a protocol buffer to be used by extra_action rules.
   *
   * <p>As this method is called from the ExtraAction, make sure it is ok to call this method from
   * a different thread than the one this action is executed on.
   */
  ExtraActionInfo.Builder getExtraActionInfo();

  /**
   * Returns the action type. Must not be {@code null}.
   */
  MiddlemanType getActionType();

  /**
   * The action type.
   */
  public enum MiddlemanType {

    /** A normal action. */
    NORMAL,

    /** A normal middleman, which just encapsulates a list of artifacts. */
    AGGREGATING_MIDDLEMAN,

    /**
     * A middleman that enforces action ordering, is not validated by the dependency checker, but
     * allows errors to be propagated.
     */
    ERROR_PROPAGATING_MIDDLEMAN,

    /**
     * A runfiles middleman, which is validated by the dependency checker, but is not expanded
     * in blaze. Instead, the runfiles manifest is sent to remote execution client, which
     * performs the expansion.
     */
    RUNFILES_MIDDLEMAN;

    public boolean isMiddleman() {
      return this != NORMAL;
    }
  }
}
