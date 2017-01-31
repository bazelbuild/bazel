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

import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.profiler.Describable;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * An Action represents a function from Artifacts to Artifacts executed as an atomic build step.
 * Examples include compilation of a single C++ source file, or linking a single library.
 *
 * <p>All subclasses of Action need to follow a strict set of invariants to ensure correctness on
 * incremental builds. In our experience, getting this wrong is a lot more expensive than any
 * benefits it might entail.
 *
 * <p>Use {@link com.google.devtools.build.lib.analysis.actions.SpawnAction} or {@link
 * com.google.devtools.build.lib.analysis.actions.FileWriteAction} where possible, and avoid writing
 * a new custom subclass.
 *
 * <p>These are the most important requirements for subclasses:
 * <ul>
 *   <li>Actions must be generally immutable; we currently make an exception for C++, and that has
 *       been a constant source of correctness issues; there are still ongoing incremental
 *       correctness issues for C++ compilations, despite several rounds of fixes and even though
 *       this is the oldest part of the code base.
 *   <li>Actions should be as lazy as possible - storing full lists of inputs or full command lines
 *       in every action generally results in quadratic memory consumption. Use {@link
 *       com.google.devtools.build.lib.collect.nestedset.NestedSet} for inputs, and {@link
 *       com.google.devtools.build.lib.analysis.actions.CustomCommandLine} for command lines where
 *       possible to share as much data between the different actions and their owning configured
 *       targets.
 *   <li>However, actions must not reference configured targets or rule contexts themselves; only
 *       reference the necessary underlying artifacts or strings, preferably as nested sets. Bazel
 *       may attempt to garbage collect configured targets and rule contexts before execution to
 *       keep memory consumption down, and referencing them prevents that.
 *   <li>In particular, avoid anonymous inner classes - when created in a non-static method, they
 *       implicitly keep a reference to their enclosing class, even if that reference is unnecessary
 *       for correct operation. Not doing so has caused significant increases in memory consumption
 *       in the past.
 *   <li>Correct cache key computation in {@link #getKey} is critical for the correctness of
 *       incremental builds; you may be tempted to intentionally exclude data from the cache key,
 *       but be aware that every time we've done that, it later resulted in expensive debugging
 *       sessions and bug fixes.
 *   <li>As much as possible, make the cache key computation obvious - fully hash every field
 *       (except input contents, but including input and output names if they appear in the command
 *       line) in the class, and avoid referencing anything that isn't needed for action execution,
 *       such as {@link com.google.devtools.build.lib.analysis.config.BuildConfiguration} objects or
 *       even fragments thereof; if the action has a command line, err on the side of hashing the
 *       entire command line, even if that seems expensive. It's always safe to hash too much - the
 *       negative effect on incremental build times is usually negligible.
 *   <li>Add test coverage for the cache key computation; use {@link
 *       com.google.devtools.build.lib.analysis.util.ActionTester} to generate as many combinations
 *       of field values as possible; add test coverage every time you add another field.
 * </ul>
 *
 * <p>These constraints are not easily enforced or tested for (e.g., ActionTester only checks that a
 * known set of fields is covered, not that all fields are covered), so carefully check all changes
 * to action subclasses.
 */
public interface Action extends ActionExecutionMetadata, Describable {

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
   * {@link ActionExecutionMetadata#discoversInputs}. Returns null if action's inputs will be
   * discovered during execution proper.
   */
  @Nullable
  Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /**
   * Used in combination with {@link #discoverInputs} if inputs need to be found before execution in
   * multiple steps. Returns null if two-stage input discovery isn't necessary.
   *
   * <p>Any deps requested here must not change unless one of the action's inputs changes.
   * Otherwise, changes to nodes that should cause re-execution of actions might be prevented by the
   * action cache.
   */
  @Nullable
  Iterable<Artifact> discoverInputsStage2(SkyFunction.Environment env)
      throws ActionExecutionException, InterruptedException;

  /**
   * Method used to resolve action inputs based on the information contained in the action cache. It
   * will be called iff inputsKnown() is false for the given action instance and there is a related
   * cache entry in the action cache.
   *
   * <p>Method must be redefined for any action that may return inputsKnown() == false.
   *
   * @param artifactResolver the artifact factory that can be used to manufacture artifacts
   * @param resolver object which helps to resolve some of the artifacts
   * @param inputPaths List of relative (to the execution root) input paths
   * @return List of Artifacts corresponding to inputPaths, or null if some dependencies were
   *     missing and we need to try again later.
   * @throws PackageRootResolutionException on failure to determine package roots of inputPaths
   */
  @Nullable
  Iterable<Artifact> resolveInputsFromCache(
      ArtifactResolver artifactResolver,
      PackageRootResolver resolver,
      Collection<PathFragment> inputPaths)
      throws PackageRootResolutionException, InterruptedException;

  /**
   * Informs the action that its inputs are {@code inputs}, and that its inputs are now known. Can
   * only be called for actions that discover inputs. After this method is called,
   * {@link ActionExecutionMetadata#inputsKnown} should return true.
   */
  void updateInputs(Iterable<Artifact> inputs);

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
   * Returns true if the output should bypass output filtering. This is used for test actions.
   */
  boolean showsOutputUnconditionally();

  /**
   * Returns true if an {@link com.google.devtools.build.lib.rules.extra.ExtraAction} action can be
   * attached to this action. If not, extra actions should not be attached to this action.
   */
  boolean extraActionCanAttach();

  /**
   * Called by {@link com.google.devtools.build.lib.rules.extra.ExtraAction} at execution time to
   * extract information from this action into a protocol buffer to be used by extra_action rules.
   *
   * <p>As this method is called from the ExtraAction, make sure it is ok to call this method from
   * a different thread than the one this action is executed on.
   */
  ExtraActionInfo.Builder getExtraActionInfo();
}
