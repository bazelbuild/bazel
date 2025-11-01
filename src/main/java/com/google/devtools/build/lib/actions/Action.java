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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ConditionallyThreadCompatible;
import com.google.devtools.build.lib.vfs.BulkDeleter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;
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
 *
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
 *       such as {@link com.google.devtools.build.lib.analysis.config.BuildConfigurationValue}
 *       objects or even fragments thereof; if the action has a command line, err on the side of
 *       hashing the entire command line, even if that seems expensive. It's always safe to hash too
 *       much - the negative effect on incremental build times is usually negligible.
 *   <li>Add test coverage for the cache key computation; use {@link
 *       com.google.devtools.build.lib.analysis.util.ActionTester} to generate as many combinations
 *       of field values as possible; add test coverage every time you add another field.
 * </ul>
 *
 * <p>These constraints are not easily enforced or tested for (e.g., ActionTester only checks that a
 * known set of fields is covered, not that all fields are covered), so carefully check all changes
 * to action subclasses.
 */
public interface Action extends ActionExecutionMetadata {
  /**
   * Prepares for executing this action; called by the Builder prior to executing the Action itself.
   * This method should prepare the file system, so that the execution of the Action can write the
   * output files. At a minimum any pre-existing and write protected output files should be removed
   * or the permissions should be changed, so that they can be safely overwritten by the action.
   *
   * @throws IOException if there is an error deleting the outputs.
   * @throws InterruptedException if the execution is interrupted
   */
  void prepare(
      Path execRoot,
      ArtifactPathResolver pathResolver,
      @Nullable BulkDeleter bulkDeleter,
      boolean cleanupArchivedArtifacts)
      throws IOException, InterruptedException;

  /**
   * Executes this action. This method <i>unconditionally does the work of the Action</i>, although
   * it may delegate some of that work to {@link ActionContext} instances obtained from the {@link
   * ActionExecutionContext}, which may in turn perform caching at smaller granularity than an
   * entire action.
   *
   * <p>This method may not be invoked if an equivalent action (as determined by the hashes of the
   * input files, the list of output files, and the action cache key) has been previously executed,
   * possibly on another machine.
   *
   * <p>The framework guarantees that:
   *
   * <ul>
   *   <li>all declared inputs have already been successfully created,
   *   <li>the output directory for each file in <code>getOutputs()</code> has already been created,
   *   <li>this method is only called by at most one thread at a time, but subsequent calls may be
   *       made from different threads,
   *   <li>for shared actions, at most one instance is executed per build.
   * </ul>
   *
   * <p>Multiple instances of the same action implementation may be called in parallel.
   * Implementations must therefore be thread-compatible. Also see the class documentation for
   * additional invariants.
   *
   * <p>Implementations should attempt to detect interrupts, and exit quickly with an {@link
   * InterruptedException}.
   *
   * @param actionExecutionContext services in the scope of the action, like the output and error
   *     streams to use for messages arising during action execution
   * @return returns an ActionResult containing action execution metadata
   * @throws ActionExecutionException if execution fails for any reason
   * @throws InterruptedException if the execution is interrupted
   */
  @ConditionallyThreadCompatible
  ActionResult execute(ActionExecutionContext actionExecutionContext)
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
   * Runs input discovery on this action.
   *
   * <p>May only be called if {@link #discoversInputs} returns true. Returns the set of input
   * artifacts that were not known at analysis time. May also call {@link #updateInputs}; if it
   * doesn't, the action itself must arrange for the newly discovered artifacts to be available
   * during action execution, probably by keeping state in the action instance and using a custom
   * action execution context and for {@link #updateInputs} to be called during the execution of the
   * action.
   *
   * <p>Since keeping state within an action is bad, don't do that unless there is a very good
   * reason to do so.
   *
   * <p>May return {@code null} if more dependencies were requested from skyframe but were
   * unavailable, meaning a restart is necessary.
   */
  @Nullable
  NestedSet<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException;

  /** Prepare for input discovery, called before the first call to {@link #discoverInputs}. */
  default void prepareInputDiscovery() {}

  /**
   * Resets this action's inputs to a pre {@linkplain #discoverInputs input discovery} state.
   *
   * <p>This may be called on input-discovering actions during non-incremental builds, when it is
   * not worthwhile to retain the discovered inputs after the action completes execution. It may
   * still be necessary to rewind the action, so it must retain state necessary for re-execution.
   */
  void resetDiscoveredInputs();

  /**
   * Returns the set of artifacts that can possibly be inputs. It will be called iff {@link
   * #inputsKnown} is false for the given action instance and there is a related cache entry in the
   * action cache.
   *
   * <p>Method must be redefined for any action for which {@link #inputsKnown} may return false.
   *
   * <p>The method is allowed to return source artifacts. They are useless, though, since exec paths
   * in the action cache referring to source artifacts are always resolved.
   */
  NestedSet<Artifact> getAllowedDerivedInputs();

  @Nullable Artifact getInputDiscoveryInvalidationArtifact();

  /**
   * Called on {@linkplain #discoversInputs input-discovering} actions when the inputs of the action
   * become known, either during {@link #discoverInputs} or during {@link #execute}.
   *
   * <p>When an action discovers inputs, this method must have been called by the time {@code
   * #execute} returns.
   *
   * <p>In addition to being called from action implementations, it is also called by {@link
   * ActionCacheChecker} when an action is loaded from the on-disk action cache.
   */
  void updateInputs(NestedSet<Artifact> inputs);

  /**
   * Returns true if the output should bypass output filtering. This is used for test actions.
   */
  boolean showsOutputUnconditionally();

  /**
   * Called by {@link com.google.devtools.build.lib.analysis.extra.ExtraAction} at execution time to
   * extract information from this action into a protocol buffer to be used by extra_action rules.
   *
   * <p>As this method is called from the ExtraAction, make sure it is ok to call this method from a
   * different thread than the one this action is executed on.
   */
  ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException, InterruptedException;

  /**
   * Called by {@link com.google.devtools.build.lib.analysis.actions.StarlarkAction} to use its
   * shadowed action, if any, complete list of environment variables in the Starlark action Spawn.
   *
   * <p>As this method is called from the StarlarkAction, make sure it is ok to call it from a
   * different thread than the one this action is executed on. By definition, the method should not
   * mutate any of the called action data but if necessary, its implementation must synchronize any
   * accesses to mutable data.
   */
  ImmutableMap<String, String> getEffectiveEnvironment(Map<String, String> clientEnv)
      throws CommandLineExpansionException;
}
