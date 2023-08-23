// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/*
 * TODO(b/236456122): We should break this class up into separate classes for each kind of Starlark
 * evaluation environment, as opposed to storing possibly inapplicable nullable fields on this
 * class. Then we can use static methods with signatures like fromOrFail(StarlarkThread, String) in
 * place of the Phase enum and its associated check* methods.
 *
 * Some kinds of evaluation environments include:
 *   - .bzl loading
 *   - BUILD evaluation (should absorb PackageFactory.PackageContext -- no need to store multiple
 *     thread-locals in StarlarkThread)
 *   - WORKSPACE evaluation (shares logic with BUILD)
 *   - rule and aspect analysis implementation (can/should we store the RuleContext here?)
 *   - implicit outputs
 *   - computed defaults
 *   - transition implementation
 *   - Args.map_each
 *   - probably others
 *
 * BazelStarlarkContext itself could probably be deleted. The only thing common to almost all Bazel
 * Starlark environments is SymbolGenerator, which in principle could be used to uniquely identify
 * depsets or even StarlarkFunction, though in practice it doesn't have nearly that widespread usage
 * yet. SymbolGenerator could be promoted to a core feature of StarlarkThread if it reaches that
 * point. If we do keep BazelStarlarkContext around as a common base class of the other context
 * classes, it should be renamed BazelThreadContext for symmetry with BazelModuleContext.
 *
 * Even if we can otherwise get rid of BazelStarlarkContext, it may still be handy to retain this
 * class as a static namespace for helper methods for storing and retrieving contexts on
 * StarlarkThreads. In particular, it'd avoid the very likely bug of a storeInThread
 * implementation in one of the context classes forgetting to setUncheckedExceptionContext(this). It
 * also would give us a place to keep javadoc about the proper way to use these context objects in
 * Bazel.
 */
/**
 * Bazel-specific contextual information associated with a Starlark evaluation thread.
 *
 * <p>This is stored in the StarlarkThread object as a thread-local. A subclass of this class may be
 * used for certain kinds of Starlark evaluations; in that case it is still keyed in the
 * thread-locals under {@code BazelStarlarkContext.class}.
 *
 * <p>This object is mutable and should not be accessed simultaneously or reused for more than one
 * Starlark thread.
 */
public class BazelStarlarkContext implements StarlarkThread.UncheckedExceptionContext {

  /** The phase to which this Starlark thread belongs. */
  // TODO(b/236456122): Eliminate.
  public enum Phase {
    WORKSPACE,
    LOADING,
    ANALYSIS
  }

  /**
   * Retrieves this context from a Starlark thread, or throws {@link IllegalStateException} if
   * unavailable.
   */
  public static BazelStarlarkContext from(StarlarkThread thread) {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    // ISE rather than NPE for symmetry with subclasses.
    Preconditions.checkState(
        ctx != null, "Expected BazelStarlarkContext to be available in this Starlark thread");
    return ctx;
  }

  /**
   * Saves this {@code BazelStarlarkContext} in the specified Starlark thread. Call only once,
   * before evaluation begins.
   */
  public void storeInThread(StarlarkThread thread) {
    Preconditions.checkState(thread.getThreadLocal(BazelStarlarkContext.class) == null);
    thread.setThreadLocal(BazelStarlarkContext.class, this);
    // TODO(b/236456122): We can probably replace the concept of setUncheckedExceptionContext with a
    // string parameter to StarlarkThread construction. In that case, storeInThread() becomes more
    // superfluous, and there's one less reason to keep the inheritance hierarchy of
    // BazelStarlarkContext and its children.
    thread.setUncheckedExceptionContext(this);
  }

  // A generic counter for uniquely identifying symbols created in this Starlark evaluation.
  private final SymbolGenerator<?> symbolGenerator;

  // TODO(b/236456122): Eliminate Phase, migrate analysisRuleLabel to a separate context class.
  private final Phase phase;
  @Nullable private final Label analysisRuleLabel;

  /**
   * @param phase the phase to which this Starlark thread belongs
   * @param symbolGenerator a {@link SymbolGenerator} to be used when creating objects to be
   *     compared using reference equality.
   * @param analysisRuleLabel is the label of the rule for an analysis phase (rule or aspect
   */
  // TODO(b/236456122): Consider taking an owner in place of a SymbolGenerator, and constructing
  // the latter ourselves. Seems like we don't want to tempt anyone into sharing a SymbolGenerator.
  public BazelStarlarkContext(
      Phase phase, SymbolGenerator<?> symbolGenerator, @Nullable Label analysisRuleLabel) {
    this.phase = Preconditions.checkNotNull(phase);
    this.symbolGenerator = Preconditions.checkNotNull(symbolGenerator);
    this.analysisRuleLabel = analysisRuleLabel;
  }

  /** Returns the phase associated with this context. */
  public Phase getPhase() {
    return phase;
  }

  public SymbolGenerator<?> getSymbolGenerator() {
    return symbolGenerator;
  }

  /**
   * Returns the label of the rule, if this is an analysis-phase (rule or aspect 'implementation')
   * thread, or null otherwise.
   */
  @Nullable
  public Label getAnalysisRuleLabel() {
    return analysisRuleLabel;
  }

  @Override
  public String getContextForUncheckedException() {
    return firstNonNull(analysisRuleLabel, phase).toString();
  }

  /**
   * Checks that the Starlark thread is in the loading or the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  // TODO(b/236456122): The Phase enum is incomplete. Ex: `Args.map_each` evaluation happens at
  // execution time. So this is a misnomer and possibly wrong in those contexts.
  public void checkLoadingOrWorkspacePhase(String function) throws EvalException {
    if (phase == Phase.ANALYSIS) {
      throw Starlark.errorf("'%s' cannot be called during the analysis phase", function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the loading phase.
   *
   * @param function name of a function that requires this check
   */
  public void checkLoadingPhase(String function) throws EvalException {
    if (phase != Phase.LOADING) {
      throw Starlark.errorf("'%s' can only be called during the loading phase", function);
    }
  }

  /**
   * Checks that the current StarlarkThread is in the workspace phase.
   *
   * @param function name of a function that requires this check
   */
  public void checkWorkspacePhase(String function) throws EvalException {
    if (phase != Phase.WORKSPACE) {
      throw Starlark.errorf("'%s' can only be called during workspace loading", function);
    }
  }
}
