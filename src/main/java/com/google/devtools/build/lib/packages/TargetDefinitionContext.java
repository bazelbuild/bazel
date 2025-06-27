// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.PackageLimits;
import com.google.devtools.build.lib.packages.Package.Metadata;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroFrame;
import com.google.devtools.build.lib.packages.TargetRecorder.NameConflictException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;

/**
 * Base class of {@link Package.Builder} and {@link PackagePiece.Builder} that encapsulates all the
 * operations that may need to occur in the middle of BUILD file or symbolic macro evaluation,
 * without including operations specific to the setup or finalization of {@code Package}
 * construction.
 *
 * <p>In other words, if a {@link Package.Builder} or and {@link PackagePiece.Builder} method needs
 * to be called as a result of Starlark evaluation of either the BUILD file or its macros, the
 * operation belongs in this base class.
 *
 * <p>The motivation for this split is two-fold: 1) It keeps the size of Package.java smaller. 2) It
 * will make it easier to factor out common code for evaluating a whole package vs an individual
 * symbolic macro of that package (lazy macro evaluation).
 */
public abstract class TargetDefinitionContext extends StarlarkThreadContext {

  // TODO: #19922 - Avoid protected fields, encapsulate with getters/setters. Temporary state on way
  // to separating this class from Package.Builder.

  private final SymbolGenerator<?> symbolGenerator;

  // Same as pkg.metadata.
  protected final Metadata metadata;

  /**
   * The {@link Package} to be constructed with the help of this context.
   *
   * <p>Since the package has not yet been constructed, it is in an intermediate state and some
   * operations may fail unexpectedly. {@code TargetDefinitionContext} only uses this field to help
   * create the cyclic links between packages and their targets.
   */
  protected final Packageoid pkg;

  // The container object on which targets and macro instances are added and conflicts are
  // detected.
  protected final TargetRecorder recorder;

  // Initialized from outside but also potentially set by `workspace()` function in WORKSPACE
  // file.
  protected String workspaceName;

  private final boolean simplifyUnconditionalSelectsInRuleAttrs;

  /** Converts label literals to Label objects within this package. */
  private final LabelConverter labelConverter;

  /**
   * Semaphore held by the Skyframe thread when performing CPU work.
   *
   * <p>This should be released when performing I/O.
   */
  @Nullable // Only non-null when inside PackageFunction.compute and the semaphore is enabled.
  private final Semaphore cpuBoundSemaphore;

  /** Estimates the cost of this packageoid. */
  protected final PackageOverheadEstimator packageOverheadEstimator;

  // TreeMap so that the iteration order of variables is consistent regardless of insertion order
  // (which may change due to serialization). This is useful so that the serialized representation
  // is deterministic.
  protected final TreeMap<String, String> makeEnv = new TreeMap<>();

  protected final StoredEventHandler localEventHandler = new StoredEventHandler();

  @Nullable protected String ioExceptionMessage = null;
  @Nullable protected IOException ioException = null;
  @Nullable protected DetailedExitCode ioExceptionDetailedExitCode = null;

  // Used by glob(). Null for contexts where glob() is disallowed, including WORKSPACE files and
  // some tests.
  @Nullable private final Globber globber;

  protected final Map<Label, EnvironmentGroup> environmentGroups = new HashMap<>();

  private final Interner<ImmutableList<?>> listInterner = new ThreadCompatibleInterner<>();

  private final ImmutableMap<Location, String> generatorMap;

  private final PackageLimits packageLimits;

  protected final TestSuiteImplicitTestsAccumulator testSuiteImplicitTestsAccumulator =
      new TestSuiteImplicitTestsAccumulator();

  // A packageoid's FailureDetail field derives from the events on its Builder's event handler.
  // During package deserialization, those events are unavailable, because those events aren't
  // serialized [*]. Its FailureDetail value is serialized, however. During deserialization, that
  // value is assigned here, so that it can be assigned to the deserialized package.
  //
  // Likewise, during workspace part assembly, errors from parent parts should propagate to their
  // children.
  //
  // [*] Not in the context of the package, anyway. Skyframe values containing a package may
  // serialize events emitted during its construction/evaluation.
  @Nullable private FailureDetail failureDetailOverride = null;

  protected boolean alreadyBuilt = false;

  private long computationSteps = 0;

  /** Retrieves this object from a Starlark thread. Returns null if not present. */
  @Nullable
  public static TargetDefinitionContext fromOrNull(StarlarkThread thread) {
    StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
    return ctx instanceof TargetDefinitionContext targetDefinitionContext
        ? targetDefinitionContext
        : null;
  }

  /**
   * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
   * with an error message indicating that {@code what} can only be used in a target definition
   * context - meaning in a BUILD file, a legacy or symbolic macro, or a WORKSPACE file.
   */
  @CanIgnoreReturnValue
  public static TargetDefinitionContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
    if (ctx instanceof TargetDefinitionContext targetDefinitionContext) {
      return targetDefinitionContext;
    }
    boolean symbolicMacrosEnabled =
        thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_ENABLE_FIRST_CLASS_MACROS);
    throw Starlark.errorf(
        "%s can only be used while evaluating a BUILD file, a %smacro, or a WORKSPACE file",
        what, symbolicMacrosEnabled ? "legacy or symbolic " : "");
  }

  /**
   * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
   * with an error message indicating that {@code what} can only be used in a BUILD file, a
   * finalizer symbolic macro, or a WORKSPACE file.
   */
  @CanIgnoreReturnValue
  public static TargetDefinitionContext fromOrFailDisallowNonFinalizerMacros(
      StarlarkThread thread, String what) throws EvalException {
    @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
    if (ctx instanceof TargetDefinitionContext targetDefinitionContext
        && !targetDefinitionContext.recorder.currentlyInNonFinalizerMacro()) {
      return targetDefinitionContext;
    }
    throw newFromOrFailException(
        what, thread.getSemantics(), EnumSet.of(FromOrFailMode.ONLY_FINALIZER_MACROS));
  }

  /**
   * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
   * with an error message indicating that {@code what} can only be used in a BUILD file or a legacy
   * or symbolic macro.
   */
  @CanIgnoreReturnValue
  public static TargetDefinitionContext fromOrFailDisallowWorkspace(
      StarlarkThread thread, String what, String participle) throws EvalException {
    @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
    if (ctx instanceof TargetDefinitionContext targetDefinitionContext
        && !targetDefinitionContext.isRepoRulePackage()) {
      return targetDefinitionContext;
    }
    throw newFromOrFailException(
        what, participle, thread.getSemantics(), EnumSet.of(FromOrFailMode.NO_WORKSPACE));
  }

  /**
   * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
   * with an error message indicating that {@code what} can only be used in a BUILD file or a legacy
   * or symbolic macro.
   */
  @CanIgnoreReturnValue
  public static TargetDefinitionContext fromOrFailDisallowWorkspace(
      StarlarkThread thread, String what) throws EvalException {
    return fromOrFailDisallowWorkspace(thread, what, "used");
  }

  enum FromOrFailMode {
    NO_MACROS,
    ONLY_FINALIZER_MACROS,
    NO_WORKSPACE,
  }

  static EvalException newFromOrFailException(
      String what, StarlarkSemantics semantics, EnumSet<FromOrFailMode> modes) {
    return newFromOrFailException(what, "used", semantics, modes);
  }

  static EvalException newFromOrFailException(
      String what, String participle, StarlarkSemantics semantics, EnumSet<FromOrFailMode> modes) {
    // TODO(bazel-team): append a description of the current evaluation context to the error, e.g.
    // "foo() can only be used while evaluating a BUILD file or a legacy macro; in particular, it
    // cannot be used at the top level of a .bzl file"
    boolean symbolicMacrosEnabled =
        semantics.getBool(BuildLanguageOptions.EXPERIMENTAL_ENABLE_FIRST_CLASS_MACROS);
    ArrayList<String> allowedUses = new ArrayList<>();
    allowedUses.add("a BUILD file");
    allowedUses.add(
        String.format(
            "a %s%smacro",
            symbolicMacrosEnabled ? "legacy " : "",
            symbolicMacrosEnabled
                    && !modes.contains(FromOrFailMode.NO_MACROS)
                    && !modes.contains(FromOrFailMode.ONLY_FINALIZER_MACROS)
                ? "or symbolic "
                : ""));
    if (symbolicMacrosEnabled && modes.contains(FromOrFailMode.ONLY_FINALIZER_MACROS)) {
      allowedUses.add("a rule finalizer");
    }
    if (!modes.contains(FromOrFailMode.NO_WORKSPACE)) {
      allowedUses.add("a WORKSPACE file");
    }

    return Starlark.errorf(
        "%s can only be %s while evaluating %s",
        what, participle, StringUtil.joinEnglishList(allowedUses));
  }

  /**
   * Returns an auto-closeable resource to synchronize the computation step count between this
   * context and its thread which has started execution.
   */
  public StartedThreadComputationStepUpdater updateStartedThreadComputationSteps(
      StarlarkThread thread) {
    return new StartedThreadComputationStepUpdater(this, thread);
  }

  /**
   * Returns an auto-closeable resource to synchronize the computation step count between this
   * context and its thread whose execution is being paused, e.g. before pushing a new macro frame.
   */
  public PausedThreadComputationStepUpdater updatePausedThreadComputationSteps(
      StarlarkThread thread) {
    return new PausedThreadComputationStepUpdater(this, thread);
  }

  /**
   * An auto-closeable resource to synchronize the computation step count between a {@link
   * TargetDefinitionContext} and its thread which has started execution.
   */
  public static final class StartedThreadComputationStepUpdater implements AutoCloseable {
    private final TargetDefinitionContext context;
    private final StarlarkThread thread;
    private boolean closed = false;

    public StartedThreadComputationStepUpdater(
        TargetDefinitionContext context, StarlarkThread thread) {
      this.context = context;
      this.thread = thread;
      // Initialize the thread's computation step count to the context's total computation step
      // count.
      thread.incrementExecutedSteps(context.computationSteps);
      long threadMaxExecutionSteps = context.packageLimits.maxStarlarkComputationStepsPerPackage();
      if (threadMaxExecutionSteps < Long.MAX_VALUE) {
        // StarlarkThread.setMaxExecutionSteps(limit) throws if we hit limit, but we want to allow
        // hitting the limit (but not going over).
        threadMaxExecutionSteps++;
      }
      thread.setMaxExecutionSteps(threadMaxExecutionSteps);
    }

    @Override
    public void close() {
      if (!closed) {
        context.setComputationSteps(thread.getExecutedSteps());
      }
      closed = true;
    }
  }

  /**
   * An auto-closeable resource to synchronize the computation step count between a {@link
   * TargetDefinitionContext} and its thread whose execution is being paused.
   */
  public static final class PausedThreadComputationStepUpdater implements AutoCloseable {
    private final TargetDefinitionContext context;
    private final StarlarkThread thread;
    private boolean closed = false;

    public PausedThreadComputationStepUpdater(
        TargetDefinitionContext context, StarlarkThread thread) {
      this.context = context;
      this.thread = thread;
      context.setComputationSteps(thread.getExecutedSteps());
    }

    @Override
    public void close() {
      if (!closed) {
        checkState(
            thread.getExecutedSteps() <= context.computationSteps,
            "previously paused thread computation steps = %s cannot be greater than currently"
                + " recorded computation steps = %s",
            thread.getExecutedSteps(),
            context.computationSteps);
        thread.incrementExecutedSteps(context.computationSteps - thread.getExecutedSteps());
      }
      closed = true;
    }
  }

  /**
   * Sets the context's computation step count from the computation step count of the current
   * thread.
   */
  private void setComputationSteps(long threadComputationSteps) {
    checkState(
        threadComputationSteps >= computationSteps,
        "currently running thread computation steps = %s cannot be less than previously recorded"
            + " computation steps = %s",
        threadComputationSteps,
        computationSteps);
    computationSteps = threadComputationSteps;
  }

  /** Returns the "generator_name" to use for a given call site location in a BUILD file. */
  @Nullable
  String getGeneratorNameByLocation(Location loc) {
    return generatorMap.get(loc);
  }

  /**
   * Returns the map from BUILD file locations to "generator_name" values; intended only for use by
   * skyframe.PackageFunction.
   */
  public ImmutableMap<Location, String> getGeneratorMap() {
    return generatorMap;
  }

  /**
   * Returns the value to use for {@code test_suite}s' {@code $implicit_tests} attribute, as-is,
   * when the {@code test_suite} doesn't specify an explicit, non-empty {@code tests} value. The
   * returned list is mutated by the package-building process - it may be observed to be empty or
   * incomplete before package loading is complete. When package loading is complete it will contain
   * the label of each non-manual test matching the provided tags in the package, in label order.
   *
   * <p>This method <b>MUST</b> be called before the package is built - otherwise the requested
   * implicit tests won't be accumulated.
   */
  List<Label> getTestSuiteImplicitTestsRef(List<String> tags) {
    return testSuiteImplicitTestsAccumulator.getTestSuiteImplicitTestsRefForTags(tags);
  }

  @ThreadCompatible
  private static final class ThreadCompatibleInterner<T> implements Interner<T> {
    private final Map<T, T> interns = new HashMap<>();

    @Override
    public T intern(T sample) {
      T existing = interns.putIfAbsent(sample, sample);
      return firstNonNull(existing, sample);
    }
  }

  TargetDefinitionContext(
      Metadata metadata,
      Packageoid pkg,
      SymbolGenerator<?> symbolGenerator,
      boolean simplifyUnconditionalSelectsInRuleAttrs,
      String workspaceName,
      RepositoryMapping mainRepositoryMapping,
      @Nullable Semaphore cpuBoundSemaphore,
      PackageOverheadEstimator packageOverheadEstimator,
      @Nullable ImmutableMap<Location, String> generatorMap,
      @Nullable Globber globber,
      boolean enableNameConflictChecking,
      boolean trackFullMacroInformation,
      boolean enableTargetMapSnapshotting,
      PackageLimits packageLimits) {
    super(() -> mainRepositoryMapping);
    this.metadata = metadata;
    this.pkg = pkg;
    this.symbolGenerator = symbolGenerator;
    this.workspaceName = Preconditions.checkNotNull(workspaceName);
    this.simplifyUnconditionalSelectsInRuleAttrs = simplifyUnconditionalSelectsInRuleAttrs;
    this.labelConverter =
        new LabelConverter(metadata.packageIdentifier(), metadata.repositoryMapping());
    this.cpuBoundSemaphore = cpuBoundSemaphore;
    this.packageOverheadEstimator = packageOverheadEstimator;
    this.generatorMap = (generatorMap == null) ? ImmutableMap.of() : generatorMap;
    this.globber = globber;
    this.recorder =
        new TargetRecorder(
            enableNameConflictChecking, trackFullMacroInformation, enableTargetMapSnapshotting);
    this.packageLimits = packageLimits;
  }

  public Metadata getMetadata() {
    return metadata;
  }

  SymbolGenerator<?> getSymbolGenerator() {
    return symbolGenerator;
  }

  PackageIdentifier getPackageIdentifier() {
    return metadata.packageIdentifier();
  }

  /**
   * Returns a short, lower-case description of the packageoid under construction, e.g. for use in
   * logging and error messages.
   */
  String getShortDescription() {
    return pkg.getShortDescription();
  }

  /**
   * Determine whether this package should contain build rules (returns {@code false}) or repo rules
   * (returns {@code true}).
   */
  public boolean isRepoRulePackage() {
    return metadata.isRepoRulePackage();
  }

  /**
   * Returns the name of the workspace this package is in. Used as a prefix for the runfiles
   * directory. This can be set in the WORKSPACE file. This must be a valid target name.
   */
  String getWorkspaceName() {
    // Current value is stored in the builder field, final value is copied to the Package in
    // finishInit().
    return workspaceName;
  }

  /**
   * Returns the name of the Bzlmod module associated with the repo this package is in. If this
   * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
   * this is the name of the module hosting the extension.
   */
  Optional<String> getAssociatedModuleName() {
    return metadata.associatedModuleName();
  }

  /**
   * Returns the version of the Bzlmod module associated with the repo this package is in. If this
   * package is not from a Bzlmod repo, this is empty. For repos generated by module extensions,
   * this is the version of the module hosting the extension.
   */
  Optional<String> getAssociatedModuleVersion() {
    return metadata.associatedModuleVersion();
  }

  public LabelConverter getLabelConverter() {
    return labelConverter;
  }

  Interner<ImmutableList<?>> getListInterner() {
    return listInterner;
  }

  RootedPath getFilename() {
    return metadata.buildFilename();
  }

  /** Returns the {@link StoredEventHandler} associated with this builder. */
  public StoredEventHandler getLocalEventHandler() {
    return localEventHandler;
  }

  /**
   * Retrieves the current package args. Note that during BUILD file evaluation these are still
   * subject to mutation.
   */
  public PackageArgs getPartialPackageArgs() {
    return pkg.getDeclarations().getPackageArgs();
  }

  public boolean containsErrors() {
    return recorder.containsErrors();
  }

  /**
   * Declares that errors were encountering while loading this package.
   *
   * <p>If this method is called, then there should also be an ERROR event added to the handler on
   * the {@link Package.Builder}. The event should include a {@link FailureDetail}.
   */
  public void setContainsErrors() {
    checkState(
        pkg.targets == null,
        "TargetDefinitionContext.setContainsErrors() can only be used before finishBuild() has"
            + " propagated the builder's error status to the packageoid");
    recorder.setContainsErrors();
  }

  void setIOException(IOException e, String message, DetailedExitCode detailedExitCode) {
    this.ioException = e;
    this.ioExceptionMessage = message;
    this.ioExceptionDetailedExitCode = detailedExitCode;
    setContainsErrors();
  }

  /**
   * Returns the {@link Globber} used to implement {@code glob()} functionality during BUILD
   * evaluation. Null for contexts where globbing is not possible, including WORKSPACE files and
   * some tests.
   */
  @Nullable
  public Globber getGlobber() {
    return globber;
  }

  /**
   * Returns true if values of conditional rule attributes which only contain unconditional selects
   * should be simplified and stored as a non-select value.
   */
  public boolean simplifyUnconditionalSelectsInRuleAttrs() {
    return this.simplifyUnconditionalSelectsInRuleAttrs;
  }

  /**
   * Returns the innermost currently executing symbolic macro, or null if not in a symbolic macro.
   */
  @Nullable
  public MacroInstance currentMacro() {
    MacroFrame frame = recorder.getCurrentMacroFrame();
    return frame == null ? null : frame.macroInstance;
  }

  /**
   * Creates a new {@link Rule} {@code r} where {@code r.getPackageoid()} is the {@link Packageoid}
   * associated with this {@link Builder}.
   *
   * <p>The created {@link Rule} will have no output files and therefore will be in an invalid
   * state.
   *
   * @param threadCallStack the call stack of the thread that created the rule. Call stacks for
   *     threads of enclosing symbolic macros (if any) will be prepended to it automatically to form
   *     the rule's full call stack.
   */
  Rule createRule(
      Label label, RuleClass ruleClass, List<StarlarkThread.CallStackEntry> threadCallStack) {
    CallStack.Node fullInteriorCallStack;
    final Location location;
    if (currentMacro() != null) {
      location = currentMacro().getBuildFileLocation();
      fullInteriorCallStack = CallStack.compact(threadCallStack, /* start= */ 0);
      for (MacroInstance macro = currentMacro(); macro != null; macro = macro.getParent()) {
        fullInteriorCallStack =
            CallStack.concatenate(macro.getParentCallStack(), fullInteriorCallStack);
      }
    } else {
      location = threadCallStack.isEmpty() ? Location.BUILTIN : threadCallStack.get(0).location;
      fullInteriorCallStack = CallStack.compact(threadCallStack, /* start= */ 1);
    }
    return createRule(label, ruleClass, location, fullInteriorCallStack);
  }

  Rule createRule(
      Label label,
      RuleClass ruleClass,
      Location location,
      @Nullable CallStack.Node interiorCallStack) {
    return new Rule(pkg, label, ruleClass, location, interiorCallStack);
  }

  /** Creates a new {@link MacroInstance} in this builder's packageoid. */
  MacroInstance createMacro(
      MacroClass macroClass,
      String name,
      int sameNameDepth,
      List<StarlarkThread.CallStackEntry> parentCallStack)
      throws LabelSyntaxException, EvalException {
    MacroInstance parent = currentMacro();
    final Location location;
    final CallStack.Node compactParentCallStack;
    if (parent != null) {
      location = parent.getBuildFileLocation();
      compactParentCallStack = CallStack.compact(parentCallStack, /* start= */ 0);
    } else {
      location = parentCallStack.isEmpty() ? Location.BUILTIN : parentCallStack.get(0).location;
      compactParentCallStack = CallStack.compact(parentCallStack, /* start= */ 1);
    }
    return new MacroInstance(
        pkg.getMetadata(),
        pkg.getDeclarations(),
        parent,
        parent != null ? parent.getGeneratorName() : generatorMap.get(location),
        location,
        compactParentCallStack,
        macroClass,
        Label.create(pkg.getMetadata().packageIdentifier(), name),
        sameNameDepth);
  }

  /** Returns true if symbolic macros should be eagerly expanded in this context. */
  public abstract boolean eagerlyExpandMacros();

  @Nullable
  public MacroFrame getCurrentMacroFrame() {
    return recorder.getCurrentMacroFrame();
  }

  @Nullable
  public MacroFrame setCurrentMacroFrame(@Nullable MacroFrame frame) {
    return recorder.setCurrentMacroFrame(frame);
  }

  public boolean currentlyInNonFinalizerMacro() {
    return recorder.currentlyInNonFinalizerMacro();
  }

  @Nullable
  public Target getTarget(String name) {
    return recorder.getTarget(name);
  }

  // TODO: #19922 - Refactor finalizer expansion such that TargetDefinitionContext can handle
  // working with finalizer macros. At that point, getRulesSnapshotView() and
  // getNonFinalizerInstantiatedRule() must account for the snapshot view here rather than in the
  // override in Package.Builder.

  /**
   * Returns a lightweight snapshot view of the names of all rule targets belonging to this package
   * at the time of this call; in finalizer expansion stage, returns a lightweight snapshot view of
   * only the non-finalizer-instantiated rule targets.
   *
   * @throws IllegalStateException if this method is called after {@link
   *     Package.Builder#beforeBuild} has been called.
   */
  Map<String, Rule> getRulesSnapshotView() {
    if (recorder.getTargetMap() instanceof SnapshottableBiMap<?, ?>) {
      return Maps.transformValues(
          ((SnapshottableBiMap<String, Target>) recorder.getTargetMap()).getTrackedSnapshot(),
          target -> (Rule) target);
    } else {
      // TODO(https://github.com/bazelbuild/bazel/issues/23852): if we are in a PackagePiece
      // builder, trigger a skyframe restart and request a full Package.
      throw new IllegalStateException(
          "getRulesSnapshotView() cannot be used after beforeBuild() has been called");
    }
  }

  /**
   * Returns a non-finalizer-instantiated rule target with the provided name belonging to this
   * package at the time of this call. If such a rule target cannot be returned, returns null.
   */
  // TODO(https://github.com/bazelbuild/bazel/issues/23765): when we restrict
  // native.existing_rule() to be usable only in finalizer context, we can replace this method
  // with {@code getRulesSnapshotView().get(name)}; we don't do so at present because we do not
  // want to make unnecessary snapshots.
  @Nullable
  Rule getNonFinalizerInstantiatedRule(String name) {
    Target target = recorder.getTargetMap().get(name);
    return target instanceof Rule ? (Rule) target : null;
  }

  /**
   * Creates an input file target in this package with the specified name, if it does not yet exist.
   *
   * <p>This operation is idempotent.
   *
   * @param targetName name of the input file. This must be a valid target name as defined by {@link
   *     com.google.devtools.build.lib.cmdline.LabelValidator#validateTargetName}.
   * @return the newly-created {@code InputFile}, or the old one if it already existed.
   * @throws NameConflictException if the name was already taken by another target that is not an
   *     input file
   * @throws IllegalArgumentException if the name is not a valid label
   */
  InputFile createInputFile(String targetName, Location location) throws NameConflictException {
    Target existing = recorder.getTargetMap().get(targetName);

    if (existing instanceof InputFile) {
      return (InputFile) existing; // idempotent
    }

    InputFile inputFile;
    try {
      inputFile = new InputFile(pkg, createLabel(targetName), location);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(
          "FileTarget in package " + metadata.getName() + " has illegal name: " + targetName, e);
    }

    recorder.addTarget(inputFile);
    return inputFile;
  }

  /**
   * Sets the visibility and license for an input file. The input file must already exist as a
   * member of this package.
   *
   * @throws IllegalArgumentException if the input file doesn't exist in this package's target map.
   */
  // TODO: #19922 - Don't allow exports_files() to modify visibility of targets that the current
  // symbolic macro did not create. Fun pathological example: exports_files() modifying the
  // visibility of :BUILD inside a symbolic macro.
  void setVisibilityAndLicense(InputFile inputFile, RuleVisibility visibility, License license) {
    String filename = inputFile.getName();
    Target cacheInstance = recorder.getTargetMap().get(filename);
    if (!(cacheInstance instanceof InputFile)) {
      throw new IllegalArgumentException(
          "Can't set visibility for nonexistent FileTarget "
              + filename
              + " in package "
              + metadata.getName()
              + ".");
    }
    if (!((InputFile) cacheInstance).isVisibilitySpecified()
        || cacheInstance.getVisibility() != visibility
        || !Objects.equals(cacheInstance.getLicense(), license)) {
      recorder.replaceInputFileUnchecked(
          new VisibilityLicenseSpecifiedInputFile(
              pkg, cacheInstance.getLabel(), cacheInstance.getLocation(), visibility, license));
    }
  }

  /**
   * Creates a label for a target inside this package.
   *
   * @throws LabelSyntaxException if the {@code targetName} is invalid
   */
  Label createLabel(String targetName) throws LabelSyntaxException {
    return Label.create(metadata.packageIdentifier(), targetName);
  }

  /** Adds a package group to the package. */
  void addPackageGroup(
      String name,
      Collection<String> packages,
      Collection<Label> includes,
      boolean allowPublicPrivate,
      boolean repoRootMeansCurrentRepo,
      EventHandler eventHandler,
      Location location)
      throws NameConflictException, LabelSyntaxException {
    PackageGroup group =
        new PackageGroup(
            createLabel(name),
            pkg,
            packages,
            includes,
            allowPublicPrivate,
            repoRootMeansCurrentRepo,
            eventHandler,
            location);
    recorder.addTarget(group);

    if (group.containsErrors()) {
      setContainsErrors();
    }
  }

  public void addRule(Rule rule) throws NameConflictException {
    Preconditions.checkArgument(rule.getPackageoid() == pkg);
    recorder.addRule(rule);
  }

  public void addMacro(MacroInstance macro) throws NameConflictException {
    checkState(!isRepoRulePackage(), "Cannot instantiate symbolic macros in this context");
    recorder.addMacro(macro);
  }

  @Nullable
  public Semaphore getCpuBoundSemaphore() {
    return cpuBoundSemaphore;
  }

  void setFailureDetailOverride(FailureDetail failureDetail) {
    failureDetailOverride = failureDetail;
  }

  @Nullable
  FailureDetail getFailureDetail() {
    if (failureDetailOverride != null) {
      return failureDetailOverride;
    }

    List<Event> undetailedEvents = null;
    for (Event event : localEventHandler.getEvents()) {
      if (event.getKind() != EventKind.ERROR) {
        continue;
      }
      DetailedExitCode detailedExitCode = event.getProperty(DetailedExitCode.class);
      if (detailedExitCode != null && detailedExitCode.getFailureDetail() != null) {
        return detailedExitCode.getFailureDetail();
      }
      if (containsErrors()) {
        if (undetailedEvents == null) {
          undetailedEvents = new ArrayList<>();
        }
        undetailedEvents.add(event);
      }
    }
    if (undetailedEvents != null) {
      BugReport.sendNonFatalBugReport(
          new IllegalStateException(
              "TargetDefinitionContext has undetailed error from "
                  + undetailedEvents
                  + " for packageoid "
                  + pkg));
    }
    return null;
  }

  /**
   * Returns the number of Starlark computation steps executed thus far by threads performing
   * evaluation of this packageoid, which are recorded by updaters created by {@link
   * #updateStartedThreadComputationSteps} and {@link #updatePausedThreadComputationSteps}.
   */
  long getComputationSteps() {
    return computationSteps;
  }

  //
  // Packageoid (package or package piece) construction methods, intended for use only by
  // PackageFunction and friends.
  //

  @CanIgnoreReturnValue
  protected TargetDefinitionContext beforeBuild() throws NoSuchPackageException {
    if (ioException != null) {
      throw new NoSuchPackageException(
          getPackageIdentifier(), ioExceptionMessage, ioException, ioExceptionDetailedExitCode);
    }

    // TODO(bazel-team): We run testSuiteImplicitTestsAccumulator here in beforeBuild(), but what
    // if one of the accumulated tests is later removed in PackageFunction, between the call to
    // buildPartial() and finishBuild(), due to a label-crossing-subpackage-boundary error? Seems
    // like that would mean a test_suite is referencing a Target that's been deleted from its
    // Package.

    // Clear tests before discovering them again in order to keep this method idempotent -
    // otherwise we may double-count tests if we're called twice due to a skyframe restart, etc.
    testSuiteImplicitTestsAccumulator.clearAccumulatedTests();
    for (Rule rule : recorder.getRules()) {
      testSuiteImplicitTestsAccumulator.processRule(rule);
    }
    // Make sure all accumulated values are sorted for determinism.
    testSuiteImplicitTestsAccumulator.sortTests();

    return this;
  }

  /** Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only. */
  // TODO(bazel-team): It seems like the motivation for this method (added in cl/74794332) is to
  // allow PackageFunction to delete targets that are found to violate the
  // label-crossing-subpackage-boundaries check. Is there a simpler way to express this idea that
  // doesn't make package-building a multi-stage process?
  @CanIgnoreReturnValue
  public TargetDefinitionContext buildPartial() throws NoSuchPackageException {
    if (alreadyBuilt) {
      return this;
    }
    return beforeBuild();
  }

  /**
   * Intended for use by {@link com.google.devtools.build.lib.skyframe.PackageFunction} only.
   *
   * <p>This method is intended to be overridden by subclasses to perform packageoid-specific final
   * initialization steps.
   */
  // Non-final only to allow subclasses to return a more specific type.
  public Packageoid finishBuild() {
    if (alreadyBuilt) {
      return pkg;
    }
    alreadyBuilt = true;

    // Freeze rules, compacting their attributes' representations.
    for (Rule rule : recorder.getRules()) {
      rule.freeze();
    }

    // Freeze macros, compacting their attributes' representations.
    for (MacroInstance macro : recorder.getMacroMap().values()) {
      macro.freeze();
    }

    // Last chance to set the builder's error status.
    finalBuilderValidationHook();

    // Initialize packageoid.
    pkg.containsErrors |= containsErrors();
    pkg.failureDetail = getFailureDetail();
    pkg.targets = ImmutableSortedMap.copyOf(recorder.getTargetMap());

    packageoidInitializationHook();

    // Overhead should be estimated after all packageoid fields have been set.
    OptionalLong overheadEstimate = packageOverheadEstimator.estimatePackageOverhead(pkg);
    pkg.packageOverhead = overheadEstimate.orElse(Packageoid.PACKAGE_OVERHEAD_UNSET);

    // Verify that we haven't introduced new errors on the builder since the call to
    // finalBuilderValidationHook().
    if (containsErrors()) {
      checkState(
          pkg.containsErrors(), "Builder error status not propagated to package or package piece");
    }

    return pkg;
  }

  /**
   * Performs final builder validations (if needed), possibly modifying the builder's error status.
   *
   * <p>This method is intended to be overridden by subclasses; it is invoked by {@link
   * #finishBuild()} immediately before initializing the packageoid and copying error status from
   * the builder to the packageoid.
   */
  protected void finalBuilderValidationHook() {}

  /**
   * Sets remaining subclass-specific fields on the packageoid.
   *
   * <p>This method is intended to be overridden by subclasses; it is invoked by {@link
   * #finishBuild()} after {@link #finalBuilderValidationHook()} has passed and the packageoid's
   * base fields (such as error information, targets, and macros) have been frozen and set. This
   * method must not call {@link #setContainsErrors()} on the builder; but it is allowed to set
   * packageoid fields that impact overhead estimation.
   */
  protected void packageoidInitializationHook() {}
}
