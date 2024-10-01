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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Metadata;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroFrame;
import com.google.devtools.build.lib.packages.TargetRecorder.NameConflictException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;

/**
 * Base class of {@link Package.Builder} that encapsulates all the operations that may need to occur
 * in the middle of BUILD file evaluation, without including operations specific to the setup or
 * finalization of {@code Package} construction.
 *
 * <p>In other words, if a {@code Package.Builder} method needs to be called as a result of Starlark
 * evaluation of the BUILD file or its macros, the operation belongs in this base class.
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
  protected final Package pkg;

  // The container object on which targets and macro instances are added and conflicts are
  // detected.
  protected final TargetRecorder recorder = new TargetRecorder();

  // Initialized from outside but also potentially set by `workspace()` function in WORKSPACE
  // file.
  protected String workspaceName;

  protected final Label buildFileLabel;

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

  /** True iff the "package" function has already been called in this package. */
  private boolean packageFunctionUsed;

  private final Interner<ImmutableList<?>> listInterner = new ThreadCompatibleInterner<>();

  private final ImmutableMap<Location, String> generatorMap;

  protected final TestSuiteImplicitTestsAccumulator testSuiteImplicitTestsAccumulator =
      new TestSuiteImplicitTestsAccumulator();

  /** Returns the "generator_name" to use for a given call site location in a BUILD file. */
  @Nullable
  String getGeneratorNameByLocation(Location loc) {
    return generatorMap.get(loc);
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
      Package pkg,
      SymbolGenerator<?> symbolGenerator,
      boolean simplifyUnconditionalSelectsInRuleAttrs,
      String workspaceName,
      RepositoryMapping mainRepositoryMapping,
      @Nullable Semaphore cpuBoundSemaphore,
      @Nullable ImmutableMap<Location, String> generatorMap,
      @Nullable Globber globber) {
    super(() -> mainRepositoryMapping);
    this.metadata = metadata;
    this.pkg = pkg;
    this.symbolGenerator = symbolGenerator;
    this.workspaceName = Preconditions.checkNotNull(workspaceName);

    try {
      this.buildFileLabel =
          Label.create(
              metadata.packageIdentifier(),
              metadata.buildFilename().getRootRelativePath().getBaseName());
    } catch (LabelSyntaxException e) {
      // This can't actually happen.
      throw new AssertionError(
          "Package BUILD file has an illegal name: " + metadata.buildFilename(), e);
    }

    this.simplifyUnconditionalSelectsInRuleAttrs = simplifyUnconditionalSelectsInRuleAttrs;
    this.labelConverter =
        new LabelConverter(metadata.packageIdentifier(), metadata.repositoryMapping());
    if (metadata.getName().startsWith("javatests/")) {
      mergePackageArgsFrom(PackageArgs.builder().setDefaultTestOnly(true));
    }
    this.cpuBoundSemaphore = cpuBoundSemaphore;
    this.generatorMap = (generatorMap == null) ? ImmutableMap.of() : generatorMap;
    this.globber = globber;

    // Add target for the BUILD file itself.
    // (This may be overridden by an exports_file declaration.)
    // TODO: #19922 - Figure out exactly where this line belongs once TargetDefinitionContext is a
    // base class for both Package construction and PackagePiece construction.
    recorder.addInputFileUnchecked(
        new InputFile(
            pkg, buildFileLabel, Location.fromFile(metadata.buildFilename().asPath().toString())));
  }

  SymbolGenerator<?> getSymbolGenerator() {
    return symbolGenerator;
  }

  PackageIdentifier getPackageIdentifier() {
    return metadata.packageIdentifier();
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

  public Label getBuildFileLabel() {
    return buildFileLabel;
  }

  RootedPath getFilename() {
    return metadata.buildFilename();
  }

  /** Returns the {@link StoredEventHandler} associated with this builder. */
  public StoredEventHandler getLocalEventHandler() {
    return localEventHandler;
  }

  public void setMakeVariable(String name, String value) {
    makeEnv.put(name, value);
  }

  public abstract void mergePackageArgsFrom(PackageArgs newPackageArgs);

  public abstract void mergePackageArgsFrom(PackageArgs.Builder builder);

  /**
   * Retrieves the current package args. Note that during BUILD file evaluation these are still
   * subject to mutation.
   */
  public abstract PackageArgs getPartialPackageArgs();

  /** Uses the workspace name from {@code //external} to set this package's workspace name. */
  @VisibleForTesting
  public void setWorkspaceName(String workspaceName) {
    this.workspaceName = workspaceName;
  }

  /** Returns whether the "package" function has been called yet */
  boolean isPackageFunctionUsed() {
    return packageFunctionUsed;
  }

  void setPackageFunctionUsed() {
    packageFunctionUsed = true;
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
   * Creates a new {@link Rule} {@code r} where {@code r.getPackage()} is the {@link Package}
   * associated with this {@link Builder}.
   *
   * <p>The created {@link Rule} will have no output files and therefore will be in an invalid
   * state.
   */
  Rule createRule(Label label, RuleClass ruleClass, List<StarlarkThread.CallStackEntry> callstack) {
    return createRule(
        label,
        ruleClass,
        callstack.isEmpty() ? Location.BUILTIN : callstack.get(0).location,
        CallStack.compactInterior(callstack));
  }

  Rule createRule(
      Label label,
      RuleClass ruleClass,
      Location location,
      @Nullable CallStack.Node interiorCallStack) {
    return new Rule(pkg, label, ruleClass, location, interiorCallStack);
  }

  /**
   * Creates a new {@link MacroInstance} {@code m} where {@code m.getPackage()} is the {@link
   * Package} associated with this {@link Builder}.
   */
  MacroInstance createMacro(
      MacroClass macroClass, Map<String, Object> attrValues, int sameNameDepth)
      throws EvalException {
    MacroInstance parent = currentMacro();
    return new MacroInstance(pkg, parent, macroClass, attrValues, sameNameDepth);
  }

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

  /**
   * Returns true if any labels in the given list appear multiple times, reporting an appropriate
   * error message if so.
   *
   * <p>TODO(bazel-team): apply this to all build functions (maybe automatically?), possibly
   * integrate with RuleClass.checkForDuplicateLabels.
   */
  private static boolean hasDuplicateLabels(
      List<Label> labels,
      String owner,
      String attrName,
      Location location,
      EventHandler eventHandler) {
    Set<Label> dupes = CollectionUtils.duplicatedElementsOf(labels);
    for (Label dupe : dupes) {
      eventHandler.handle(
          Package.error(
              location,
              String.format(
                  "label '%s' is duplicated in the '%s' list of '%s'", dupe, attrName, owner),
              Code.DUPLICATE_LABEL));
    }
    return !dupes.isEmpty();
  }

  /** Adds an environment group to the package. Not valid within symbolic macros. */
  void addEnvironmentGroup(
      String name,
      List<Label> environments,
      List<Label> defaults,
      EventHandler eventHandler,
      Location location)
      throws NameConflictException, LabelSyntaxException {
    Preconditions.checkState(currentMacro() == null);

    if (hasDuplicateLabels(environments, name, "environments", location, eventHandler)
        || hasDuplicateLabels(defaults, name, "defaults", location, eventHandler)) {
      setContainsErrors();
      return;
    }

    EnvironmentGroup group =
        new EnvironmentGroup(createLabel(name), pkg, environments, defaults, location);
    recorder.addTarget(group);

    // Invariant: once group is inserted into targets, it must also:
    // (a) be inserted into environmentGroups, or
    // (b) have its group.processMemberEnvironments called.
    // Otherwise it will remain uninitialized,
    // causing crashes when it is later toString-ed.

    for (Event error : group.validateMembership()) {
      eventHandler.handle(error);
      setContainsErrors();
    }

    // For each declared environment, make sure it doesn't also belong to some other group.
    for (Label environment : group.getEnvironments()) {
      EnvironmentGroup otherGroup = environmentGroups.get(environment);
      if (otherGroup != null) {
        eventHandler.handle(
            Package.error(
                location,
                String.format(
                    "environment %s belongs to both %s and %s",
                    environment, group.getLabel(), otherGroup.getLabel()),
                Code.ENVIRONMENT_IN_MULTIPLE_GROUPS));
        setContainsErrors();
        // Ensure the orphan gets (trivially) initialized.
        group.processMemberEnvironments(ImmutableMap.of());
      } else {
        environmentGroups.put(environment, group);
      }
    }
  }

  public void addRule(Rule rule) throws NameConflictException {
    Preconditions.checkArgument(rule.getPackage() == pkg);
    recorder.addRule(rule);
  }

  public void addMacro(MacroInstance macro) throws NameConflictException {
    Preconditions.checkState(
        !isRepoRulePackage(), "Cannot instantiate symbolic macros in this context");
    recorder.addMacro(macro);
  }

  @Nullable
  public Semaphore getCpuBoundSemaphore() {
    return cpuBoundSemaphore;
  }
}
