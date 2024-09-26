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

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A context in which targets and symbolic macros for a specific package may be added.
 *
 * <p>This object is responsible for recording the existence of these targets and macros, and
 * enforcing naming requirements on them. It is used by {@link Package.Builder} as part of package
 * construction.
 */
public final class TargetRecorder {

  /** Used for constructing macro namespace violation error messages. */
  static final String MACRO_NAMING_RULES =
      "Name must be the same as the macro's name, or the macro's name followed by '_'"
          + " (recommended), '-', or '.', and a non-empty string.";

  private boolean containsErrors = false;

  // All targets added to the package.
  //
  // We use SnapshottableBiMap to help track insertion order of Rule targets, for use by
  // native.existing_rules().
  private BiMap<String, Target> targetMap =
      new SnapshottableBiMap<>(target -> target instanceof Rule);

  // All instances of symbolic macros created during package construction, indexed by id (not
  // name).
  private final Map<String, MacroInstance> macroMap = new LinkedHashMap<>();

  /**
   * Represents the innermost currently executing symbolic macro, or null if none are running.
   *
   * <p>Logically, this is the top entry of a stack of frames where each frame corresponds to a
   * nested symbolic macro invocation. In actuality, symbolic macros do not necessarily run eagerly
   * when they are invoked, so this is not really a call stack per se. We leave it to the pkgbuilder
   * client to set the current frame, so that the choice of whether to push and pop, or process a
   * worklist of queued evaluations, is up to them.
   *
   * <p>The state of this field is used to determine what Starlark APIs are available (see user
   * documentation on {@code macro()} at {@link StarlarkRuleFunctionsApi#macro}), and to help
   * enforce naming requirements on targets and macros.
   */
  @Nullable private MacroFrame currentMacroFrame = null;

  /**
   * Represents the state of a running symbolic macro (see {@link #currentMacroFrame}). Semi-opaque.
   */
  static class MacroFrame {
    final MacroInstance macroInstance;
    // Most name conflicts are caught by checking the keys of the `targetMap` and `macroMap` maps.
    // It is not a conflict for a target or macro to have the same name as the macro it is
    // declared in, yet such a target or macro may still conflict with siblings in the same macro.
    // We use this bool to track whether or not a newly introduced macro, M, having the same name
    // as its parent (the current macro), would clash with an already defined sibling of M.
    private boolean mainSubmacroHasBeenDefined = false;

    MacroFrame(MacroInstance macroInstance) {
      this.macroInstance = macroInstance;
    }
  }

  private enum NameConflictCheckingPolicy {
    UNKNOWN,
    NOT_GUARANTEED,
    ENABLED;
  }

  /**
   * Whether to do all validation checks for name clashes among targets, macros, and output file
   * prefixes.
   *
   * <p>The {@code NOT_GUARANTEED} value should only be used when the package data has already been
   * validated, e.g. in package deserialization.
   *
   * <p>Setting it to {@code NOT_GUARANTEED} does not necessarily turn off *all* checking, just some
   * of the more expensive ones. Do not rely on being able to violate these checks.
   */
  private NameConflictCheckingPolicy nameConflictCheckingPolicy =
      NameConflictCheckingPolicy.UNKNOWN;

  /**
   * Stores labels for each rule so that we don't have to call the costly {@link Rule#getLabels}
   * twice (once for {@link Package.Builder#checkForInputOutputConflicts} and once for {@link
   * Package.Builder#beforeBuild}).
   *
   * <p>This field is null if name conflict checking is disabled. It is also null after the package
   * is built.
   */
  // TODO(#19922): Technically we don't need to store entries for rules that were created by
  // macros; see rulesCreatedInMacros, below.
  @Nullable private Map<Rule, List<Label>> ruleLabels = new HashMap<>();

  /**
   * Stores labels of rule targets that were created in symbolic macros. We don't implicitly create
   * input files on behalf of such targets (though they may still be created on behalf of other
   * targets not in macros).
   *
   * <p>This field is null if name conflict checking is disabled. It is also null after the package
   * is built.
   */
  // TODO(#19922): This can be eliminated once we have Targets directly store a reference to the
  // MacroInstance that instantiated them. (This is a little nontrivial because we'd like to avoid
  // simply adding a new field to Target subclasses, and instead want to combine it with the
  // existing Package-typed field.)
  @Nullable private Set<Rule> rulesCreatedInMacros = new HashSet<>();

  /**
   * A map from names of targets declared in a symbolic macro which violate macro naming rules, such
   * as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro instance where
   * they were declared.
   *
   * <p>This field is null if name conflict checking is disabled. The content of the map is
   * manipulated only in {@link #checkRuleAndOutputs}.
   */
  @Nullable
  private LinkedHashMap<String, String> macroNamespaceViolatingTargets = new LinkedHashMap<>();

  /**
   * A map from target name to the (innermost) macro instance that declared it. See {@link
   * Package#targetsToDeclaringMacros}.
   */
  private final LinkedHashMap<String, MacroInstance> targetsToDeclaringMacros =
      new LinkedHashMap<>();

  /**
   * The collection of the prefixes of every output file. Maps each prefix to an arbitrary output
   * file having that prefix. Used for error reporting.
   *
   * <p>This field is null if name conflict checking is disabled. It is also null after the package
   * is built. The content of the map is manipulated only in {@link #checkRuleAndOutputs}.
   */
  @Nullable private Map<String, OutputFile> outputFilePrefixes = new HashMap<>();

  public Map<String, Target> getTargetMap() {
    return targetMap;
  }

  public Map<String, MacroInstance> getMacroMap() {
    return macroMap;
  }

  public List<Label> getRuleLabels(Rule rule) {
    return (ruleLabels != null) ? ruleLabels.get(rule) : rule.getLabels();
  }

  public boolean isRuleCreatedInMacro(Rule rule) {
    return rulesCreatedInMacros.contains(rule);
  }

  /**
   * Returns a map from names of targets declared in a symbolic macro which violate macro naming
   * rules, such as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro
   * instance where they were declared.
   */
  public Map<String, String> getMacroNamespaceViolatingTargets() {
    return macroNamespaceViolatingTargets != null
        ? macroNamespaceViolatingTargets
        : ImmutableMap.of();
  }

  /**
   * A map from target name to the (innermost) macro instance that declared it. See {@link
   * Package#targetsToDeclaringMacros}.
   */
  public Map<String, MacroInstance> getTargetsToDeclaringMacros() {
    return targetsToDeclaringMacros;
  }

  /**
   * Declares that errors were encountering while loading this package.
   *
   * <p>If this method is called, then there should also be an ERROR event added to the handler on
   * the {@link Package.Builder}. The event should include a {@link FailureDetail}.
   */
  // TODO(bazel-team): For simplicity it would be nice to replace the use of an error bit with
  // pkgBuilder.getLocalEventHandler().hasErrors(), since that would prevent the kind of
  // inconsistency where we have reported an ERROR event but not called setContainsErrors(), or vice
  // versa. We could even assert that the error event has a FailureDetail, though that's a linear
  // scan unless we customize the event handler.
  // TODO(bazel-team): At the moment the pkgBuilder's error bit is stored here on this class. But
  // there are ways that Package.Builder#setContainsErrors gets called that have nothing to do with
  // broken targets, e.g. a Starlark eval error. One fix is to put the error bit on the pkgBuilder
  // only, and have this class accept a callback to invoke when registering a target that's in
  // error, and set that callback to pkgBuilder::setContainsErrors. Another fix is to have both
  // classes store error bits, and have the builder union this class's error bit into its own in
  // finishBuild().
  public void setContainsErrors() {
    this.containsErrors = true;
  }

  public boolean containsErrors() {
    return containsErrors;
  }

  /**
   * Inserts a target into {@code targetMap}. Returns the previous target if one was present, or
   * null.
   *
   * <p>No validation is done on the target's name.
   */
  @CanIgnoreReturnValue
  @Nullable
  private Target putTargetInternal(Target target) {
    Target existing = targetMap.put(target.getName(), target);
    if (currentMacroFrame != null) {
      targetsToDeclaringMacros.put(target.getName(), currentMacroFrame.macroInstance);
    }
    return existing;
  }

  /**
   * Inserts a target into the target map.
   *
   * <p>The target must have a valid name (for the current macro) and cannot have already been
   * added.
   */
  public void addTarget(Target target) throws NameConflictException {
    if (target instanceof Rule rule) {
      // Use addRule() to ensure all rule-related maps and caches are consulted.
      // checkTargetName() and putTargetInternal() are both reached through addRule().
      addRule(rule);
    } else {
      checkTargetName(target);
      putTargetInternal(target);
    }
  }

  /**
   * Inserts an input file into the target map.
   *
   * <p>No validation is done on the target's name.
   *
   * <p>The target must not have already been added, and there cannot be any existing target by the
   * same name.
   */
  public void addInputFileUnchecked(InputFile file) {
    Target prev = putTargetInternal(file);
    Preconditions.checkState(prev == null);
  }

  /**
   * Inserts an input file into the target map, replacing an existing file by the same name.
   *
   * <p>It is an error if no input file by that name already exists.
   */
  public void replaceInputFileUnchecked(InputFile file) {
    Target prev = putTargetInternal(file);
    Preconditions.checkState(prev instanceof InputFile, prev);
  }

  @Nullable
  public Target getTarget(String name) {
    return targetMap.get(name);
  }

  public void unwrapSnapshottableBiMap() {
    Preconditions.checkState(targetMap instanceof SnapshottableBiMap<?, ?>);
    this.targetMap = ((SnapshottableBiMap<String, Target>) targetMap).getUnderlyingBiMap();
  }

  /**
   * Replaces a target in the {@link Package} under construction with a new target with the same
   * name and belonging to the same package.
   *
   * <p>There must already be an existing target by the same name.
   *
   * <p>Requires that {@link #disableNameConflictChecking} was not called.
   *
   * <p>A hack needed for {@link WorkspaceFactoryHelper}.
   */
  public void replaceTarget(Target newTarget) {
    ensureNameConflictChecking();

    Preconditions.checkArgument(
        targetMap.containsKey(newTarget.getName()),
        "No existing target with name '%s' in the targets map",
        newTarget.getName());
    Target oldTarget = putTargetInternal(newTarget);
    if (newTarget instanceof Rule) {
      List<Label> ruleLabelsForOldTarget = ruleLabels.remove(oldTarget);
      if (ruleLabelsForOldTarget != null) {
        // TODO(brandjon): Can the new target have different labels than the old? If so, we
        // probably need newTarget.getLabels() here instead. Moot if we can delete this along with
        // WORKSPACE logic.
        ruleLabels.put((Rule) newTarget, ruleLabelsForOldTarget);
      }
    }
  }

  // TODO(bazel-team): This method allows target deletion via the returned view, which is used in
  // PackageFunction#handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions.
  // Let's disallow that and make removal go through a dedicated method.
  public Set<Target> getTargets() {
    return targetMap.values();
  }

  /**
   * Returns an {@link Iterable} of all the rule instance targets belonging to this package.
   *
   * <p>The returned {@link Iterable} will be deterministically ordered, in the order the rule
   * instance targets were instantiated.
   */
  public Iterable<Rule> getRules() {
    return Iterables.filter(targetMap.values(), Rule.class);
  }

  /**
   * Turns off (some) conflict checking for name clashes between targets, macros, and output file
   * prefixes. (It is not guaranteed to disable all checks, since it is intended as an optimization
   * and not for semantic effect.)
   *
   * <p>This should only be done for data that has already been validated, e.g. during package
   * deserialization. Do not call this unless you know what you're doing.
   *
   * <p>This method must be called prior to {@link #addRuleUnchecked}. It may not be called, neither
   * before nor after, a call to {@link #addRule} or {@link #replaceTarget}.
   */
  public void disableNameConflictChecking() {
    Preconditions.checkState(nameConflictCheckingPolicy == NameConflictCheckingPolicy.UNKNOWN);
    this.nameConflictCheckingPolicy = NameConflictCheckingPolicy.NOT_GUARANTEED;
    this.ruleLabels = null;
    this.rulesCreatedInMacros = null;
    this.macroNamespaceViolatingTargets = null;
    this.outputFilePrefixes = null;
  }

  public void ensureNameConflictChecking() {
    Preconditions.checkState(
        nameConflictCheckingPolicy != NameConflictCheckingPolicy.NOT_GUARANTEED);
    this.nameConflictCheckingPolicy = NameConflictCheckingPolicy.ENABLED;
  }

  /**
   * Adds a rule and its outputs to the targets map, and propagates the error bit from the rule to
   * the package.
   */
  private void addRuleInternal(Rule rule) {
    for (OutputFile outputFile : rule.getOutputFiles()) {
      putTargetInternal(outputFile);
    }
    putTargetInternal(rule);
    if (rule.containsErrors()) {
      setContainsErrors();
    }
  }

  /**
   * Adds a rule without certain validation checks. Requires that {@link
   * #disableNameConflictChecking} was already called.
   */
  public void addRuleUnchecked(Rule rule) {
    Preconditions.checkState(
        nameConflictCheckingPolicy == NameConflictCheckingPolicy.NOT_GUARANTEED);
    addRuleInternal(rule);
  }

  /**
   * Adds a rule, subject to the usual validation checks. Requires that {@link
   * #disableNameConflictChecking} was not called.
   */
  public void addRule(Rule rule) throws NameConflictException {
    ensureNameConflictChecking();

    List<Label> labels = rule.getLabels();
    checkRuleAndOutputs(rule, labels);
    addRuleInternal(rule);
    ruleLabels.put(rule, labels);
    if (currentMacroFrame != null) {
      rulesCreatedInMacros.add(rule);
    }
  }

  /** Adds a symbolic macro instance to the package. */
  public void addMacro(MacroInstance macro) throws NameConflictException {
    checkMacroName(macro);
    Object prev = macroMap.put(macro.getId(), macro);
    Preconditions.checkState(prev == null);

    // Track whether a main submacro has been seen yet. Conflict checking for this is done in
    // checkMacroName().
    if (currentMacroFrame != null) {
      if (macro.getName().equals(currentMacroFrame.macroInstance.getName())) {
        currentMacroFrame.mainSubmacroHasBeenDefined = true;
      }
    }
  }

  /** Returns the current macro frame, or null if there is no currently running symbolic macro. */
  @Nullable
  public MacroFrame getCurrentMacroFrame() {
    return currentMacroFrame;
  }

  /**
   * Returns true if a symbolic macro is running and the current macro frame is not a rule
   * finalizer.
   *
   * <p>Note that this function examines only the current macro frame, not any parent frames; and
   * thus returns true even if the current non-finalizer macro was called within a finalizer macro.
   */
  public boolean currentlyInNonFinalizerMacro() {
    return currentMacroFrame != null
        && !currentMacroFrame.macroInstance.getMacroClass().isFinalizer();
  }

  /**
   * Returns true if a symbolic macro is running and the current macro frame is a rule finalizer.
   */
  public boolean currentlyInFinalizer() {
    return currentMacroFrame != null
        && currentMacroFrame.macroInstance.getMacroClass().isFinalizer();
  }

  /**
   * Sets the current macro frame and returns the old one.
   *
   * <p>Either the new or old frame may be null, indicating no currently running symbolic macro.
   */
  @Nullable
  public MacroFrame setCurrentMacroFrame(@Nullable MacroFrame frame) {
    MacroFrame prev = currentMacroFrame;
    currentMacroFrame = frame;
    return prev;
  }

  /**
   * Precondition check for {@link #addRule} (to be called before the rule and its outputs are in
   * the targets map). Verifies that:
   *
   * <ul>
   *   <li>The added rule's name, and the names of its output files, are not the same as the name of
   *       any target already declared in the package.
   *   <li>The added rule's output files list does not contain the same name twice.
   *   <li>The added rule does not have an input file and an output file that share the same name.
   *   <li>For each of the added rule's output files, no directory prefix of that file matches the
   *       name of another output file in the package; and conversely, the file is not itself a
   *       prefix for another output file. (This check statefully mutates the {@code
   *       outputFilePrefixes} field.)
   * </ul>
   */
  // TODO(bazel-team): We verify that all prefixes of output files are distinct from other output
  // file names, but not that they're distinct from other target names in the package. What
  // happens if you define an input file "abc" and output file "abc/xyz"?
  private void checkRuleAndOutputs(Rule rule, List<Label> labels) throws NameConflictException {
    Preconditions.checkNotNull(outputFilePrefixes); // ensured by addRule's precondition

    // Check the name of the new rule itself.
    String ruleName = rule.getName();
    checkTargetName(rule);

    ImmutableList<OutputFile> outputFiles = rule.getOutputFiles();
    Map<String, OutputFile> outputFilesByName = Maps.newHashMapWithExpectedSize(outputFiles.size());

    // Check the new rule's output files, both for direct conflicts and prefix conflicts.
    for (OutputFile outputFile : outputFiles) {
      String outputFileName = outputFile.getName();
      // Check for duplicate within a single rule. (Can't use checkTargetName since this rule's
      // outputs aren't in the target map yet.)
      if (outputFilesByName.put(outputFileName, outputFile) != null) {
        throw new NameConflictException(
            String.format(
                "rule '%s' has more than one generated file named '%s'", ruleName, outputFileName));
      }
      // Check for conflict with any other already added target.
      checkTargetName(outputFile);
      // TODO(bazel-team): We also need to check for a conflict between an output file and its own
      // rule, which is not yet in the targets map.

      // Check if this output file is the prefix of an already existing one.
      if (outputFilePrefixes.containsKey(outputFileName)) {
        throw overlappingOutputFilePrefixes(outputFile, outputFilePrefixes.get(outputFileName));
      }

      // Check if a prefix of this output file matches an already existing one.
      PathFragment outputFileFragment = PathFragment.create(outputFileName);
      int segmentCount = outputFileFragment.segmentCount();
      for (int i = 1; i < segmentCount; i++) {
        String prefix = outputFileFragment.subFragment(0, i).toString();
        if (outputFilesByName.containsKey(prefix)) {
          throw overlappingOutputFilePrefixes(outputFile, outputFilesByName.get(prefix));
        }
        if (targetMap.get(prefix) instanceof OutputFile) {
          throw overlappingOutputFilePrefixes(outputFile, (OutputFile) targetMap.get(prefix));
        }

        // Store in persistent map, for checking when adding future rules.
        outputFilePrefixes.putIfAbsent(prefix, outputFile);
      }
    }

    // Check for the same file appearing as both an input and output of the new rule.
    PackageIdentifier packageIdentifier = rule.getLabel().getPackageIdentifier();
    for (Label inputLabel : labels) {
      if (packageIdentifier.equals(inputLabel.getPackageIdentifier())
          && outputFilesByName.containsKey(inputLabel.getName())) {
        throw new NameConflictException(
            String.format(
                "rule '%s' has file '%s' as both an input and an output",
                ruleName, inputLabel.getName()));
      }
    }
  }

  /**
   * Returns whether a given {@code name} is within the namespace that would be owned by a macro
   * called {@code macroName}.
   *
   * <p>This is purely a string operation and does not reference actual targets and macros.
   *
   * <p>A macro named "foo" owns the namespace consisting of "foo" and all "foo_${BAR}",
   * "foo-${BAR}", or "foo.${BAR}", where ${BAR} is a non-empty string. ("_" is the recommended
   * separator; "." is required for file extensions.) This criteria is transitive; a submacro's
   * namespace is a subset of the parent macro's namespace. Therefore, if a name is valid w.r.t. the
   * macro that declares it, it is also valid for all ancestor macros.
   *
   * <p>Note that just because a name is within a macro's namespace does not necessarily mean the
   * corresponding target or macro was declared within this macro.
   */
  public static boolean nameIsWithinMacroNamespace(String name, String macroName) {
    if (name.equals(macroName)) {
      return true;
    } else if (name.startsWith(macroName)) {
      String suffix = name.substring(macroName.length());
      // 0-length suffix handled above.
      if (suffix.length() >= 2
          && (suffix.startsWith("_") || suffix.startsWith(".") || suffix.startsWith("-"))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Throws {@link NameConflictException} if the given target's name can't be added because of a
   * conflict. If the given target's name violates symbolic macro naming rules, this method doesn't
   * throw but instead records that the target's name is in violation, so that an attempt to use the
   * target will fail during the analysis phase.
   *
   * <p>The given target must *not* have already been added.
   *
   * <p>We defer enforcement of symbolic macro naming rules for targets to the analysis phase
   * because otherwise, we could not use java rules (which declare lib%{name}-src.jar implicit
   * outputs) transitively in any symbolic macro.
   */
  // TODO(#19922): Provide a way to allow targets which violate naming rules to be configured
  // (either only as a dep to other targets declared in the current macro, or also externally).
  // TODO(#19922): Ensure `bazel build //pkg:all` (or //pkg:*) ignores violating targets.
  private void checkTargetName(Target target) throws NameConflictException {
    // We only care about the target's name, but we accept the full Target object to produce better
    // error messages.
    checkForExistingTargetName(target);

    checkForExistingMacroName(target.getName(), "target");

    if (currentMacroFrame != null
        && !nameIsWithinMacroNamespace(
            target.getName(), currentMacroFrame.macroInstance.getName())) {
      macroNamespaceViolatingTargets.put(
          target.getName(), currentMacroFrame.macroInstance.getName());
    }
  }

  /**
   * Add all given map entries to the builder's map from names of targets declared in a symbolic
   * macro which violate macro naming rules to the name of the macro instance where they were
   * declared.
   *
   * <p>Intended to be used for package deserialization.
   */
  public void putAllMacroNamespaceViolatingTargets(
      Map<String, String> macroNamespaceViolatingTargets) {
    if (this.macroNamespaceViolatingTargets == null) {
      this.macroNamespaceViolatingTargets = new LinkedHashMap<>();
    }
    this.macroNamespaceViolatingTargets.putAll(macroNamespaceViolatingTargets);
  }

  /**
   * Throws {@link NameConflictException} if the given target's name matches that of an existing
   * target in the package, or an existing macro in the package that is not its ancestor.
   *
   * <p>The given target must *not* have already been added.
   */
  private void checkForExistingTargetName(Target target) throws NameConflictException {
    Target existing = targetMap.get(target.getName());
    if (existing == null) {
      return;
    }

    String subject = String.format("%s '%s'", target.getTargetKind(), target.getName());
    if (target instanceof OutputFile givenOutput) {
      subject += String.format(" in rule '%s'", givenOutput.getGeneratingRule().getName());
    }

    String object =
        existing instanceof OutputFile existingOutput
            ? String.format(
                "generated file from rule '%s'", existingOutput.getGeneratingRule().getName())
            : existing.getTargetKind();
    object += ", defined at " + existing.getLocation();

    throw new NameConflictException(
        String.format("%s conflicts with existing %s", subject, object));
  }

  /**
   * Throws {@link NameConflictException} if the given macro's name can't be added, either because
   * of a conflict or because of a violation of symbolic macro naming rules (if applicable).
   *
   * <p>The given macro must *not* have already been added (via {@link #addMacro}).
   */
  private void checkMacroName(MacroInstance macro) throws NameConflictException {
    String name = macro.getName();

    // A macro can share names with its main target but no other target. Since the macro hasn't
    // even been added yet, it hasn't run, and its main target is not yet defined. Therefore, any
    // match in the targets map represents a real conflict.
    Target existingTarget = targetMap.get(name);
    if (existingTarget != null) {
      throw new NameConflictException(
          String.format("macro '%s' conflicts with an existing target.", name));
    }

    checkForExistingMacroName(name, "macro");

    if (currentMacroFrame != null
        && !nameIsWithinMacroNamespace(name, currentMacroFrame.macroInstance.getName())) {
      throw new MacroNamespaceViolationException(
          String.format(
              "macro '%s' cannot declare submacro named '%s'. %s",
              currentMacroFrame.macroInstance.getName(), name, MACRO_NAMING_RULES));
    }
  }

  /**
   * Throws {@link NameConflictException} if the given name (of a hypothetical target or macro)
   * matches the name of an existing macro in the package, and the existing macro is not currently
   * executing (i.e. on the macro stack).
   *
   * <p>{@code what} must be either "macro" or "target".
   */
  private void checkForExistingMacroName(String name, String what) throws NameConflictException {
    // Macros are indexed by id, not name, so we can't just use macroMap.get() directly.
    // Instead, we reason that if at least one macro by the given name exists, then there is one
    // with an id suffix of ":1".
    MacroInstance existing = macroMap.get(name + ":1");
    if (existing == null) {
      return;
    }

    // A conflict is still ok if it's only with enclosing macros. It's enough to check that 1) we
    // have the same name as the immediately enclosing macro (relying inductively on the check
    // that was done when that macro was added), and 2) there is no sibling macro of the same name
    // already defined in the current frame.
    if (currentMacroFrame != null) {
      if (name.equals(currentMacroFrame.macroInstance.getName())
          && !currentMacroFrame.mainSubmacroHasBeenDefined) {
        return;
      }
    }

    // TODO(#19922): Add definition location info for the existing object, like we have in
    // checkForExistingTargetName. Complicated by the fact that there may be more than one macro
    // of that name.
    throw new NameConflictException(
        String.format(
            "%s '%s' conflicts with an existing macro (and was not created by it)", what, name));
  }

  /**
   * Returns a {@link NameConflictException} about two output files clashing (i.e., due to one being
   * a prefix of the other)
   */
  private static NameConflictException overlappingOutputFilePrefixes(
      OutputFile added, OutputFile existing) {
    if (added.getGeneratingRule() == existing.getGeneratingRule()) {
      return new NameConflictException(
          String.format(
              "rule '%s' has conflicting output files '%s' and '%s'",
              added.getGeneratingRule().getName(), added.getName(), existing.getName()));
    } else {
      return new NameConflictException(
          String.format(
              "output file '%s' of rule '%s' conflicts with output file '%s' of rule '%s'",
              added.getName(),
              added.getGeneratingRule().getName(),
              existing.getName(),
              existing.getGeneratingRule().getName()));
    }
  }

  /**
   * An exception used when the name of a target or symbolic macro clashes with another entity
   * defined in the package.
   *
   * <p>Common examples of conflicts include two targets or symbolic macros sharing the same name,
   * and one output file being a prefix of another. See {@link Package.Builder#checkForExistingName}
   * and {@link Package.Builder#checkRuleAndOutputs} for more details.
   */
  public static sealed class NameConflictException extends Exception
      permits MacroNamespaceViolationException {
    public NameConflictException(String message) {
      super(message);
    }
  }

  /**
   * An exception used when the name of a target or submacro declared within a symbolic macro
   * violates symbolic macro naming rules.
   *
   * <p>An example might be a target named "libfoo" declared within a macro named "foo".
   */
  public static final class MacroNamespaceViolationException extends NameConflictException {
    public MacroNamespaceViolationException(String message) {
      super(message);
    }
  }
}
