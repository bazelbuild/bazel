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

import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildType.SelectorList;
import com.google.devtools.build.lib.packages.Package.Declarations;
import com.google.devtools.build.lib.util.HashCodes;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Represents a use of a symbolic macro in a package.
 *
 * <p>There is one {@code MacroInstance} for each call to a {@link
 * StarlarkRuleClassFunctions#MacroFunction} that is executed during a package's evaluation. Just as
 * a {@link MacroClass} is analogous to a {@link RuleClass}, {@code MacroInstance} is analogous to a
 * {@link Rule} (i.e. a rule target).
 *
 * <p>Macro instance names are not guaranteed to be unique within a package; see {@link #getId}.
 */
public final class MacroInstance extends RuleOrMacroInstance {

  // TODO: #19922 - If we want to save the cost of a field here, we can merge pkg and parent into a
  // single field of type Object, and walk up the parent hierarchy to answer getPackage() queries.
  private final Package.Metadata packageMetadata;

  // TODO(bazel-team): This is only needed for RuleOrMacroInstance#getPackageDeclarations(), which
  // is used by the attribute mapper logic. That might only be needed for rules rather than macros.
  // Consider removing it and pushing getPackageDeclarations() down to Rule.
  private final Package.Declarations packageDeclarations;

  // TODO(https://github.com/bazelbuild/bazel/issues/26128): replace with a parent identifier. The
  // existence of a parent pointer prevents change pruning on outer macro instances, forcing an
  // unconditional re-evaluation of all inner macros when an outer macro is invalidated.
  @Nullable private final MacroInstance parent;

  // Null if this symbolic macro was instantiated as a result of a legacy macro call without a
  // "name" parameter made at the top level of a BUILD file.
  @Nullable private final String generatorName;

  // TODO(https://github.com/bazelbuild/bazel/issues/26128): move location and Starlark stack to the
  // owning PackagePiece to make MacroInstance more change pruning friendly; we don't want the macro
  // to be invalidated if line numbers in a BUILD file or an ancestor macro's definition .bzl file
  // change.
  private final Location buildFileLocation;
  private final CallStack.Node parentCallStack;

  private final MacroClass macroClass;

  private final int sameNameDepth;

  /**
   * Instantiates the given macro class.
   *
   * <p>{@code sameNameDepth} is the number of macro instances that this one is inside of that share
   * its name. For most instances it is 1, but for the main submacro of a parent macro it is one
   * more than the parent's depth.
   */
  MacroInstance(
      Package.Metadata packageMetadata,
      Declarations packageDeclarations,
      @Nullable MacroInstance parent,
      @Nullable String generatorName,
      Location buildFileLocation,
      CallStack.Node parentCallStack,
      MacroClass macroClass,
      Label label,
      int sameNameDepth) {
    super(label, macroClass.getAttributeProvider().getAttributeCount());
    this.packageMetadata = packageMetadata;
    this.packageDeclarations = packageDeclarations;
    this.parent = parent;
    this.generatorName = generatorName;
    this.buildFileLocation = buildFileLocation;
    this.parentCallStack = parentCallStack;
    this.macroClass = macroClass;
    Preconditions.checkArgument(sameNameDepth > 0);
    this.sameNameDepth = sameNameDepth;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    // TODO(https://github.com/bazelbuild/bazel/issues/26128): consider comparing digests instead.
    return obj instanceof MacroInstance other
        && super.equalsHelper(other)
        && Objects.equals(packageMetadata, other.packageMetadata)
        && Objects.equals(packageDeclarations, other.packageDeclarations)
        && Objects.equals(parent, other.parent)
        && Objects.equals(generatorName, other.generatorName)
        && Objects.equals(buildFileLocation, other.buildFileLocation)
        && Objects.equals(parentCallStack, other.parentCallStack)
        && Objects.equals(macroClass, other.macroClass)
        && sameNameDepth == other.sameNameDepth;
  }

  @Override
  public int hashCode() {
    return super.hashCodeHelper()
        + HashCodes.MULTIPLIER
            * HashCodes.hashObjects(
                packageMetadata,
                packageDeclarations,
                parent,
                generatorName,
                buildFileLocation,
                parentCallStack,
                macroClass,
                sameNameDepth);
  }

  @Override
  public Package.Metadata getPackageMetadata() {
    return packageMetadata;
  }

  @Override
  Declarations getPackageDeclarations() {
    return packageDeclarations;
  }

  /**
   * Returns the macro instance that instantiated this one, or null if this was created directly
   * during BUILD evaluation.
   */
  // TODO(bazel-team): Consider merging into getDeclaringMacro().
  // TODO(https://github.com/bazelbuild/bazel/issues/26128): Avoid new uses of this method; it is
  // hostile to change pruning for lazy macro expansion. Replace with a method that either returns
  // the parent identifier, or takes a context argument that allows retrieving the parent by id.
  @Nullable
  public MacroInstance getParent() {
    return parent;
  }

  // TODO(https://github.com/bazelbuild/bazel/issues/26128): Avoid new uses of this method; it is
  // hostile to change pruning for lazy macro expansion. Replace with a method that either returns
  // the parent identifier, or takes a context argument that allows retrieving the parent by id.
  @Override
  @Nullable
  public MacroInstance getDeclaringMacro() {
    return parent;
  }

  /**
   * Returns the location in the BUILD file at which this macro was created or its outermost
   * enclosing symbolic or legacy macro was called.
   */
  public Location getBuildFileLocation() {
    return buildFileLocation;
  }

  /**
   * Returns the value of the "name" parameter of the top-level call in a BUILD file which resulted
   * in this macro being instantiated.
   *
   * <p>This is either the "name" attribute of this macro's outermost symbolic macro ancestor, if it
   * was defined directly at the top level of a BUILD file; or the "name" parameter of the outermost
   * legacy macro wrapping it.
   *
   * <p>Null if this symbolic macro was instantiated as a result of a legacy macro call without a
   * "name" parameter made at the top level of a BUILD file.
   */
  @Nullable
  public String getGeneratorName() {
    return generatorName;
  }

  /**
   * Returns the call stack of the Starlark thread that created this macro instance.
   *
   * <p>If this macro was instantiated in a BUILD file thread (as contrasted with a symbolic macro
   * thread), the call stack does not include the frame for the BUILD file top level, since it's
   * redundant with {@link #getBuildFileLocation}.
   */
  CallStack.Node getParentCallStack() {
    return parentCallStack;
  }

  /**
   * Returns the call stack of the Starlark thread that created this macro instance.
   *
   * <p>Requires reconstructing the call stack from a compact representation, so should only be
   * called when the full call stack is needed.
   */
  @VisibleForTesting
  public ImmutableList<StarlarkThread.CallStackEntry> reconstructParentCallStack() {
    ImmutableList.Builder<StarlarkThread.CallStackEntry> stack = ImmutableList.builder();
    if (parent == null) {
      stack.add(StarlarkThread.callStackEntry(StarlarkThread.TOP_LEVEL, buildFileLocation));
    }
    for (CallStack.Node node = parentCallStack; node != null; node = node.next()) {
      stack.add(node.toCallStackEntry());
    }
    return stack.build();
  }

  /** Returns the {@link MacroClass} (i.e. schema info) that this instance parameterizes. */
  public MacroClass getMacroClass() {
    return macroClass;
  }

  /**
   * The depth of this macro instance in a chain of nested macros having the same name.
   *
   * <p>1 for any macro that is not declared in a macro of the same name.
   *
   * <p>Used by {@link #getId}.
   */
  public int getSameNameDepth() {
    return sameNameDepth;
  }

  /**
   * Returns the id of this macro instance. The id is the name, concatenated with {@code ":n"} where
   * n is an integer distinguishing this from other macro instances of the same name in the package.
   *
   * <p>Within a package, two macro instances are not allowed to share the same name except when one
   * of them is the main submacro of the other. More generally, there may be a contiguous chain of
   * nested main submacros that all share the same name, but these may not share with any other
   * macro outside the chain. We allow this exception so that the build does not break if the rule
   * of a main target is refactored into a macro. The tradeoff of this design is that the name alone
   * is not enough to disambiguate between macros in the chain.
   *
   * <p>The number n is simply the depth of the macro in the chain of same-named macros, starting at
   * 1. For example, if we have a chain of macro expansions foo -> foo_bar -> foo_bar -> foo_bar ->
   * foo_bar_baz, then the ids of these macros are respectively "foo:1", "foo_bar:1", "foo_bar:2",
   * "foo_bar:3", "foo_bar_baz:1".
   *
   * <p>Note that ids only serve to canonically identify macro instances, and play no role in naming
   * or name conflict detection.
   */
  public String getId() {
    return getName() + ":" + sameNameDepth;
  }

  @Override
  public RuleVisibility getDefaultVisibility() {
    return RuleVisibility.parseUnchecked(
        ImmutableList.of(
            Label.createUnvalidated(
                macroClass.getDefiningBzlLabel().getPackageIdentifier(), "__pkg__")));
  }

  /**
   * Returns the visibility of this macro instance, analogous to {@link Target#getActualVisibility}.
   *
   * <p>This value will be observed as the {@code visibility} parameter of the implementation
   * function. It is not necessarily the same as the {@code visibility} value passed in when
   * instantiating the macro, since the latter needs processing to add the call site's location and
   * possibly apply the package's default visibility.
   *
   * <p>It can be assumed that the returned list satisfies {@link RuleVisibility#validate}.
   */
  public ImmutableList<Label> getActualVisibility() {
    @SuppressWarnings("unchecked")
    List<Label> visibility = (List<Label>) Preconditions.checkNotNull(getAttr("visibility"));
    return ImmutableList.copyOf(visibility);
  }

  /**
   * Returns the package containing the .bzl file from which this macro instance's macro class was
   * exported.
   *
   * <p>This is considered to be the place where the macro's code lives, and is used as the place
   * where a target is instantiated for the purposes of Macro-Aware Visibility.
   */
  public PackageIdentifier getDefinitionPackage() {
    return macroClass.getDefiningBzlLabel().getPackageIdentifier();
  }

  /**
   * Visits all labels appearing in non-implicit attributes of {@link Type.LabelClass#DEPENDENCY}
   * label type, i.e. ignoring nodep labels.
   *
   * <p>This is useful for checking whether a given label was passed as an input to this macro by
   * the caller, which in turn is needed in order to decide whether the caller delegated a
   * visibility privilege to us.
   */
  public void visitExplicitAttributeLabels(Consumer<Label> consumer) {
    for (Attribute attribute : macroClass.getAttributeProvider().getAttributes()) {
      String name = attribute.getName();
      Type<?> type = attribute.getType();
      if (name.startsWith("_")) {
        continue;
      }
      if (type.getLabelClass() != Type.LabelClass.DEPENDENCY) {
        continue;
      }
      Object value = getAttr(name, type);
      visitAttributeLabels(value, type, attribute, consumer);
    }
  }

  // Separate method needed to satisfy type system w.r.t. Type<T>.
  // `value` is either a T or SelectorList<T>.
  private <T> void visitAttributeLabels(
      Object value, Type<T> type, Attribute attribute, Consumer<Label> consumer) {
    if (value == null) {
      return;
    }

    Type.LabelVisitor visitor =
        (label, unusedAttribute) -> {
          if (label != null) {
            consumer.accept(label);
          }
        };

    if (value instanceof SelectorList) {
      @SuppressWarnings("unchecked") // safe by precondition assumption
      SelectorList<T> selectorList = (SelectorList<T>) value;
      AggregatingAttributeMapper.visitLabelsInSelect(
          selectorList,
          attribute,
          type,
          visitor,
          /* rule= */ null, // safe because late-bound defaults aren't a thing for macros
          /* includeKeys= */ false,
          /* includeValues= */ true);
    } else {
      T castValue = type.cast(value);
      type.visitLabels(visitor, castValue, attribute);
    }
  }

  @Override
  public AttributeProvider getAttributeProvider() {
    return macroClass.getAttributeProvider();
  }

  @Override
  void reportError(String message, EventHandler eventHandler) {
    eventHandler.handle(Event.error(message));
  }

  @Override
  public boolean isRuleInstance() {
    return false;
  }

  @Override
  public boolean isRuleCreatedInMacro() {
    return false;
  }

  /**
   * Logical tuple of the package and id within the package. Used to label the Starlark evaluation
   * environment.
   */
  record UniqueId(PackageIdentifier packageId, String id) {
    UniqueId {
      requireNonNull(packageId, "packageId");
      requireNonNull(id, "id");
    }

    static UniqueId create(PackageIdentifier packageId, String id) {
      return new UniqueId(packageId, id);
    }
  }
}
