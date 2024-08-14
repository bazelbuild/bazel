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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Package.Builder.MacroFrame;
import com.google.devtools.build.lib.packages.TargetDefinitionContext.NameConflictException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.spelling.SpellChecker;

/**
 * Represents a symbolic macro, defined in a .bzl file, that may be instantiated during Package
 * evaluation.
 *
 * <p>This is analogous to {@link RuleClass}. In essence, a {@code MacroClass} consists of the
 * macro's schema and its implementation function.
 */
public final class MacroClass {

  /**
   * Names that users may not pass as keys of the {@code attrs} dict when calling {@code macro()}.
   *
   * <p>Of these, {@code name} is special cased as an actual attribute, and the rest do not exist.
   */
  // Keep in sync with `macro()`'s `attrs` user documentation in StarlarkRuleFunctionsApi.
  public static final ImmutableSet<String> RESERVED_MACRO_ATTR_NAMES =
      ImmutableSet.of("name", "visibility", "deprecation", "tags", "testonly", "features");

  private final String name;
  private final StarlarkFunction implementation;
  // Implicit attributes are stored under their given name ("_foo"), not a mangled name ("$foo").
  private final ImmutableMap<String, Attribute> attributes;
  private final boolean isFinalizer;

  public MacroClass(
      String name,
      StarlarkFunction implementation,
      ImmutableMap<String, Attribute> attributes,
      boolean isFinalizer) {
    this.name = name;
    this.implementation = implementation;
    this.attributes = attributes;
    this.isFinalizer = isFinalizer;
  }

  /** Returns the macro's exported name. */
  public String getName() {
    return name;
  }

  public StarlarkFunction getImplementation() {
    return implementation;
  }

  public ImmutableMap<String, Attribute> getAttributes() {
    return attributes;
  }

  /**
   * Returns whether this symbolic macro is a finalizer. All finalizers are run deferred to the end
   * of the BUILD file's evaluation, rather than synchronously with their instantiation.
   */
  public boolean isFinalizer() {
    return isFinalizer;
  }

  /** Builder for {@link MacroClass}. */
  public static final class Builder {
    @Nullable private String name = null;
    private final StarlarkFunction implementation;
    private final ImmutableMap.Builder<String, Attribute> attributes = ImmutableMap.builder();
    private boolean isFinalizer = false;

    public Builder(StarlarkFunction implementation) {
      this.implementation = implementation;
    }

    @CanIgnoreReturnValue
    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addAttribute(Attribute attribute) {
      attributes.put(attribute.getName(), attribute);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setIsFinalizer() {
      this.isFinalizer = true;
      return this;
    }

    public MacroClass build() {
      Preconditions.checkNotNull(name);
      return new MacroClass(
          name, implementation, attributes.buildOrThrow(), /* isFinalizer= */ isFinalizer);
    }
  }

  /**
   * Constructs and returns a new {@link MacroInstance} associated with this {@code MacroClass}.
   *
   * <p>See {@link #instantiateAndAddMacro}.
   */
  // TODO(#19922): Consider reporting multiple events instead of failing on the first one. See
  // analogous implementation in RuleClass#populateDefinedRuleAttributeValues.
  private MacroInstance instantiateMacro(Package.Builder pkgBuilder, Map<String, Object> kwargs)
      throws EvalException {
    // A word on edge cases:
    //   - If an attr is implicit but does not have a default specified, its value is just the
    //     default value for its attr type (e.g. `[]` for `attr.label_list()`).
    //   - If an attr is implicit but also mandatory, it's impossible to instantiate it without
    //     error.
    //   - If an attr is mandatory but also has a default, the default is meaningless.
    // These behaviors align with rule attributes.

    LinkedHashMap<String, Object> attrValues = new LinkedHashMap<>();

    // For each given attr value, validate that the attr exists and can be set.
    for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
      String attrName = entry.getKey();
      Object value = entry.getValue();
      Attribute attr = attributes.get(attrName);

      // Check for unknown attr.
      if (attr == null) {
        throw Starlark.errorf(
            "no such attribute '%s' in '%s' macro%s",
            attrName,
            name,
            SpellChecker.didYouMean(
                attrName,
                attributes.values().stream()
                    .filter(Attribute::isDocumented)
                    .map(Attribute::getName)
                    .collect(toImmutableList())));
      }

      // Setting an attr to None is the same as omitting it (except that it's still an error to set
      // an unknown attr to None).
      if (value == Starlark.NONE) {
        continue;
      }

      // Can't set implicit default.
      // (We don't check Attribute#isImplicit() because that assumes "_" -> "$" prefix mangling.)
      if (attr.getName().startsWith("_")) {
        throw Starlark.errorf("cannot set value of implicit attribute '%s'", attr.getName());
      }

      attrValues.put(attrName, value);
    }

    // Populate defaults for the rest, and validate that no mandatory attr was missed.
    for (Attribute attr : attributes.values()) {
      if (attrValues.containsKey(attr.getName())) {
        continue;
      }
      if (attr.isMandatory()) {
        throw Starlark.errorf(
            "missing value for mandatory attribute '%s' in '%s' macro", attr.getName(), name);
      } else {
        // Already validated at schema creation time that the default is not a computed default or
        // late-bound default
        attrValues.put(attr.getName(), attr.getDefaultValueUnchecked());
      }
    }

    // Normalize and validate all attr values. (E.g., convert strings to labels, fail if bool was
    // passed instead of label, ensure values are immutable.)
    for (Map.Entry<String, Object> entry : ImmutableMap.copyOf(attrValues).entrySet()) {
      String attrName = entry.getKey();
      Attribute attribute = attributes.get(attrName);
      Object normalizedValue =
          // copyAndLiftStarlarkValue ensures immutability.
          BuildType.copyAndLiftStarlarkValue(
              name, attribute, entry.getValue(), pkgBuilder.getLabelConverter());
      // TODO(#19922): Validate that LABEL_LIST type attributes don't contain duplicates, to match
      // the behavior of rules. This probably requires factoring out logic from
      // AggregatingAttributeMapper.
      if (attribute.isConfigurable() && !(normalizedValue instanceof SelectorList)) {
        normalizedValue = SelectorList.wrapSingleValue(normalizedValue);
      }
      attrValues.put(attrName, normalizedValue);
    }

    // Type and existence enforced by RuleClass.NAME_ATTRIBUTE.
    String name = (String) Preconditions.checkNotNull(attrValues.get("name"));
    // Determine the id for this macro. If we're in another macro by the same name, increment the
    // number, otherwise use 1 for the number.
    @Nullable MacroFrame parentMacroFrame = pkgBuilder.getCurrentMacroFrame();
    int sameNameDepth =
        parentMacroFrame == null || !name.equals(parentMacroFrame.macroInstance.getName())
            ? 1
            : parentMacroFrame.macroInstance.getSameNameDepth() + 1;

    return new MacroInstance(this, attrValues, sameNameDepth);
  }

  /**
   * Constructs a new {@link MacroInstance} associated with this {@code MacroClass}, adds it to the
   * package, and returns it.
   *
   * @param pkgBuilder The builder corresponding to the package in which this instance will live.
   * @param kwargs A map from attribute name to its given Starlark value, such as passed in a BUILD
   *     file (i.e., prior to attribute type conversion, {@code select()} promotion, default value
   *     substitution, or even validation that the attribute exists).
   */
  public MacroInstance instantiateAndAddMacro(
      Package.Builder pkgBuilder, Map<String, Object> kwargs) throws EvalException {
    MacroInstance macroInstance = instantiateMacro(pkgBuilder, kwargs);
    try {
      pkgBuilder.addMacro(macroInstance);
    } catch (NameConflictException e) {
      throw new EvalException(e);
    }
    return macroInstance;
  }

  /**
   * Executes a symbolic macro's implementation function, in a new Starlark thread, mutating the
   * given package under construction.
   */
  // TODO: #19922 - Take a new type, PackagePiece.Builder, in place of Package.Builder. PackagePiece
  // would represent the collection of targets/macros instantiated by expanding a single symbolic
  // macro.
  public static void executeMacroImplementation(
      MacroInstance macro, Package.Builder builder, StarlarkSemantics semantics)
      throws InterruptedException {
    try (Mutability mu =
        Mutability.create("macro", builder.getPackageIdentifier(), macro.getName())) {
      StarlarkThread thread =
          StarlarkThread.create(
              mu,
              semantics,
              /* contextDescription= */ "",
              SymbolGenerator.create(
                  MacroClassId.create(builder.getPackageIdentifier(), macro.getName())));
      thread.setPrintHandler(Event.makeDebugPrintHandler(builder.getLocalEventHandler()));

      // TODO: #19922 - Technically the embedded SymbolGenerator field should use a different key
      // than the one in the main BUILD thread, but that'll be fixed when we change the type to
      // PackagePiece.Builder.
      builder.storeInThread(thread);

      // TODO: #19922 - If we want to support creating analysis_test rules inside symbolic macros,
      // we'd need to call `thread.setThreadLocal(RuleDefinitionEnvironment.class,
      // ruleClassProvider)`. In that case we'll need to consider how to get access to the
      // ConfiguredRuleClassProvider. For instance, we could put it in the builder.

      MacroFrame childMacroFrame = new MacroFrame(macro);
      @Nullable MacroFrame parentMacroFrame = builder.setCurrentMacroFrame(childMacroFrame);
      try {
        Starlark.call(
            thread,
            macro.getMacroClass().getImplementation(),
            /* args= */ ImmutableList.of(),
            /* kwargs= */ macro.getAttrValues());
      } catch (EvalException ex) {
        builder
            .getLocalEventHandler()
            .handle(
                Package.error(
                    /* location= */ null, ex.getMessageWithStack(), Code.STARLARK_EVAL_ERROR));
        builder.setContainsErrors();
      } finally {
        // Restore the previously running symbolic macro's state (if any).
        @Nullable MacroFrame top = builder.setCurrentMacroFrame(parentMacroFrame);
        Preconditions.checkState(top == childMacroFrame, "inconsistent macro stack state");
        // Mark the macro as having completed, even if it was in error (or interrupted?).
        builder.markMacroComplete(macro);
      }
    }
  }

  @AutoValue
  abstract static class MacroClassId {
    static MacroClassId create(PackageIdentifier id, String name) {
      return new AutoValue_MacroClass_MacroClassId(id, name);
    }

    abstract PackageIdentifier packageId();

    abstract String name();
  }
}
