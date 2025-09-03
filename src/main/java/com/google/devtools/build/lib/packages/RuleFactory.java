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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Attribute.StarlarkComputedDefaultTemplate.CannotPrecomputeDefaultsException;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TargetRecorder.NameConflictException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;
import net.starlark.java.eval.Tuple;

/**
 * Static utility class for defining Starlark callables for builtin rules (i.e., {@link
 * RuleFunction} objects for builtin rules' {@link RuleClass} objects), and instantiating those
 * rules to produce targets (i.e., {@link Rule} objects).
 */
public class RuleFactory {

  private RuleFactory() {} // uninstantiable

  /**
   * Creates and returns a rule instance.
   *
   * <p>It is the caller's responsibility to add the rule to the package (the caller may choose not
   * to do so if, for example, the rule has errors).
   *
   * @param callstack the stack of the Starlark thread where the rule is created. In the case of
   *     rules instantiated by a symbolic macro, this is the inner macro's stack, and does not
   *     include frames for enclosing macros or the BUILD file.
   */
  public static Rule createRule(
      TargetDefinitionContext targetDefinitionContext,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      boolean failOnUnknownAttributes,
      ImmutableList<StarlarkThread.CallStackEntry> callstack)
      throws InvalidRuleException, InterruptedException {
    Preconditions.checkNotNull(ruleClass);
    String ruleClassName = ruleClass.getName();
    Object nameObject = attributeValues.getAttributeValue("name");
    if (nameObject == null) {
      throw new InvalidRuleException(ruleClassName + " rule has no 'name' attribute");
    } else if (!(nameObject instanceof String)) {
      throw new InvalidRuleException(ruleClassName + " 'name' attribute must be a string");
    }
    String name = (String) nameObject;
    Label label;
    try {
      // Test that this would form a valid label name -- in particular, this
      // catches cases where Makefile variables $(foo) appear in "name".
      label = targetDefinitionContext.createLabel(name);
    } catch (LabelSyntaxException e) {
      throw new InvalidRuleException("illegal rule name: " + name + ": " + e.getMessage());
    }

    // Add the generator_name attribute.
    BuildLangTypedAttributeValuesMap processedAttributes;
    @Nullable
    String generatorName = getGeneratorName(targetDefinitionContext, attributeValues, callstack);
    // Don't bother copying anything if nothing changed.
    if (generatorName != null) {
      ImmutableMap.Builder<String, Object> builder =
          ImmutableMap.builderWithExpectedSize(attributeValues.attributeValues.size() + 1);
      builder.putAll(attributeValues.attributeValues);
      builder.put("generator_name", generatorName);
      processedAttributes = new BuildLangTypedAttributeValuesMap(builder.buildKeepingLast());
    } else {
      processedAttributes = attributeValues;
    }

    // The raw stack is of the form [<toplevel>@BUILD:1, macro@lib.bzl:1, cc_library@<builtin>].
    // Pop the innermost frame for the rule, since it's obvious.
    callstack = callstack.subList(0, callstack.size() - 1); // pop

    try {
      return ruleClass.createRule(
          targetDefinitionContext, label, processedAttributes, failOnUnknownAttributes, callstack);
    } catch (LabelSyntaxException | CannotPrecomputeDefaultsException e) {
      throw new RuleFactory.InvalidRuleException(ruleClass + " " + e.getMessage());
    }
  }

  /**
   * Creates a {@link Rule} instance, adds it to the {@link Package.Builder} and returns it.
   *
   * @param pkgBuilder the under-construction {@link Package.Builder} to which the rule belongs
   * @param ruleClass the {@link RuleClass} of the rule
   * @param attributeValues a {@link BuildLangTypedAttributeValuesMap} mapping attribute names to
   *     attribute values of build-language type. Each attribute must be defined for this class of
   *     rule, and have a build-language-typed value which can be converted to the appropriate
   *     native type of the attribute (i.e. via {@link BuildType#convertFromBuildLangType}). There
   *     must be a map entry for each non-optional attribute of this class of rule.
   * @param eventHandler a eventHandler on which errors and warnings are reported during rule
   *     creation
   * @param callstack the stack of active calls in the Starlark thread
   * @throws InvalidRuleException if the rule could not be constructed for any reason (e.g. no
   *     {@code name} attribute is defined)
   * @throws NameConflictException if the rule's name or output files conflict with others in this
   *     package
   * @throws InterruptedException if interrupted
   */
  @CanIgnoreReturnValue
  public static Rule createAndAddRule(
      TargetDefinitionContext targetDefinitionContext,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      boolean failOnUnknownAttributes,
      ImmutableList<StarlarkThread.CallStackEntry> callstack)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    Rule rule =
        createRule(
            targetDefinitionContext,
            ruleClass,
            attributeValues,
            failOnUnknownAttributes,
            callstack);
    targetDefinitionContext.addRule(rule);
    return rule;
  }

  /**
   * InvalidRuleException is thrown by {@link Rule} creation methods if the {@link Rule} could not
   * be constructed. It contains an error message.
   */
  public static class InvalidRuleException extends Exception {
    public InvalidRuleException(String message) {
      super(message);
    }
  }

  /**
   * A wrapper around an map of named attribute values that specifies whether the map's values are
   * of "build-language" or of "native" types.
   */
  public interface AttributeValues<T> {
    /**
     * Returns {@code true} if all the map's values are "build-language typed", i.e., resulting from
     * the evaluation of an expression in the build language. Returns {@code false} if all the map's
     * values are "natively typed", i.e. of a type returned by {@link
     * BuildType#convertFromBuildLangType}.
     */
    boolean valuesAreBuildLanguageTyped();

    Iterable<T> getAttributeAccessors();

    String getName(T attributeAccessor);

    Object getValue(T attributeAccessor);

    boolean isExplicitlySpecified(T attributeAccessor);
  }

  /** A {@link AttributeValues} of explicit "build-language" values. */
  public static final class BuildLangTypedAttributeValuesMap
      implements AttributeValues<Map.Entry<String, Object>> {
    private final Map<String, Object> attributeValues;

    public BuildLangTypedAttributeValuesMap(Map<String, Object> attributeValues) {
      this.attributeValues = attributeValues;
    }

    private boolean containsAttributeNamed(String attributeName) {
      return attributeValues.containsKey(attributeName);
    }

    private Object getAttributeValue(String attributeName) {
      return attributeValues.get(attributeName);
    }

    @Override
    public boolean valuesAreBuildLanguageTyped() {
      return true;
    }

    @Override
    public Set<Map.Entry<String, Object>> getAttributeAccessors() {
      return attributeValues.entrySet();
    }

    @Override
    public String getName(Map.Entry<String, Object> attributeAccessor) {
      return attributeAccessor.getKey();
    }

    @Override
    public Object getValue(Map.Entry<String, Object> attributeAccessor) {
      return attributeAccessor.getValue();
    }

    @Override
    public boolean isExplicitlySpecified(Map.Entry<String, Object> attributeAccessor) {
      return true;
    }
  }

  /**
   * Given the call stack and attribute values of a rule being instantiated, computes and returns
   * the value of the special {@code generator_name} attribute to be added, or returns null if it
   * shouldn't be added.
   *
   * <p>The {@code generator_name} attribute is set for targets instantiated within a legacy macro
   * (and which are not also within a symbolic macro). Its value is the name argument of the
   * top-level macro on the call stack, if its value can be determined statically (see {@link
   * PackageFactory#checkBuildSyntax}), or just the name of the target otherwise.
   *
   * @param callstack the stack of the Starlark thread where the rule is created. In the case of
   *     rules instantiated by a symbolic macro, this is the inner macro's stack, and does not
   *     include frames for enclosing macros or the BUILD file.
   */
  @Nullable
  private static String getGeneratorName(
      TargetDefinitionContext targetDefinitionContext,
      BuildLangTypedAttributeValuesMap args,
      ImmutableList<CallStackEntry> callstack) {
    @Nullable MacroInstance macro = targetDefinitionContext.currentMacro();
    // The "generator" of a rule is the function outermost in the BUILD file thread's call stack
    // (regardless of whether or not it was passed a "name" parameter). For rules with generators,
    // the BUILD file thread's call stack must contain at least two entries:
    //   0: the outermost function (BUILD file top level),
    //   1: the function called by it (e.g. a macro in a .bzl file),
    // optionally followed by other Starlark or built-in functions, and finally - if the rule is
    // instantiated in the BUILD file thread - the rule instantiation function.
    if (macro == null
        && (callstack.size() < 2 || !callstack.get(1).location.file().endsWith(".bzl"))) {
      // We are in a BUILD file thread, and the rule is being instantiated directly at the top
      // level of the BUILD file.
      // TODO(bazel-team): Tolerate ".scl" extension in the above if? An .scl file can instantiate a
      // rule if the rule function is passed as an argument.
      return null;
    }

    if (args.containsAttributeNamed("generator_name")) {
      // generator_name is explicitly set. Don't override it.
      // TODO(b/274802222): Should this be prohibited?
      return null;
    }

    String generatorName =
        macro != null
            ? macro.getGeneratorName()
            : targetDefinitionContext.getGeneratorNameByLocation(callstack.get(0).location);
    if (generatorName == null) {
      // Fall back on target name (meh).
      generatorName = (String) args.getAttributeValue("name");
    }
    return generatorName;
  }

  /**
   * Builds a map from rule names to (newly constructed)) Starlark callables that instantiate them.
   */
  public static ImmutableMap<String, BuiltinRuleFunction> buildRuleFunctions(
      Map<String, RuleClass> ruleClassMap) {
    ImmutableMap.Builder<String, BuiltinRuleFunction> result = ImmutableMap.builder();
    for (String ruleClassName : ruleClassMap.keySet()) {
      RuleClass cl = ruleClassMap.get(ruleClassName);
      if (cl.getRuleClassType() == RuleClassType.NORMAL
          || cl.getRuleClassType() == RuleClassType.TEST
          || cl.getRuleClassType() == RuleClassType.BUILD_ONLY) {
        result.put(ruleClassName, new BuiltinRuleFunction(cl));
      }
    }
    return result.buildOrThrow();
  }

  /** A callable Starlark value that creates Rules for native RuleClasses. */
  // TODO(adonovan): why is this distinct from RuleClass itself?
  // Make RuleClass implement StarlarkCallable directly.
  private static class BuiltinRuleFunction implements RuleFunction {
    private final RuleClass ruleClass;

    BuiltinRuleFunction(RuleClass ruleClass) {
      this.ruleClass = Preconditions.checkNotNull(ruleClass);
    }

    @Override
    public NoneType call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw Starlark.errorf("unexpected positional arguments");
      }
      try {
        TargetDefinitionContext targetDefinitionContext =
            switch (ruleClass.getRuleClassType()) {
              case BUILD_ONLY ->
                  Package.AbstractBuilder.fromOrFailAllowBuildOnly(
                      thread, String.format("%s rule", ruleClass.getName()), "instantiated");
              default -> TargetDefinitionContext.fromOrFail(thread, "a rule", "instantiated");
            };
        RuleFactory.createAndAddRule(
            targetDefinitionContext,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(kwargs),
            thread
                .getSemantics()
                .getBool(BuildLanguageOptions.INCOMPATIBLE_FAIL_ON_UNKNOWN_ATTRIBUTES),
            thread.getCallStack());
      } catch (RuleFactory.InvalidRuleException | NameConflictException e) {
        throw new EvalException(e);
      }
      return Starlark.NONE;
    }

    @Override
    public RuleClass getRuleClass() {
      return ruleClass;
    }

    @Override
    public String getName() {
      return ruleClass.getName();
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<built-in rule " + getName() + ">");
    }

    @Override
    public String toString() {
      return getName() + "(...)";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }
  }
}
