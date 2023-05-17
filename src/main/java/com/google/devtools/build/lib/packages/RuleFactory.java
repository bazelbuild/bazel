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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.StarlarkComputedDefaultTemplate.CannotPrecomputeDefaultsException;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.Set;
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
   */
  public static Rule createRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      EventHandler eventHandler,
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
      label = pkgBuilder.createLabel(name);
    } catch (LabelSyntaxException e) {
      throw new InvalidRuleException("illegal rule name: " + name + ": " + e.getMessage());
    }
    boolean inWorkspaceFile = pkgBuilder.isRepoRulePackage();
    if (ruleClass.getWorkspaceOnly() && !inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " must be in the WORKSPACE file " + "(used by " + label + ")");
    } else if (!ruleClass.getWorkspaceOnly() && inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " cannot be in the WORKSPACE file " + "(used by " + label + ")");
    }

    BuildLangTypedAttributeValuesMap attributes =
        generatorAttributesForMacros(pkgBuilder, attributeValues, callstack);

    // The raw stack is of the form [<toplevel>@BUILD:1, macro@lib.bzl:1, cc_library@<builtin>].
    // Pop the innermost frame for the rule, since it's obvious.
    callstack = callstack.subList(0, callstack.size() - 1); // pop

    try {
      return ruleClass.createRule(pkgBuilder, label, attributes, eventHandler, callstack);
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
   *     native type of the attribute (i.e. via {@link BuildType#selectableConvert}). There must be
   *     a map entry for each non-optional attribute of this class of rule.
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
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      EventHandler eventHandler,
      ImmutableList<StarlarkThread.CallStackEntry> callstack)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    Rule rule = createRule(pkgBuilder, ruleClass, attributeValues, eventHandler, callstack);
    pkgBuilder.addRule(rule);
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
     * values are "natively typed", i.e. of a type returned by {@link BuildType#selectableConvert}.
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
   * If the rule was created by a macro, sets the appropriate value for the generator_name attribute
   * and returns all attributes.
   *
   * <p>Otherwise, it returns the given attributes without any changes.
   */
  private static BuildLangTypedAttributeValuesMap generatorAttributesForMacros(
      Package.Builder pkgBuilder,
      BuildLangTypedAttributeValuesMap args,
      ImmutableList<CallStackEntry> stack) {
    // The "generator" of a rule is the function (sometimes called "macro") outermost in the call
    // stack. For rules with generators, the stack must contain at least two entries:
    // 0: the outermost function (e.g. a BUILD file),
    // 1: the function called by it (e.g. a "macro" in a .bzl file).
    // optionally followed by other Starlark or built-in functions, and finally the rule
    // instantiation function.
    if (stack.size() < 2 || !stack.get(1).location.file().endsWith(".bzl")) {
      return args; // Not instantiated by a Starlark macro.
    }

    if (args.containsAttributeNamed("generator_name")) {
      // generator_name is explicitly set. Return early to avoid a map key conflict.
      // TODO(b/274802222): Should this be prohibited?
      return args;
    }

    ImmutableMap.Builder<String, Object> builder =
        ImmutableMap.builderWithExpectedSize(args.attributeValues.size() + 1);
    builder.putAll(args.attributeValues);

    String generatorName = pkgBuilder.getGeneratorNameByLocation(stack.get(0).location);
    if (generatorName == null) {
      generatorName = (String) args.getAttributeValue("name");
    }
    builder.put("generator_name", generatorName);

    return new BuildLangTypedAttributeValuesMap(builder.buildOrThrow());
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
          || cl.getRuleClassType() == RuleClassType.TEST) {
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
      BazelStarlarkContext.from(thread).checkLoadingOrWorkspacePhase(ruleClass.getName());
      try {
        PackageContext context = PackageFactory.getContext(thread);
        RuleFactory.createAndAddRule(
            context.pkgBuilder,
            ruleClass,
            new BuildLangTypedAttributeValuesMap(kwargs),
            context.eventHandler,
            thread.getCallStack());
      } catch (RuleFactory.InvalidRuleException | Package.NameConflictException e) {
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
