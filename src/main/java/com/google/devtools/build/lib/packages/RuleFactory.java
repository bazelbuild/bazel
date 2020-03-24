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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute.SkylarkComputedDefaultTemplate.CannotPrecomputeDefaultsException;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Given a {@link RuleClass} and a set of attribute values, returns a {@link Rule} instance. Also
 * performs a number of checks and associates the {@link Rule} and the owning {@link Package}
 * with each other.
 *
 * <p>This class is immutable, once created the set of managed {@link RuleClass}es will not change.
 *
 * <p>Note: the code that actually populates the RuleClass map has been moved to {@link
 * RuleClassProvider}.
 */
public class RuleFactory {

  /**
   * Maps rule class name to the metaclass instance for that rule.
   */
  private final ImmutableMap<String, RuleClass> ruleClassMap;

  /** Constructs a RuleFactory instance. */
  public RuleFactory(RuleClassProvider provider) {
    this.ruleClassMap = ImmutableMap.copyOf(provider.getRuleClassMap());
  }

  /**
   * Returns the (immutable, unordered) set of names of all the known rule classes.
   */
  public Set<String> getRuleClassNames() {
    return ruleClassMap.keySet();
  }

  /**
   * Returns the RuleClass for the specified rule class name.
   */
  public RuleClass getRuleClass(String ruleClassName) {
    return ruleClassMap.get(ruleClassName);
  }

  /**
   * Creates and returns a rule instance.
   *
   * <p>It is the caller's responsibility to add the rule to the package (the caller may choose not
   * to do so if, for example, the rule has errors).
   */
  static Rule createRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      EventHandler eventHandler,
      StarlarkSemantics semantics,
      ImmutableList<StarlarkThread.CallStackEntry> callstack,
      AttributeContainer attributeContainer)
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
    boolean inWorkspaceFile = pkgBuilder.isWorkspace();
    if (ruleClass.getWorkspaceOnly() && !inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " must be in the WORKSPACE file " + "(used by " + label + ")");
    } else if (!ruleClass.getWorkspaceOnly() && inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " cannot be in the WORKSPACE file " + "(used by " + label + ")");
    }

    AttributesAndLocation generator =
        generatorAttributesForMacros(pkgBuilder, attributeValues, callstack, label);

    // The raw stack is of the form [<toplevel>@BUILD:1, macro@lib.bzl:1, cc_library@<builtin>].
    // If we're recording it (--record_rule_instantiation_callstack),
    // pop the innermost frame for the rule, since it's obvious.
    callstack =
        semantics.recordRuleInstantiationCallstack()
            ? callstack.subList(0, callstack.size() - 1) // pop
            : ImmutableList.of(); // save space

    try {
      // Examines --incompatible_disable_third_party_license_checking to see if we should check
      // third party targets for license existence.
      //
      // This flag is overridable by RuleClass.ThirdPartyLicenseEnforcementPolicy (which is checked
      // in RuleClass). This lets Bazel and Blaze migrate away from license logic on independent
      // timelines. See --incompatible_disable_third_party_license_checking comments for details.
      boolean checkThirdPartyLicenses = !semantics.incompatibleDisableThirdPartyLicenseChecking();
      return ruleClass.createRule(
          pkgBuilder,
          label,
          generator.attributes,
          eventHandler,
          generator.location, // see b/23974287 for rationale
          callstack,
          attributeContainer,
          checkThirdPartyLicenses);
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
   * @param semantics the Starlark semantics
   * @param callstack the stack of active calls in the Starlark thread
   * @param attributeContainer the {@link AttributeContainer} the rule will contain
   * @throws InvalidRuleException if the rule could not be constructed for any reason (e.g. no
   *     {@code name} attribute is defined)
   * @throws NameConflictException if the rule's name or output files conflict with others in this
   *     package
   * @throws InterruptedException if interrupted
   */
  static Rule createAndAddRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      EventHandler eventHandler,
      StarlarkSemantics semantics,
      ImmutableList<StarlarkThread.CallStackEntry> callstack,
      AttributeContainer attributeContainer)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    Rule rule =
        createRule(
            pkgBuilder,
            ruleClass,
            attributeValues,
            eventHandler,
            semantics,
            callstack,
            attributeContainer);
    pkgBuilder.addRule(rule);
    return rule;
  }

  /**
   * Creates a {@link Rule} instance, adds it to the {@link Package.Builder} and returns it.
   *
   * @param context the package-building context in which this rule was declared
   * @param ruleClass the {@link RuleClass} of the rule
   * @param attributeValues a {@link BuildLangTypedAttributeValuesMap} mapping attribute names to
   *     attribute values of build-language type. Each attribute must be defined for this class of
   *     rule, and have a build-language-typed value which can be converted to the appropriate
   *     native type of the attribute (i.e. via {@link BuildType#selectableConvert}). There must be
   *     a map entry for each non-optional attribute of this class of rule.
   * @param loc the location of the rule expression
   * @param thread the lexical environment of the function call which declared this rule (optional)
   * @param attributeContainer the {@link AttributeContainer} the rule will contain
   * @throws InvalidRuleException if the rule could not be constructed for any reason (e.g. no
   *     {@code name} attribute is defined)
   * @throws NameConflictException if the rule's name or output files conflict with others in this
   *     package
   * @throws InterruptedException if interrupted
   */
  public static Rule createAndAddRule(
      PackageContext context,
      RuleClass ruleClass,
      BuildLangTypedAttributeValuesMap attributeValues,
      StarlarkSemantics semantics,
      ImmutableList<StarlarkThread.CallStackEntry> callstack,
      AttributeContainer attributeContainer)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    return createAndAddRule(
        context.pkgBuilder,
        ruleClass,
        attributeValues,
        context.eventHandler,
        semantics,
        callstack,
        attributeContainer);
  }

  /**
   * InvalidRuleException is thrown by {@link Rule} creation methods if the {@link Rule} could
   * not be constructed. It contains an error message.
   */
  public static class InvalidRuleException extends Exception {
    private InvalidRuleException(String message) {
      super(message);
    }
  }

  /** A pair of attributes and location. */
  private static final class AttributesAndLocation {
    final BuildLangTypedAttributeValuesMap attributes;
    final Location location;

    AttributesAndLocation(BuildLangTypedAttributeValuesMap attributes, Location location) {
      this.attributes = attributes;
      this.location = location;
    }
  }

  /**
   * A wrapper around an map of named attribute values that specifies whether the map's values
   * are of "build-language" or of "native" types.
   */
  public interface AttributeValues<T> {
    /**
     * Returns {@code true} if all the map's values are "build-language typed", i.e., resulting
     * from the evaluation of an expression in the build language. Returns {@code false} if all
     * the map's values are "natively typed", i.e. of a type returned by {@link
     * BuildType#selectableConvert}.
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
    public Iterable<Map.Entry<String, Object>> getAttributeAccessors() {
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
   * If the rule was created by a macro, this method sets the appropriate values for the attributes
   * generator_{name, function, location} and returns all attributes.
   *
   * <p>Otherwise, it returns the given attributes without any changes.
   */
  private static AttributesAndLocation generatorAttributesForMacros(
      Package.Builder pkgBuilder,
      BuildLangTypedAttributeValuesMap args,
      ImmutableList<StarlarkThread.CallStackEntry> stack,
      Label label) {
    // For a callstack [BUILD <toplevel>, .bzl <function>, <rule>],
    // location is that of the caller of 'rule' (the .bzl function).
    Location location = stack.size() < 2 ? Location.BUILTIN : stack.get(stack.size() - 2).location;

    boolean hasName = args.containsAttributeNamed("generator_name");
    boolean hasFunc = args.containsAttributeNamed("generator_function");
    // TODO(bazel-team): resolve cases in our code where hasName && !hasFunc, or hasFunc && !hasName
    if (hasName || hasFunc) {
      return new AttributesAndLocation(args, location);
    }

    // The "generator" of a rule is the function (sometimes called "macro")
    // outermost in the call stack.
    // The stack must contain at least two entries:
    // 0: the outermost function (e.g. a BUILD file),
    // 1: the function called by it (e.g. a "macro" in a .bzl file).
    if (stack.size() < 2 || !stack.get(1).location.file().endsWith(".bzl")) {
      return new AttributesAndLocation(args, location); // macro is not a Starlark function
    }
    Location generatorLocation = stack.get(0).location; // location of call to generator
    String generatorFunction = stack.get(1).name;
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    for (Map.Entry<String, Object> attributeAccessor : args.getAttributeAccessors()) {
      String attributeName = args.getName(attributeAccessor);
      builder.put(attributeName, args.getValue(attributeAccessor));
    }
    String generatorName = pkgBuilder.getGeneratorNameByLocation().get(generatorLocation);
    if (generatorName == null) {
      generatorName = (String) args.getAttributeValue("name");
    }
    builder.put("generator_name", generatorName);
    builder.put("generator_function", generatorFunction);
    String relativePath = maybeGetRelativeLocation(generatorLocation, label);
    if (relativePath != null) {
      builder.put("generator_location", relativePath);
    }

    try {
      args = new BuildLangTypedAttributeValuesMap(builder.build());
    } catch (IllegalArgumentException unused) {
      // We just fall back to the default case and swallow any messages.
    }

    // TODO(adonovan): is it appropriate to use generatorLocation as the rule's main location?
    // Or would 'location' (the immediate call) be more informative? When there are errors, the
    // location of the toplevel call of the generator may be quite unrelated to the error message.
    return new AttributesAndLocation(args, generatorLocation);
  }

  /**
   * Uses the given label to retrieve the workspace-relative path of the given location (including
   * the line number).
   *
   * <p>For example, the location /usr/local/workspace/my/cool/package/BUILD:3:1 and the label
   * //my/cool/package:BUILD would lead to "my/cool/package:BUILD:3".
   *
   * @return The workspace-relative path of the given location, or null if it could not be computed.
   */
  // TODO(b/151151653): make Starlark file Locations relative from the outset.
  @Nullable
  private static String maybeGetRelativeLocation(@Nullable Location location, Label label) {
    if (location == null) {
      return null;
    }
    // Determining the workspace root only works reliably if both location and label point to files
    // in the same package.
    // It would be preferable to construct the path from the label itself, but this doesn't work for
    // rules created from function calls in a subincluded file, even if both files share a path
    // prefix (for example, when //a/package:BUILD subincludes //a/package/with/a/subpackage:BUILD).
    // We can revert to that approach once subincludes aren't supported anymore.
    //
    // TODO(b/151165647): this logic has always been wrong:
    // it spuriously matches occurrences of the package name earlier in the path.
    String absolutePath = location.toString();
    int pos = absolutePath.indexOf(label.getPackageName());
    return (pos < 0) ? null : absolutePath.substring(pos);
  }
}
