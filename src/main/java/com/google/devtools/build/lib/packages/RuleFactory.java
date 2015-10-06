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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.Pair;

import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Given a rule class and a set of attributes, returns a Rule instance. Also
 * performs a number of checks and associates the rule and the owning package
 * with each other.
 *
 * <p>Note: the code that actually populates the RuleClass map has been moved
 * to {@link RuleClassProvider}.
 */
public class RuleFactory {

  /**
   * Maps rule class name to the metaclass instance for that rule.
   */
  private final ImmutableMap<String, RuleClass> ruleClassMap;

  /**
   * Constructs a RuleFactory instance.
   */
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
   * to do so if, for example, the rule has errors).</p>
   */
  static Rule createRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      Map<String, Object> attributeValues,
      EventHandler eventHandler,
      FuncallExpression ast,
      Location location,
      @Nullable Environment env)
      throws InvalidRuleException, InterruptedException {
    Preconditions.checkNotNull(ruleClass);
    String ruleClassName = ruleClass.getName();
    Object nameObject = attributeValues.get("name");
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
    boolean inWorkspaceFile =
        location.getPath() != null && location.getPath().getBaseName().contains("WORKSPACE");
    if (ruleClass.getWorkspaceOnly() && !inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " must be in the WORKSPACE file " + "(used by " + label + ")");
    } else if (!ruleClass.getWorkspaceOnly() && inWorkspaceFile) {
      throw new RuleFactory.InvalidRuleException(
          ruleClass + " cannot be in the WORKSPACE file " + "(used by " + label + ")");
    }

    AttributesAndLocation generator =
        generatorAttributesForMacros(attributeValues, env, location, label);
    try {
      return ruleClass.createRuleWithLabel(
          pkgBuilder, label, generator.attributes, eventHandler, ast, generator.location);
    } catch (LabelSyntaxException e) {
      throw new RuleFactory.InvalidRuleException(ruleClass + " " + e.getMessage());
    }
  }

  /**
   * Creates a rule instance, adds it to the package and returns it.
   *
   * @param pkgBuilder the under-construction package to which the rule belongs
   * @param ruleClass the class of the rule; this must not be null
   * @param attributeValues a map of attribute names to attribute values. Each
   *        attribute must be defined for this class of rule, and have a value
   *        of the appropriate type. There must be a map entry for each
   *        non-optional attribute of this class of rule.
   * @param eventHandler a eventHandler on which errors and warnings are reported during
   *        rule creation
   * @param ast the abstract syntax tree of the rule expression (optional)
   * @param location the location at which this rule was declared
   * @throws InvalidRuleException if the rule could not be constructed for any
   *         reason (e.g. no <code>name</code> attribute is defined)
   * @throws InvalidRuleException, NameConflictException
   */
  static Rule createAndAddRule(
      Package.Builder pkgBuilder,
      RuleClass ruleClass,
      Map<String, Object> attributeValues,
      EventHandler eventHandler,
      FuncallExpression ast,
      Location location,
      Environment env)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    Rule rule = createRule(
        pkgBuilder, ruleClass, attributeValues, eventHandler, ast, location, env);
    pkgBuilder.addRule(rule);
    return rule;
  }

  public static Rule createAndAddRule(
      PackageContext context,
      RuleClass ruleClass,
      Map<String, Object> attributeValues,
      FuncallExpression ast,
      Environment env)
      throws InvalidRuleException, NameConflictException, InterruptedException {
    return createAndAddRule(
        context.pkgBuilder,
        ruleClass,
        attributeValues,
        context.eventHandler,
        ast,
        ast.getLocation(),
        env);
  }

  /**
   * InvalidRuleException is thrown by createRule() if the Rule could not be
   * constructed. It contains an error message.
   */
  public static class InvalidRuleException extends Exception {
    private InvalidRuleException(String message) {
      super(message);
    }
  }

  /** Pair of attributes and location */
  private static final class AttributesAndLocation {
    final Map<String, Object> attributes;
    final Location location;

    AttributesAndLocation(Map<String, Object> attributes, Location location) {
      this.attributes = attributes;
      this.location = location;
    }
  }

  /**
   * If the rule was created by a macro, this method sets the appropriate values for the
   * attributes generator_{name, function, location} and returns all attributes.
   *
   * <p>Otherwise, it returns the given attributes without any changes.
   */
  private static AttributesAndLocation generatorAttributesForMacros(
      Map<String, Object> args, @Nullable Environment env, Location location, Label label) {
    // Returns the original arguments if a) there is only the rule itself on the stack
    // trace (=> no macro) or b) the attributes have already been set by Python pre-processing.
    if (env == null) {
      return new AttributesAndLocation(args, location);
    }
    boolean hasName = args.containsKey("generator_name");
    boolean hasFunc = args.containsKey("generator_function");
    // TODO(bazel-team): resolve cases in our code where hasName && !hasFunc, or hasFunc && !hasName
    if (hasName || hasFunc) {
      return new AttributesAndLocation(args, location);
    }
    Pair<FuncallExpression, BaseFunction> topCall = env.getTopCall();
    if (topCall == null || !(topCall.second instanceof UserDefinedFunction)) {
      return new AttributesAndLocation(args, location);
    }

    FuncallExpression generator = topCall.first;
    BaseFunction function = topCall.second;
    String name = generator.getNameArg();
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    builder.putAll(args);
    builder.put("generator_name", (name == null) ? args.get("name") : name);
    builder.put("generator_function", function.getName());

    if (generator.getLocation() != null) {
      location = generator.getLocation();
      builder.put("generator_location", getRelativeLocation(location, label));
    }

    try {
      return new AttributesAndLocation(builder.build(), location);
    } catch (IllegalArgumentException ex) {
      // We just fall back to the default case and swallow any messages.
      return new AttributesAndLocation(args, location);
    }
  }

  /**
   * Uses the given label to retrieve the workspace-relative path of the given location.
   */
  private static String getRelativeLocation(Location location, Label label) {
    String absolutePath = Location.printPathAndLine(location);
    // Instead of using this substring-based approach, we would prefer to construct the path from
    // the label itself, e.g.
    // buildFileLabel.getPackageFragment().getRelative(buildFileLabel.getName()).getPathString().
    // However, this seems to conflict with python pre-processing since we had seen cases where the
    // label of the BUILD file is something like //package:BUILD while the location is actually
    // /path/to/workspace/package/with/subpackage/BUILD. Consequently, we would lose the
    // "with/subpackage" part.
    int pos = absolutePath.indexOf(label.getPackageName());
    Preconditions.checkArgument(pos > -1, "Cannot retrieve relative path for %s", absolutePath);
    return absolutePath.substring(pos);
  }
}
