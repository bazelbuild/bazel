// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
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
      throws InvalidRuleException {
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
    } catch (Label.SyntaxException e) {
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

    try {
      return ruleClass.createRuleWithLabel(
          pkgBuilder,
          label,
          addGeneratorAttributesForMacros(attributeValues, env),
          eventHandler,
          ast,
          location);
    } catch (SyntaxException e) {
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
      throws InvalidRuleException, NameConflictException {
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
      throws InvalidRuleException, NameConflictException {
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

  /**
   * If the rule was created by a macro, this method sets the appropriate values for the
   * attributes generator_{name, function, location} and returns all attributes.
   *
   * <p>Otherwise, it returns the given attributes without any changes.
   */
  private static Map<String, Object> addGeneratorAttributesForMacros(
      Map<String, Object> args, @Nullable Environment env) {
    // Returns the original arguments if a) there is only the rule itself on the stack
    // trace (=> no macro) or b) the attributes have already been set by Python pre-processing.
    if (env == null) {
      return args;
    }
    boolean hasName = args.containsKey("generator_name");
    boolean hasFunc = args.containsKey("generator_function");
    // TODO(bazel-team): resolve cases in our code where hasName && !hasFunc, or hasFunc && !hasName
    if (hasName || hasFunc) {
      return args;
    }
    Pair<FuncallExpression, BaseFunction> topCall = env.getTopCall();
    if (topCall == null || !(topCall.second instanceof UserDefinedFunction)) {
      return args;
    }

    FuncallExpression generator = topCall.first;
    BaseFunction function = topCall.second;
    String name = generator.getNameArg();
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    builder.putAll(args);
    builder.put("generator_name", (name == null) ? args.get("name") : name);
    builder.put("generator_function", function.getName());
    builder.put("generator_location", Location.printPathAndLine(generator.getLocation()));

    try {
      return builder.build();
    } catch (IllegalArgumentException ex) {
      // Just to play it safe.
      return args;
    }
  }
}
