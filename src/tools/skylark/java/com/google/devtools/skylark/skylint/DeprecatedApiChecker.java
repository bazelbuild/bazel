// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DotExpression;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.ReturnStatement;
import com.google.devtools.build.lib.syntax.SyntaxTreeVisitor;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Checks for operations that are deprecated */
public class DeprecatedApiChecker extends AstVisitorWithNameResolution {
  private static final String DEPRECATED_API = "deprecated-api";
  private static final String RULE_IMPL_RETURN = "deprecated-rule-impl-return";

  private final List<Issue> issues = new ArrayList<>();

  /** True if we are currently visiting a rule implementation. */
  private boolean visitingRuleImplementation;

  /** Set of functions that are used as rule implementation. */
  private final Set<String> ruleImplSet = Sets.newHashSet();

  private DeprecatedApiChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    DeprecatedApiChecker checker = new DeprecatedApiChecker();
    checker.inferRuleImpl(ast);
    checker.visit(ast);
    return checker.issues;
  }

  /**
   * Convert a dotted expression to string, e.g. rule -> "rule" attr.label -> "attr.label"
   *
   * <p>If input contains anything else than Identifier or DotExpression, return empty string.
   */
  private static String dottedExpressionToString(Expression e) {
    if (e instanceof Identifier) {
      return ((Identifier) e).getName();
    }
    if (e instanceof DotExpression) {
      String result = dottedExpressionToString(((DotExpression) e).getObject());
      if (!result.isEmpty()) {
        return result + "." + ((DotExpression) e).getField().getName();
      }
    }

    return "";
  }

  private void inferRuleImpl(BuildFileAST ast) {
    new SyntaxTreeVisitor() {

      @Override
      public void visit(FuncallExpression node) {
        // Collect all 'x' that match this pattern:
        //   rule(implementation=x, ...)
        Expression fct = node.getFunction();
        if (!(fct instanceof Identifier) || !((Identifier) fct).getName().equals("rule")) {
          return;
        }

        boolean firstArg = true;
        for (Argument.Passed arg : node.getArguments()) {
          if (!"implementation".equals(arg.getName()) && (!firstArg || arg.isKeyword())) {
            firstArg = false;
            continue;
          }
          firstArg = false;
          Expression val = arg.getValue();
          if (val instanceof Identifier) {
            ruleImplSet.add(((Identifier) val).getName());
          }
        }
      }
    }.visit(ast);
  }

  private static final ImmutableMap<String, String> deprecatedMethods =
      ImmutableMap.<String, String>builder()
          .put("ctx.action", "Use ctx.actions.run or ctx.actions.run_shell.")
          .put("ctx.default_provider", "Use DefaultInfo.")
          .put("ctx.empty_action", "Use ctx.actions.do_nothing.")
          .put("ctx.expand_make_variables", "Use ctx.var to access the variables.")
          .put("ctx.file_action", "Use ctx.actions.write.")
          .put("ctx.new_file", "Use ctx.actions.declare_file.")
          .put("ctx.template_action", "Use ctx.actions.expand_template.")
          .put("PACKAGE_NAME", "Use native.package_name().")
          .put("REPOSITORY_NAME", "Use native.repository_name().")
          .put("FileType", "Use a list of strings.")
          .put(
              "ctx.outputs.executable",
              "See https://docs.bazel.build/versions/master/skylark/"
                  + "rules.html#executable-rules-and-test-rules")
          .build();

  private void checkDeprecated(Expression node) {
    String name = dottedExpressionToString(node);
    if (deprecatedMethods.containsKey(name)) {
      issues.add(
          Issue.create(
              DEPRECATED_API,
              name + " is deprecated: " + deprecatedMethods.get(name),
              node.getLocation()));
    }
  }

  @Override
  public void visit(Identifier node) {
    super.visit(node);
    checkDeprecated(node);
  }

  @Override
  public void visit(DotExpression node) {
    super.visit(node);
    checkDeprecated(node);
  }

  @Override
  public void visit(ReturnStatement node) {
    super.visit(node);

    // Check that rule implementation functions don't return a call to `struct`.
    if (!visitingRuleImplementation) {
      return;
    }
    Expression e = node.getReturnExpression();
    if (e == null) {
      return;
    }
    if (!(e instanceof FuncallExpression)) {
      return;
    }
    String fctName = dottedExpressionToString(((FuncallExpression) e).getFunction());
    if (fctName.equals("struct")) {
      issues.add(
          Issue.create(
              RULE_IMPL_RETURN,
              "Avoid using the legacy provider syntax. Instead of returning a `struct` from a rule "
                  + "implementation function, return a list of providers: "
                  + "https://docs.bazel.build/versions/master/skylark/rules.html"
                  + "#migrating-from-legacy-providers",
              node.getLocation()));
    }
  }

  @Override
  public void visit(FunctionDefStatement node) {
    visitingRuleImplementation = ruleImplSet.contains(node.getIdentifier().getName());
    super.visit(node);
  }
}
