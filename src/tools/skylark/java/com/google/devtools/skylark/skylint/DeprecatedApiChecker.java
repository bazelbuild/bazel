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
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DotExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import java.util.ArrayList;
import java.util.List;

/** Checks for operations that are deprecated */
public class DeprecatedApiChecker extends AstVisitorWithNameResolution {
  private static final String DEPRECATED_API = "deprecated-api";

  private final List<Issue> issues = new ArrayList<>();

  private DeprecatedApiChecker() {}

  public static List<Issue> check(BuildFileAST ast) {
    DeprecatedApiChecker checker = new DeprecatedApiChecker();
    checker.visit(ast);
    return checker.issues;
  }

  private static final ImmutableMap<String, String> deprecatedMethods =
      ImmutableMap.<String, String>builder()
          .put("ctx.action", "Use ctx.actions.run() or ctx.actions.run_shell().")
          .put("ctx.default_provider", "Use DefaultInfo.")
          .put("ctx.empty_action", "Use ctx.actions.do_nothing.")
          .put("ctx.expand_make_variables", "Use ctx.var to access the variables.")
          .put("ctx.file_action", "Use ctx.actions.write.")
          .put("ctx.new_file", "Use ctx.actions.declare_file.")
          .put("ctx.template_action", "Use ctx.actions.expand_template().")
          .build();

  @Override
  public void visit(DotExpression node) {
    super.visit(node);

    if (!(node.getObject() instanceof Identifier)) {
      return;
    }

    String name = ((Identifier) node.getObject()).getName() + "." + node.getField().getName();
    if (deprecatedMethods.containsKey(name)) {
      issues.add(
          Issue.create(
              DEPRECATED_API,
              "This method is deprecated: " + deprecatedMethods.get(name),
              node.getLocation()));
    }
  }
}
