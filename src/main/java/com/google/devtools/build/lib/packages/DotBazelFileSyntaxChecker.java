// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import net.starlark.java.syntax.Argument;
import net.starlark.java.syntax.CallExpression;
import net.starlark.java.syntax.DefStatement;
import net.starlark.java.syntax.ForStatement;
import net.starlark.java.syntax.IfStatement;
import net.starlark.java.syntax.LambdaExpression;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.NodeVisitor;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;

/**
 * A {@link NodeVisitor} that can be used to check that a Starlark AST conforms to the restricted
 * syntax that BUILD, WORKSPACE, REPO.bazel, and MODULE.bazel files use. This restricted syntax
 * disallows:
 *
 * <ul>
 *   <li>control-flow statements ({@code for} and {@code if}, but not comprehensions and {@code if}
 *       expressions),
 *   <li>function definitions ({@code def} and {@code lambda}),
 *   <li>variadic arguments ({@code *args} and {@code **kwargs}) in function call sites, and
 *   <li>optionally, {@code load()} statements.
 * </ul>
 */
@ThreadHostile
public class DotBazelFileSyntaxChecker extends NodeVisitor {
  private final String where;
  private final boolean canLoadBzl;
  private ImmutableList.Builder<SyntaxError> errors = ImmutableList.builder();

  /**
   * @param where describes the type of file being checked.
   * @param canLoadBzl whether the file type being check supports load statements. This is used to
   *     generate more informative error messages.
   */
  public DotBazelFileSyntaxChecker(String where, boolean canLoadBzl) {
    this.where = where;
    this.canLoadBzl = canLoadBzl;
  }

  public final void check(StarlarkFile file) throws SyntaxError.Exception {
    this.errors = ImmutableList.builder();
    visit(file);
    ImmutableList<SyntaxError> errors = this.errors.build();
    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(errors);
    }
  }

  protected void error(Location loc, String message) {
    errors.add(new SyntaxError(loc, message));
  }

  // Reject f(*args) and f(**kwargs) calls.
  private void rejectStarArgs(CallExpression call) {
    for (Argument arg : call.getArguments()) {
      if (arg instanceof Argument.StarStar) {
        error(
            arg.getStartLocation(),
            "**kwargs arguments are not allowed in "
                + where
                + ". Pass the arguments in explicitly.");
      } else if (arg instanceof Argument.Star) {
        error(
            arg.getStartLocation(),
            "*args arguments are not allowed in " + where + ". Pass the arguments in explicitly.");
      }
    }
  }

  @Override
  public void visit(LoadStatement node) {
    if (!canLoadBzl) {
      error(node.getStartLocation(), "`load` statements may not be used in " + where);
    }
  }

  // We prune the traversal if we encounter disallowed keywords, as we have already reported the
  // root error and there's no point reporting more.

  @Override
  public void visit(DefStatement node) {
    error(
        node.getStartLocation(),
        "functions may not be defined in "
            + where
            + (canLoadBzl ? ". You may move the function to a .bzl file and load it." : "."));
  }

  @Override
  public void visit(LambdaExpression node) {
    error(
        node.getStartLocation(),
        "functions may not be defined in "
            + where
            + (canLoadBzl ? ". You may move the function to a .bzl file and load it." : "."));
  }

  @Override
  public void visit(ForStatement node) {
    error(
        node.getStartLocation(),
        "`for` statements are not allowed in "
            + where
            + ". You may inline the loop"
            + (canLoadBzl ? ", move it to a function definition (in a .bzl file)," : "")
            + " or as a last resort use a list comprehension.");
  }

  @Override
  public void visit(IfStatement node) {
    error(
        node.getStartLocation(),
        "`if` statements are not allowed in "
            + where
            + ". You may"
            + (canLoadBzl
                ? " move conditional logic to a function definition (in a .bzl file), or"
                : "")
            + " use an `if` expression for simple cases.");
  }

  @Override
  public void visit(CallExpression node) {
    rejectStarArgs(node);
    // Continue traversal so as not to miss nested calls
    // like cc_binary(..., f(**kwargs), ...).
    super.visit(node);
  }
}
