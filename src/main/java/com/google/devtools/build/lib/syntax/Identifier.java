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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;

// TODO(bazel-team): For performance, avoid doing HashMap lookups at runtime, and compile local
// variable access into array reference with a constant index. Variable lookups are currently a
// speed bottleneck, as previously measured in an experiment.
/**
 * Syntax node for an identifier.
 */
public final class Identifier extends Expression {

  private final String name;
  // The scope of the variable. The value is set when the AST has been analysed by
  // ValidationEnvironment.
  @Nullable private ValidationEnvironment.Scope scope;

  Identifier(String name) {
    this.name = name;
  }

  /**
   *  Returns the name of the Identifier.
   */
  public String getName() {
    return name;
  }

  public boolean isPrivate() {
    return name.startsWith("_");
  }

  ValidationEnvironment.Scope getScope() {
    return scope;
  }

  void setScope(ValidationEnvironment.Scope scope) {
    Preconditions.checkState(this.scope == null);
    this.scope = scope;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.IDENTIFIER;
  }

  /** @return The {@link Identifier} of the provided name. */
  static Identifier of(String name) {
    return new Identifier(name);
  }

  /** Returns true if the string is a syntactically valid identifier. */
  public static boolean isValid(String name) {
    // TODO(laurentlb): Handle Unicode characters.
    if (name.isEmpty()) {
      return false;
    }
    for (int i = 0; i < name.length(); i++) {
      char c = name.charAt(i);
      if ((c >= 'a' && c <= 'z')
          || (c >= 'A' && c <= 'Z')
          || (c >= '0' && c <= '9')
          || (c == '_')) {
        continue;
      }
      return false;
    }
    if (name.charAt(0) >= '0' && name.charAt(0) <= '9') {
      return false;
    }

    return true;
  }

  /**
   * Returns all names bound by an LHS expression.
   *
   * <p>Examples:
   *
   * <ul>
   *   <li><{@code x = ...} binds x.
   *   <li><{@code x, [y,z] = ..} binds x, y, z.
   *   <li><{@code x[5] = ..} does not bind any names.
   * </ul>
   */
  static ImmutableSet<Identifier> boundIdentifiers(Expression expr) {
    if (expr instanceof Identifier) {
      // Common case/fast path - skip the builder.
      return ImmutableSet.of((Identifier) expr);
    } else {
      ImmutableSet.Builder<Identifier> result = ImmutableSet.builder();
      collectBoundIdentifiers(expr, result);
      return result.build();
    }
  }

  private static void collectBoundIdentifiers(
      Expression lhs, ImmutableSet.Builder<Identifier> result) {
    if (lhs instanceof Identifier) {
      result.add((Identifier) lhs);
      return;
    }
    if (lhs instanceof ListExpression) {
      ListExpression variables = (ListExpression) lhs;
      for (Expression expression : variables.getElements()) {
        collectBoundIdentifiers(expression, result);
      }
    }
  }
}
