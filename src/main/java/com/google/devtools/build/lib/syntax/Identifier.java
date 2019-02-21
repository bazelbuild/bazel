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
import com.google.devtools.build.lib.util.SpellChecker;
import java.io.IOException;
import java.util.Set;
import javax.annotation.Nullable;

// TODO(bazel-team): For performance, avoid doing HashMap lookups at runtime, and compile local
// variable access into array reference with a constant index. Variable lookups are currently a
// speed bottleneck, as previously measured in an experiment.
/**
 * Syntax node for an identifier.
 *
 * <p>Unlike most {@link ASTNode} subclasses, this one supports {@link Object#equals} and {@link
 * Object#hashCode} (but note that these methods ignore location information). They are needed
 * because {@code Identifier}s are stored in maps when constructing {@link LoadStatement}.
 */
public final class Identifier extends Expression {

  private final String name;
  // The scope of the variable. The value is set when the AST has been analysed by
  // ValidationEnvironment.
  @Nullable private ValidationEnvironment.Scope scope;

  public Identifier(String name) {
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

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(name);
  }

  @Override
  public boolean equals(@Nullable Object object) {
    // TODO(laurentlb): Remove this. AST nodes should probably not be comparable.
    if (object instanceof Identifier) {
      Identifier that = (Identifier) object;
      return this.name.equals(that.name);
    }
    return false;
  }

  @Override
  public int hashCode() {
    // TODO(laurentlb): Remove this.
    return name.hashCode();
  }

  void setScope(ValidationEnvironment.Scope scope) {
    Preconditions.checkState(this.scope == null);
    this.scope = scope;
  }

  @Override
  Object doEval(Environment env) throws EvalException {
    Object result;
    if (scope == null) {
      // Legacy behavior, to be removed.
      result = env.lookup(name);
      if (result == null) {
        throw createInvalidIdentifierException(env.getVariableNames());
      }
      return result;
    }

    switch (scope) {
      case Local:
        result = env.localLookup(name);
        break;
      case Module:
        result = env.moduleLookup(name);
        break;
      case Universe:
        result = env.universeLookup(name);
        break;
      default:
        throw new IllegalStateException(scope.toString());
    }

    if (result == null) {
      // Since Scope was set, we know that the variable is defined in the scope.
      // However, the assignment was not yet executed.
      EvalException e = getSpecialException();
      throw e != null
          ? e
          : new EvalException(
              getLocation(),
              scope.getQualifier() + " variable '" + name + "' is referenced before assignment.");
    }

    return result;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.IDENTIFIER;
  }

  /**
   * Exception to provide a better error message for using PACKAGE_NAME or REPOSITORY_NAME.
   */
  private EvalException getSpecialException() {
    if (name.equals("PACKAGE_NAME")) {
      return new EvalException(
          getLocation(),
          "The value 'PACKAGE_NAME' has been removed in favor of 'package_name()', "
              + "please use the latter ("
              + "https://docs.bazel.build/versions/master/skylark/lib/native.html#package_name). ");
    }
    if (name.equals("REPOSITORY_NAME")) {
      return new EvalException(
          getLocation(),
          "The value 'REPOSITORY_NAME' has been removed in favor of 'repository_name()', "
              + "please use the latter ("
              + "https://docs.bazel.build/versions/master/skylark/lib/native.html#repository_name).");
    }
    return null;
  }

  EvalException createInvalidIdentifierException(Set<String> symbols) {
    if (name.equals("$error$")) {
      return new EvalException(getLocation(), "contains syntax error(s)", true);
    }

    EvalException e = getSpecialException();
    if (e != null) {
      return e;
    }

    String suggestion = SpellChecker.didYouMean(name, symbols);
    return new EvalException(getLocation(), "name '" + name + "' is not defined" + suggestion);
  }

  /** @return The {@link Identifier} of the provided name. */
  public static Identifier of(String name) {
    return new Identifier(name);
  }
}
