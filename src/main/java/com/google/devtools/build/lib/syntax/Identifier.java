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

import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.Variable.SkylarkVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.SpellChecker;
import java.util.Set;
import javax.annotation.Nullable;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

// TODO(bazel-team): for extra performance:
// (1) intern the strings, so we can use == to compare, and have .equals use the assumption.
// Then have Argument and Parameter use Identifier again instead of String as keys.
// (2) Use Identifier, not String, as keys in the Environment, which will be cleaner.
// (3) For performance, avoid doing HashMap lookups at runtime, and compile local variable access
// into array reference with a constant index. Variable lookups are currently a speed bottleneck,
// as previously measured in an experiment.
/**
 *  Syntax node for an identifier.
 */
public final class Identifier extends Expression {

  private final String name;

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
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Identifier) {
      Identifier that = (Identifier) object;
      return this.name.equals(that.name);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }

  @Override
  Object doEval(Environment env) throws EvalException {
    Object value = env.lookup(name);
    if (value == null) {
      throw createInvalidIdentifierException(env.getVariableNames());
    }
    return value;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    if (!env.hasSymbolInEnvironment(name)) {
      throw createInvalidIdentifierException(env.getAllSymbols());
    }
  }

  private EvalException createInvalidIdentifierException(Set<String> symbols) {
    if (name.equals("$error$")) {
      return new EvalException(getLocation(), "contains syntax error(s)", true);
    }

    String suggestion = SpellChecker.suggest(name, symbols);
    if (suggestion == null) {
      suggestion = "";
    } else {
      suggestion = " (did you mean '" + suggestion + "'?)";
    }

    return new EvalException(getLocation(), "name '" + name + "' is not defined" + suggestion);
  }

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) {
    SkylarkVariable variable = scope.getVariable(this);
    return variable.load(scope, debugInfo.add(this));
  }
}
