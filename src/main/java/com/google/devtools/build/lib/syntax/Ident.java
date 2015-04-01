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

package com.google.devtools.build.lib.syntax;

// TODO(bazel-team): for extra performance:
// (1) intern the strings, so we can use == to compare, and have .equals use the assumption.
// Then have Argument and Parameter use Ident again instead of String as keys.
// (2) Use Ident, not String, as keys in the Environment, which will be cleaner.
// (3) For performance, avoid doing HashMap lookups at runtime, and compile local variable access
// into array reference with a constant index. Variable lookups are currently a speed bottleneck,
// as previously measured in an experiment.
/**
 *  Syntax node for an identifier.
 */
public final class Ident extends Expression {

  private final String name;

  public Ident(String name) {
    this.name = name;
  }

  /**
   *  Returns the name of the Ident.
   */
  public String getName() {
    return name;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  Object eval(Environment env) throws EvalException {
    try {
      return env.lookup(name);
    } catch (Environment.NoSuchVariableException e) {
      if (name.equals("$error$")) {
        throw new EvalException(getLocation(), "contains syntax error(s)", true);
      } else {
        throw new EvalException(getLocation(), "name '" + name + "' is not defined");
      }
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  SkylarkType validate(ValidationEnvironment env) throws EvalException {
    if (env.hasSymbolInEnvironment(name)) {
      return env.getVartype(name);
    } else if (name.equals("$error$")) {
      throw new EvalException(getLocation(), "contains syntax error(s)", true);
    } else {
      throw new EvalException(getLocation(), "name '" + name + "' is not defined");
    }
  }
}
