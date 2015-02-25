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

/**
 * Syntax node for a function argument. This can be a key/value pair such as
 * appears as a keyword argument to a function call or just an expression that
 * is used as a positional argument. It also can be used for function definitions
 * to identify the name (and optionally the default value) of the argument.
 */
public final class Argument extends ASTNode {

  private final Ident name;

  private final Expression value;

  private final boolean kwargs;

  /**
   * Create a new argument.
   * At call site: name is optional, value is mandatory. kwargs is true for ** arguments.
   * At definition site: name is mandatory, (default) value is optional.
   */
  public Argument(Ident name, Expression value, boolean kwargs) {
    this.name = name;
    this.value = value;
    this.kwargs = kwargs;
  }

  public Argument(Ident name, Expression value) {
    this.name = name;
    this.value = value;
    this.kwargs = false;
  }

  /**
   * Creates an Argument with null as name. It can be used as positional arguments
   * of function calls.
   */
  public Argument(Expression value) {
    this(null, value);
  }

  /**
   * Creates an Argument with null as value. It can be used as a mandatory keyword argument
   * of a function definition.
   */
  public Argument(Ident name) {
    this(name, null);
  }

  /**
   * Returns the name of this keyword argument or null if this argument is
   * positional.
   */
  public Ident getName() {
    return name;
  }

  /**
   * Returns the String value of the Ident of this argument. Shortcut for arg.getName().getName().
   */
  public String getArgName() {
    return name.getName();
  }

  /**
   * Returns the syntax of this argument expression.
   */
  public Expression getValue() {
    return value;
  }

  /**
   * Returns true if this argument is positional.
   */
  public boolean isPositional() {
    return name == null && !kwargs;
  }

  /**
   * Returns true if this argument is a keyword argument.
   */
  public boolean isNamed() {
    return name != null;
  }

  /**
   * Returns true if this argument is a **kwargs argument.
   */
  public boolean isKwargs() {
    return kwargs;
  }

  /**
   * Returns true if this argument has value.
   */
  public boolean hasValue() {
    return value != null;
  }

  @Override
  public String toString() {
    return isNamed() ? name + "=" + value : String.valueOf(value);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }
}
