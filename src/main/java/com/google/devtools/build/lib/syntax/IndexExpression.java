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


/**
 * An index expression ({@code obj[field]}). Not to be confused with a slice expression ({@code
 * obj[from:to]}). The object may be either a sequence or an associative mapping (most commonly
 * lists and dictionaries).
 */
public final class IndexExpression extends Expression {

  private final Expression object;

  private final Expression key;

  IndexExpression(Expression object, Expression key) {
    this.object = object;
    this.key = key;
  }

  public Expression getObject() {
    return object;
  }

  public Expression getKey() {
    return key;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.INDEX;
  }
}
