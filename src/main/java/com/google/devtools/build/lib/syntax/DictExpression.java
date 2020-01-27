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

import com.google.common.collect.ImmutableList;
import java.util.List;

/** Syntax node for dict expressions. */
public final class DictExpression extends Expression {

  /** A key/value pair in a dict expression or comprehension. */
  public static final class Entry extends Node {

    private final Expression key;
    private final Expression value;

    Entry(Expression key, Expression value) {
      this.key = key;
      this.value = value;
    }

    public Expression getKey() {
      return key;
    }

    public Expression getValue() {
      return value;
    }

    @Override
    public void accept(NodeVisitor visitor) {
      visitor.visit(this);
    }
  }

  private final ImmutableList<Entry> entries;

  DictExpression(List<Entry> entries) {
    this.entries = ImmutableList.copyOf(entries);
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.DICT_EXPR;
  }

  public ImmutableList<Entry> getEntries() {
    return entries;
  }
}
