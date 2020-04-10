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
    private final int colonOffset;
    private final Expression value;

    Entry(FileLocations locs, Expression key, int colonOffset, Expression value) {
      super(locs);
      this.key = key;
      this.colonOffset = colonOffset;
      this.value = value;
    }

    public Expression getKey() {
      return key;
    }

    public Expression getValue() {
      return value;
    }

    @Override
    public int getStartOffset() {
      return key.getStartOffset();
    }

    @Override
    public int getEndOffset() {
      return value.getEndOffset();
    }

    public Location getColonLocation() {
      return locs.getLocation(colonOffset);
    }

    @Override
    public void accept(NodeVisitor visitor) {
      visitor.visit(this);
    }
  }

  private final int lbraceOffset;
  private final ImmutableList<Entry> entries;
  private final int rbraceOffset;

  DictExpression(FileLocations locs, int lbraceOffset, List<Entry> entries, int rbraceOffset) {
    super(locs);
    this.lbraceOffset = lbraceOffset;
    this.entries = ImmutableList.copyOf(entries);
    this.rbraceOffset = rbraceOffset;
  }

  @Override
  public int getStartOffset() {
    return lbraceOffset;
  }

  @Override
  public int getEndOffset() {
    return rbraceOffset + 1;
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
