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
import java.util.List;

/** Syntax node for list and tuple expressions. */
public final class ListExpression extends Expression {

  // TODO(adonovan): split class into {List,Tuple}Expression, as a tuple may have no parens.
  // Materialize all source-level expressions as a separate ParenExpression so that we can roundtrip
  // faithfully.

  private final boolean isTuple;
  private final int lbracketOffset; // -1 => unparenthesized non-empty tuple
  private final List<Expression> elements;
  private final int rbracketOffset; // -1 => unparenthesized non-empty tuple

  ListExpression(
      FileLocations locs,
      boolean isTuple,
      int lbracketOffset,
      List<Expression> elements,
      int rbracketOffset) {
    super(locs);
    // An unparenthesized tuple must be non-empty.
    Preconditions.checkArgument(
        !elements.isEmpty() || (lbracketOffset >= 0 && rbracketOffset >= 0));
    this.lbracketOffset = lbracketOffset;
    this.isTuple = isTuple;
    this.elements = elements;
    this.rbracketOffset = rbracketOffset;
  }

  public List<Expression> getElements() {
    return elements;
  }

  /** Reports whether this is a tuple expression. */
  public boolean isTuple() {
    return isTuple;
  }

  @Override
  public int getStartOffset() {
    return lbracketOffset < 0 ? elements.get(0).getStartOffset() : lbracketOffset;
  }

  @Override
  public int getEndOffset() {
    // Unlike Python, trailing commas are not allowed in unparenthesized tuples.
    return rbracketOffset < 0
        ? elements.get(elements.size() - 1).getEndOffset()
        : rbracketOffset + 1;
  }

  @Override
  public String toString() {
    // Print [a, b, c, ...] up to a maximum of 4 elements or 32 chars.
    StringBuilder buf = new StringBuilder();
    buf.append(isTuple() ? '(' : '[');
    appendNodes(buf, elements);
    if (isTuple() && elements.size() == 1) {
      buf.append(',');
    }
    buf.append(isTuple() ? ')' : ']');
    return buf.toString();
  }

  // Appends elements to buf, comma-separated, abbreviating if they are numerous or long.
  // (Also used by CallExpression.)
  static void appendNodes(StringBuilder buf, List<? extends Node> elements) {
    int n = elements.size();
    for (int i = 0; i < n; i++) {
      if (i > 0) {
        buf.append(", ");
      }
      int mark = buf.length();
      buf.append(elements.get(i));
      // Abbreviate, dropping this element, if we exceed 32 chars,
      // or 4 elements (with more elements following).
      if (buf.length() >= 32 || (i == 4 && i + 1 < n)) {
        buf.setLength(mark);
        buf.append(String.format("+%d more", n - i));
        break;
      }
    }
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Expression.Kind kind() {
    return Expression.Kind.LIST_EXPR;
  }
}
