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

import com.google.devtools.build.lib.events.Location;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Syntax node for a slice expression, e.g. obj[:len(obj):2]. */
public final class SliceExpression extends Expression {

  private final Expression object;
  @Nullable private final Expression start;
  @Nullable private final Expression end;
  @Nullable private final Expression step;

  public SliceExpression(Expression object, Expression start, Expression end, Expression step) {
    this.object = object;
    this.start = start;
    this.end = end;
    this.step = step;
  }

  public Expression getObject() {
    return object;
  }

  public @Nullable Expression getStart() {
    return start;
  }

  public @Nullable Expression getEnd() {
    return end;
  }

  public @Nullable Expression getStep() {
    return step;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    object.prettyPrint(buffer);
    buffer.append('[');
    // The first separator colon is unconditional. The second separator appears only if step is
    // printed.
    if (start != null) {
      start.prettyPrint(buffer);
    }
    buffer.append(':');
    if (end != null) {
      end.prettyPrint(buffer);
    }
    if (step != null) {
      buffer.append(':');
      step.prettyPrint(buffer);
    }
    buffer.append(']');
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    Object objValue = object.eval(env);
    Object startValue = start == null ? Runtime.NONE : start.eval(env);
    Object endValue = end == null ? Runtime.NONE : end.eval(env);
    Object stepValue = step == null ? Runtime.NONE : step.eval(env);
    Location loc = getLocation();

    if (objValue instanceof SkylarkList) {
      return ((SkylarkList<?>) objValue).getSlice(
          startValue, endValue, stepValue, loc, env.mutability());
    } else if (objValue instanceof String) {
      String string = (String) objValue;
      List<Integer> indices = EvalUtils.getSliceIndices(startValue, endValue, stepValue,
          string.length(), loc);
      char[] result = new char[indices.size()];
      char[] original = ((String) objValue).toCharArray();
      int resultIndex = 0;
      for (int originalIndex : indices) {
        result[resultIndex] = original[originalIndex];
        ++resultIndex;
      }
      return new String(result);
    }

    throw new EvalException(
        loc,
        String.format(
            "type '%s' has no operator [:](%s, %s, %s)",
            EvalUtils.getDataTypeName(objValue),
            EvalUtils.getDataTypeName(startValue),
            EvalUtils.getDataTypeName(endValue),
            EvalUtils.getDataTypeName(stepValue)));
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.SLICE;
  }
}
