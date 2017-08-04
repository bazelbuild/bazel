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

/** Syntax node for a slice expression, e.g. obj[:len(obj):2]. */
public final class SliceExpression extends Expression {

  private final Expression object;
  private final Expression start;
  private final Expression end;
  private final Expression step;

  public SliceExpression(Expression object, Expression start, Expression end, Expression step) {
    this.object = object;
    this.start = start;
    this.end = end;
    this.step = step;
  }

  public Expression getObject() {
    return object;
  }

  public Expression getStart() {
    return start;
  }

  public Expression getEnd() {
    return end;
  }

  public Expression getStep() {
    return step;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    boolean startIsDefault =
        (start instanceof Identifier) && ((Identifier) start).getName().equals("None");
    boolean endIsDefault =
        (end instanceof Identifier) && ((Identifier) end).getName().equals("None");
    boolean stepIsDefault =
        (step instanceof IntegerLiteral) && ((IntegerLiteral) step).getValue().equals(1);

    object.prettyPrint(buffer);
    buffer.append('[');
    // Start and end are omitted if they are the literal identifier None, which is the default value
    // inserted by the parser if no bound is given. Likewise, step is omitted if it is the literal
    // integer 1.
    //
    // The first separator colon is unconditional. The second separator appears only if step is
    // printed.
    if (!startIsDefault) {
      start.prettyPrint(buffer);
    }
    buffer.append(':');
    if (!endIsDefault) {
      end.prettyPrint(buffer);
    }
    if (!stepIsDefault) {
      buffer.append(':');
      step.prettyPrint(buffer);
    }
    buffer.append(']');
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    Object objValue = object.eval(env);
    Object startValue = start.eval(env);
    Object endValue = end.eval(env);
    Object stepValue = step.eval(env);
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
  void validate(ValidationEnvironment env) throws EvalException {
    object.validate(env);
  }
}
