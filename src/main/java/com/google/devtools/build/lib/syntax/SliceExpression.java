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
import java.util.List;

/** Syntax node for an index expression. e.g. obj[field], but not obj[from:to] */
public final class SliceExpression extends Expression {

  private final Expression obj;
  private final Expression start;
  private final Expression end;
  private final Expression step;

  public SliceExpression(Expression obj, Expression start, Expression end, Expression step) {
    this.obj = obj;
    this.start = start;
    this.end = end;
    this.step = step;
  }

  public Expression getObject() {
    return obj;
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
  public String toString() {
    return String.format("%s[%s:%s%s]",
        obj,
        start,
        // Omit `end` if it's a literal `None` (default value)
        ((end instanceof Identifier) && (((Identifier) end).getName().equals("None"))) ? "" : end,
        // Omit `step` if it's an integer literal `1` (default value)
        ((step instanceof IntegerLiteral) && (((IntegerLiteral) step).value.equals(1)))
            ? "" : ":" + step
    );
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    Object objValue = obj.eval(env);
    Object startValue = start.eval(env);
    Object endValue = end.eval(env);
    Object stepValue = step.eval(env);
    Location loc = getLocation();

    if (objValue instanceof SkylarkList) {
      SkylarkList<Object> list = (SkylarkList<Object>) objValue;
      Object slice = list.getSlice(startValue, endValue, stepValue, loc);
      return SkylarkType.convertToSkylark(slice, env);
    } else if (objValue instanceof String) {
      String string = (String) objValue;
      List<Integer> indices = MethodLibrary.getSliceIndices(startValue, endValue, stepValue,
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
        Printer.format(
            "Type %s has no operator [:](%s, %s, %s)",
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
    obj.validate(env);
  }
}
