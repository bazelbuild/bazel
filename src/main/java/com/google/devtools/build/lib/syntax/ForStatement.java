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

import static com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils.append;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.FlowStatement.FlowException;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.IntegerVariableIncrease;
import com.google.devtools.build.lib.syntax.compiler.Jump;
import com.google.devtools.build.lib.syntax.compiler.Jump.PrimitiveComparison;
import com.google.devtools.build.lib.syntax.compiler.LabelAdder;
import com.google.devtools.build.lib.syntax.compiler.LoopLabels;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;

/**
 * Syntax node for a for loop statement.
 */
public final class ForStatement extends Statement {

  private final LValue variable;
  private final Expression collection;
  private final ImmutableList<Statement> block;

  /**
   * Constructs a for loop statement.
   */
  ForStatement(Expression variable, Expression collection, List<Statement> block) {
    this.variable = new LValue(Preconditions.checkNotNull(variable));
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

  public LValue getVariable() {
    return variable;
  }

  /**
   * @return The collection we iterate on, e.g. `col` in `for x in col:`
   */
  public Expression getCollection() {
    return collection;
  }

  public ImmutableList<Statement> block() {
    return block;
  }

  @Override
  public String toString() {
    // TODO(bazel-team): if we want to print the complete statement, the function
    // needs an extra argument to specify indentation level.
    return "for " + variable + " in " + collection + ": ...\n";
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    Object o = collection.eval(env);
    Iterable<?> col = EvalUtils.toIterable(o, getLocation());
    EvalUtils.lock(o, getLocation());
    try {
      for (Object it : col) {
        variable.assign(env, getLocation(), it);

        try {
          for (Statement stmt : block) {
            stmt.exec(env);
          }
        } catch (FlowException ex) {
          if (ex.mustTerminateLoop()) {
            return;
          }
        }
      }
    } finally {
      EvalUtils.unlock(o, getLocation());
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    // TODO(bazel-team): validate variable. Maybe make it temporarily readonly.
    collection.validate(env);
    variable.validate(env, getLocation());

    for (Statement stmt : block) {
      stmt.validate(env);
    }
  }

  @Override
  ByteCodeAppender compile(
      VariableScope scope, Optional<LoopLabels> outerLoopLabels, DebugInfo debugInfo)
      throws EvalException {
    // TODO(bazel-team): Remove obsolete logic for counting size of iterated collection.
    AstAccessors debugAccessors = debugInfo.add(this);
    List<ByteCodeAppender> code = new ArrayList<>();
    InternalVariable originalIterable =
        scope.freshVariable(new TypeDescription.ForLoadedType(Iterable.class));
    InternalVariable iterator =
        scope.freshVariable(new TypeDescription.ForLoadedType(Iterator.class));
    // compute the collection and get it on the stack and transform it to the right type
    code.add(collection.compile(scope, debugInfo));
    append(code, debugAccessors.loadLocation, EvalUtils.toIterable, Duplication.SINGLE);
    // save it for later concurrent modification check
    code.add(originalIterable.store());
    append(
        code,
        ByteCodeMethodCalls.BCImmutableList.copyOf,
        ByteCodeMethodCalls.BCImmutableList.iterator);
    code.add(iterator.store());
    // for counting the size during the loop
    InternalVariable sizeCounterVariable =
        scope.freshVariable(new TypeDescription.ForLoadedType(int.class));
    LabelAdder loopHeader = new LabelAdder();
    LabelAdder loopBody = new LabelAdder();
    LabelAdder breakLoop = new LabelAdder();
    // for passing on the labels for continue/break statements
    Optional<LoopLabels> loopLabels = LoopLabels.of(loopHeader.getLabel(), breakLoop.getLabel());
    append(
        code,
        // initialize loop counter
        IntegerConstant.ZERO);
    code.add(sizeCounterVariable.store());
    append(code, Jump.to(loopHeader), loopBody, iterator.load());
    append(code, ByteCodeMethodCalls.BCIterator.next);
    // store current element into l-value
    code.add(variable.compileAssignment(this, debugAccessors, scope));
    // compile code for the body
    for (Statement statement : block) {
      append(code, new IntegerVariableIncrease(sizeCounterVariable, 1));
      code.add(statement.compile(scope, loopLabels, debugInfo));
    }
    // compile code for the loop header
    append(
        code,
        loopHeader,
        iterator.load(),
        ByteCodeMethodCalls.BCIterator.hasNext,
        // falls through to end of loop if hasNext() was false, otherwise jumps back
        Jump.ifIntOperandToZero(PrimitiveComparison.NOT_EQUAL).to(loopBody),
        breakLoop);
    return ByteCodeUtils.compoundAppender(code);
  }
}
