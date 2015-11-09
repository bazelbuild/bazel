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

import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.NewObject;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Removal;

import java.util.ArrayList;
import java.util.List;

/**
 * Syntax node for lists comprehension expressions.
 *
 */
public final class ListComprehension extends AbstractComprehension {
  private final Expression outputExpression;

  public ListComprehension(Expression outputExpression) {
    super('[', ']', outputExpression);
    this.outputExpression = outputExpression;
  }

  @Override
  String printExpressions() {
    return outputExpression.toString();
  }

  @Override
  OutputCollector createCollector() {
    return new ListOutputCollector();
  }

  @Override
  InternalVariable compileInitialization(VariableScope scope, List<ByteCodeAppender> code) {
    InternalVariable list = scope.freshVariable(ArrayList.class);
    append(code, NewObject.fromConstructor(ArrayList.class).arguments());
    code.add(list.store());
    return list;
  }

  @Override
  ByteCodeAppender compileCollector(
      VariableScope scope,
      InternalVariable collection,
      DebugInfo debugInfo,
      AstAccessors debugAccessors)
      throws EvalException {
    List<ByteCodeAppender> code = new ArrayList<>();
    append(code, collection.load());
    code.add(outputExpression.compile(scope, debugInfo));
    append(code, ByteCodeMethodCalls.BCList.add, Removal.SINGLE);
    return ByteCodeUtils.compoundAppender(code);
  }

  @Override
  ByteCodeAppender compileBuilding(VariableScope scope, InternalVariable collection) {
    return new ByteCodeAppender.Simple(
        NewObject.fromConstructor(MutableList.class, Iterable.class, Environment.class)
            .arguments(collection.load(), scope.loadEnvironment()));
  }

  /**
   * Helper class that collects the intermediate results of the {@code ListComprehension} and
   * provides access to the resulting {@code List}.
   */
  private final class ListOutputCollector implements OutputCollector {
    private final List<Object> result;

    ListOutputCollector() {
      result = new ArrayList<>();
    }

    @Override
    public void evaluateAndCollect(Environment env) throws EvalException, InterruptedException {
      result.add(outputExpression.eval(env));
    }

    @Override
    public Object getResult(Environment env) throws EvalException {
      return new MutableList(result, env);
    }
  }
}
