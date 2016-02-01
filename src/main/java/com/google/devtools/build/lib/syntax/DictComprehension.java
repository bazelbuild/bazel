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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.Removal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Syntax node for dictionary comprehension expressions.
 */
public class DictComprehension extends AbstractComprehension {
  private final Expression keyExpression;
  private final Expression valueExpression;

  public DictComprehension(Expression keyExpression, Expression valueExpression) {
    super('{', '}', keyExpression, valueExpression);
    this.keyExpression = keyExpression;
    this.valueExpression = valueExpression;
  }

  @Override
  String printExpressions() {
    return String.format("%s: %s", keyExpression, valueExpression);
  }

  @Override
  OutputCollector createCollector() {
    return new DictOutputCollector();
  }

  @Override
  InternalVariable compileInitialization(VariableScope scope, List<ByteCodeAppender> code) {
    InternalVariable dict = scope.freshVariable(ImmutableMap.class);
    append(code, ByteCodeMethodCalls.BCImmutableMap.builder);
    code.add(dict.store());
    return dict;
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
    code.add(keyExpression.compile(scope, debugInfo));
    append(code, Duplication.SINGLE, EvalUtils.checkValidDictKey);
    code.add(valueExpression.compile(scope, debugInfo));
    append(code, ByteCodeMethodCalls.BCImmutableMap.Builder.put, Removal.SINGLE);
    return ByteCodeUtils.compoundAppender(code);
  }

  @Override
  ByteCodeAppender compileBuilding(VariableScope scope, InternalVariable collection) {
    return new ByteCodeAppender.Simple(
        collection.load(), ByteCodeMethodCalls.BCImmutableMap.Builder.build);
  }

  /**
   * Helper class that collects the intermediate results of the {@link DictComprehension} and
   * provides access to the resulting {@link Map}.
   */
  private final class DictOutputCollector implements OutputCollector {
    private final Map<Object, Object> result;

    DictOutputCollector() {
      // We want to keep the iteration order
      result = new LinkedHashMap<>();
    }

    @Override
    public void evaluateAndCollect(Environment env) throws EvalException, InterruptedException {
      Object key = keyExpression.eval(env);
      EvalUtils.checkValidDictKey(key);
      result.put(key, valueExpression.eval(env));
    }

    @Override
    public Object getResult(Environment env) throws EvalException {
      return ImmutableMap.copyOf(result);
    }
  }
}
