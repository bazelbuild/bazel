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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ProcessorWithUniquifier;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

/**
 * A label(attr_name, argument) expression, which computes the set of targets
 * whose labels appear in the specified attribute of some rule in 'argument'.
 *
 * <pre>expr ::= LABELS '(' WORD ',' expr ')'</pre>
 *
 * Example:
 * <pre>
 *  labels(srcs, //foo)      The 'srcs' source files to the //foo rule.
 * </pre>
 */
class LabelsFunction implements QueryFunction {
  LabelsFunction() {
  }

  @Override
  public String getName() {
    return "labels";
  }

  @Override
  public int getMandatoryArguments() {
    return 2;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.WORD, ArgumentType.EXPRESSION);
  }

  private static class ProcessorImpl<T> implements ProcessorWithUniquifier<T> {
    private final QueryExpression expression;
    private final String attrName;
    private final QueryEnvironment<T> env;

    private ProcessorImpl(QueryExpression expression, String attrName, QueryEnvironment<T> env) {
      this.expression = expression;
      this.attrName = attrName;
      this.env = env;
    }

    @Override
    public void process(
        Iterable<T> partialResult,
        Uniquifier<T> uniquifier,
        Callback<T> callback) throws QueryException, InterruptedException {
      for (T input : partialResult) {
        if (env.getAccessor().isRule(input)) {
          List<T> targets = uniquifier.unique(
              env.getAccessor().getLabelListAttr(expression, input, attrName,
                  "in '" + attrName + "' of rule " + env.getAccessor().getLabel(input) + ": "));
          List<T> result = new ArrayList<>(targets.size());
          for (T target : targets) {
            result.add(env.getOrCreate(target));
          }
          callback.process(result);
        }
      }
    }
  }

  private static <T> void doEval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) throws QueryException, InterruptedException {
    String attrName = args.get(0).getWord();
    env.eval(
        args.get(1).getExpression(),
        context,
        QueryUtil.compose(
            new ProcessorImpl<T>(expression, attrName, env),
            env.createUniquifier(),
            callback));
  }

  @Override
  public <T> void eval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) throws QueryException, InterruptedException {
    doEval(env, context, expression, args, callback);
  }

  @Override
  public <T> void parEval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      ThreadSafeCallback<T> callback,
      ForkJoinPool forkJoinPool) throws QueryException, InterruptedException {
    doEval(env, context, expression, args, callback);
  }
}
