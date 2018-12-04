// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * A label(pattern, argument) filter expression, which computes the set of resulting actions of
 * argument whose inputs matches the regexp 'pattern'. The pattern follows java.util.regex format.
 *
 * <pre>expr ::= INPUTS '(' WORD ',' expr ')'</pre>
 *
 * Example patterns:
 *
 * <pre>
 * '//third_party/a.java'      Match all actions whose inputs includes //third_party/a.java
 * '//third_party/.*\.js'      Match all actions whose inputs includes js files under //third_party/
 * '.*'                        Match all actions
 * '*'                         Error: invalid regex
 * </pre>
 */
public class InputsFunction implements QueryFunction {

  @Override
  public String getName() {
    return "inputs";
  }

  @Override
  public int getMandatoryArguments() {
    return 2;
  }

  @Override
  public Iterable<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.WORD, ArgumentType.EXPRESSION);
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {
    String inputsPattern = args.get(0).getWord();
    try {
      Pattern.compile(inputsPattern);
    } catch (PatternSyntaxException e) {
      return env.immediateFailedFuture(
          new QueryException(
              expression,
              String.format(
                  "Illegal '%s' pattern regexp '%s': %s",
                  getName(), inputsPattern, e.getMessage())));
    }

    // Do nothing, pass the expression along
    return env.eval(args.get(1).getExpression(), context, callback);
  }
}
