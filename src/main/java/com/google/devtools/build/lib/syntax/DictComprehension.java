// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableMap;

import java.util.LinkedHashMap;
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
