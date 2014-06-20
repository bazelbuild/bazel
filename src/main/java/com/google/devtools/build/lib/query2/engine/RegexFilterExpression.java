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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Predicate;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;

import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * An abstract class that provides generic regex filter expression. Actual
 * expression are implemented by the subclasses.
 */
abstract class RegexFilterExpression implements QueryFunction {
  protected RegexFilterExpression() {
  }

  @Override
  public <T> Set<Node<T>> eval(
      final QueryEnvironment<T> env, QueryExpression expression, final List<Argument> args)
      throws QueryException {
    final Pattern compiledPattern;
    try {
      compiledPattern = Pattern.compile(getPattern(args));
    } catch (IllegalArgumentException e) {
      throw new QueryException(expression, "illegal pattern regexp in '" + this + "': "
                               + e.getMessage());
    }

    QueryExpression argument = args.get(args.size() - 1).getExpression();
    return QueryUtils.filterTargets(argument.eval(env), new Predicate<T>() {
        @Override
        public boolean apply(T target) {
          String str = getFilterString(env, args, target);
          return (str != null) && compiledPattern.matcher(str).find();
        }
      });
  }

  /**
   * Returns string for the given target that must be matched against pattern.
   * May return null, in which case matching is guaranteed to fail.
   */
  protected abstract <T> String getFilterString(
      QueryEnvironment<T> env, List<Argument> args, T target);

  protected abstract String getPattern(List<Argument> args);
}
