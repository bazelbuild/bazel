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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;

import java.util.List;
import java.util.regex.Pattern;

/**
 * An abstract class that provides generic regex filter expression. Actual
 * expression are implemented by the subclasses.
 */
public abstract class RegexFilterExpression implements QueryFunction {
  protected RegexFilterExpression() {
  }

  @Override
  public <T> void eval(
      final QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      final List<Argument> args,
      Callback<T> callback)
      throws QueryException, InterruptedException {
    final Pattern compiledPattern;
    try {
      compiledPattern = Pattern.compile(getPattern(args));
    } catch (IllegalArgumentException e) {
      throw new QueryException(expression, "illegal pattern regexp in '" + this + "': "
                               + e.getMessage());
    }

    final Predicate<T> matchFilter = new Predicate<T>() {
      @Override
      public boolean apply(T target) {
        for (String str : getFilterStrings(env, args, target)) {
          if ((str != null) && compiledPattern.matcher(str).find()) {
            return true;
          }
        }
        return false;
      }
    };

    env.eval(
        Iterables.getLast(args).getExpression(),
        context,
        QueryUtil.filteredCallback(callback, matchFilter));
  }

  /**
   * Returns string for the given target that must be matched against pattern.
   * May return null, in which case matching is guaranteed to fail.
   */
  protected abstract <T> String getFilterString(
      QueryEnvironment<T> env, List<Argument> args, T target);

  /**
   * Returns a list of strings for the given target that must be matched against
   * pattern. The filter matches if *any* of these strings matches.
   *
   * <p>Unless subclasses have an explicit reason to override this method, it's fine
   * to keep the default implementation that just delegates to {@link #getFilterString}.
   * Overriding this method is useful for subclasses that want to match against a
   * universe of possible values. For example, with configurable attributes, an
   * attribute might have different values depending on the build configuration. One
   * may wish the filter to match if *any* of those values matches.
   */
  protected <T> Iterable<String> getFilterStrings(
      QueryEnvironment<T> env, List<Argument> args, T target) {
    String filterString = getFilterString(env, args, target);
    return filterString == null ? ImmutableList.<String>of() : ImmutableList.of(filterString);
  }

  protected abstract String getPattern(List<Argument> args);
}
