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
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.FilteringQueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;

/**
 * An abstract class that provides generic regex filter expression. Actual expression are
 * implemented by the subclasses.
 */
public abstract class RegexFilterExpression extends FilteringQueryFunction {
  protected final boolean invert;

  protected RegexFilterExpression(boolean invert) {
    this.invert = invert;
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      final QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      final List<Argument> args,
      Callback<T> callback) {
    String rawPattern = getPattern(args);
    final Pattern compiledPattern;
    try {
      compiledPattern = Pattern.compile(rawPattern);
    } catch (PatternSyntaxException e) {
      return env.immediateFailedFuture(
          new QueryException(
              expression,
              String.format(
                  "illegal '%s' pattern regexp '%s': %s", getName(), rawPattern, e.getMessage()),
              Query.Code.SYNTAX_ERROR));
    }

    // Note that Patttern#matcher is thread-safe and so this Predicate can safely be used
    // concurrently.
    final Predicate<T> matchFilter =
        target -> {
          for (String str : getFilterStrings(env, args, target)) {
            if ((str != null) && compiledPattern.matcher(str).find()) {
              return !invert;
            }
          }
          return invert;
        };

    return env.eval(
        args.get(getExpressionToFilterIndex()).getExpression(),
        context,
        new FilteredCallback<>(callback, matchFilter));
  }

  @Override
  public final int getExpressionToFilterIndex() {
    return getMandatoryArguments() - 1;
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

  private static final class FilteredCallback<T> implements Callback<T> {
    private final Callback<T> parentCallback;
    private final Predicate<T> retainIfTrue;

    private FilteredCallback(Callback<T> parentCallback, Predicate<T> retainIfTrue) {
      this.parentCallback = parentCallback;
      this.retainIfTrue = retainIfTrue;
    }

    @Override
    public void process(Iterable<T> partialResult) throws QueryException, InterruptedException {
      // Consume as much of the iterable as possible here. The parent callback may be synchronized,
      // so we can avoid calling it at all if everything gets filtered out.
      Iterator<T> it = partialResult.iterator();
      while (it.hasNext()) {
        T next = it.next();
        if (retainIfTrue.apply(next)) {
          Iterator<T> rest =
              Iterators.concat(
                  Iterators.singletonIterator(next), Iterators.filter(it, retainIfTrue));
          parentCallback.process(new InProgressIterable<>(rest, partialResult, retainIfTrue));
          break;
        }
      }
    }

    @Override
    public String toString() {
      return "filtered parentCallback of : " + retainIfTrue;
    }

    /**
     * Specialized {@link Iterable} to resume an in-progress iteration on the first call to {@link
     * #iterator}.
     */
    private static final class InProgressIterable<T> implements Iterable<T> {
      @Nullable private Iterator<T> inProgress;
      private final Iterable<T> original;
      private final Predicate<T> retainIfTrue;

      private InProgressIterable(
          Iterator<T> inProgress, Iterable<T> original, Predicate<T> retainIfTrue) {
        this.inProgress = inProgress;
        this.original = original;
        this.retainIfTrue = retainIfTrue;
      }

      @Override
      public Iterator<T> iterator() {
        synchronized (this) {
          if (inProgress != null) {
            Iterator<T> it = inProgress;
            inProgress = null;
            return it;
          }
        }
        return Iterators.filter(original.iterator(), retainIfTrue);
      }
    }
  }
}
