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

package com.google.devtools.build.lib.analysis.config;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents a list of regexes over mnemonics and changes to add or remove keys, parsed from a
 * {@code --modify_execution_info} option.
 */
@AutoValue
public abstract class ExecutionInfoModifier {

  private static final Pattern MODIFIER_PATTERN =
      Pattern.compile("^(?<pattern>.+)=(?<sign>[+-])(?<key>.+)$");

  private static final ExecutionInfoModifier EMPTY = create("", ImmutableList.of());

  abstract String option();

  abstract ImmutableList<Expression> expressions();

  // Configuration checksums repeatedly serialize these immutable modifiers.
  @Memoized
  @Override
  public abstract String toString();

  @AutoValue
  abstract static class Expression {
    // Patterns do not have a useful equals(), so compare by the regex and memoize the derived
    // Pattern.
    @Memoized
    Pattern pattern() {
      return Pattern.compile(regex());
    }

    abstract String regex();

    abstract boolean remove();

    abstract String key();
  }

  /** Constructs an instance of ExecutionInfoModifier by parsing an option string. */
  public static class Converter
      extends com.google.devtools.common.options.Converter.Contextless<ExecutionInfoModifier> {
    @Override
    public ExecutionInfoModifier convert(String input) throws OptionsParsingException {
      if (Strings.isNullOrEmpty(input)) {
        return EMPTY;
      }

      ImmutableList.Builder<Expression> expressionBuilder = ImmutableList.builder();
      for (String spec : Splitter.on(",").split(input)) {
        Matcher specMatcher = MODIFIER_PATTERN.matcher(spec);
        if (!specMatcher.matches()) {
          throw new OptionsParsingException(
              String.format("malformed expression '%s'", spec), input);
        }
        expressionBuilder.add(
            new AutoValue_ExecutionInfoModifier_Expression(
                // Convert to get a useful exception if it's not a valid pattern, but use the regex
                // (see comment in Expression)
                new RegexPatternConverter()
                    .convert(specMatcher.group("pattern"), /* conversionContext= */ null)
                    .regexPattern()
                    .pattern(),
                specMatcher.group("sign").equals("-"),
                specMatcher.group("key")));
      }
      return ExecutionInfoModifier.create(input, expressionBuilder.build());
    }

    @Override
    public String getTypeDescription() {
      return "regex=[+-]key,regex=[+-]key,...";
    }

    @Override
    public boolean starlarkConvertible() {
      return true;
    }

    @Override
    public String reverseForStarlark(Object converted) {
      return ((ExecutionInfoModifier) converted).option();
    }
  }

  private static ExecutionInfoModifier create(String input, ImmutableList<Expression> expressions) {
    return new AutoValue_ExecutionInfoModifier(input, expressions);
  }

  /**
   * Cache from mnemonic to the expressions whose pattern matches it.
   *
   * <p>This is queried for every action, but there are only a few distinct mnemonics, so caching
   * avoids running the regexes (and allocating a {@link Matcher}) per action. Being memoized, the
   * cache is not part of {@code equals}, {@code hashCode} or serialization.
   */
  @Memoized
  ConcurrentHashMap<String, ImmutableList<Expression>> matchingExpressionsCache() {
    return new ConcurrentHashMap<>();
  }

  /** Returns the expressions whose pattern matches the given {@code mnemonic}. */
  private ImmutableList<Expression> matchingExpressions(String mnemonic) {
    ImmutableList<Expression> expressions = expressions();
    if (expressions.isEmpty()) {
      return expressions;
    }
    ConcurrentHashMap<String, ImmutableList<Expression>> cache = matchingExpressionsCache();
    // Try a plain lookup first to avoid allocating a capturing lambda on the hot path.
    ImmutableList<Expression> result = cache.get(mnemonic);
    if (result == null) {
      result = cache.computeIfAbsent(mnemonic, this::computeMatchingExpressions);
    }
    return result;
  }

  private ImmutableList<Expression> computeMatchingExpressions(String mnemonic) {
    String unicodeMnemonic = internalToUnicode(mnemonic);
    ImmutableList.Builder<Expression> result = ImmutableList.builder();
    for (Expression expr : expressions()) {
      if (expr.pattern().matcher(unicodeMnemonic).matches()) {
        result.add(expr);
      }
    }
    return result.build();
  }

  /**
   * Determines whether the given {@code mnemonic} (e.g. "CppCompile") matches any of the patterns.
   */
  boolean matches(String mnemonic) {
    return !matchingExpressions(mnemonic).isEmpty();
  }

  /** Checks whether the {@code executionInfoList} matches the {@code mnemonic}. */
  public static boolean matches(List<ExecutionInfoModifier> executionInfoList, String mnemonic) {
    for (int i = 0; i < executionInfoList.size(); i++) {
      if (executionInfoList.get(i).matches(mnemonic)) {
        return true;
      }
    }
    return false;
  }

  /** Checks whether applying this modifier would actually change the given map. */
  // When --modify_execution_info is non-empty, this method helps reduce CPU/memory overhead from
  // modifying every action's execution info map.
  boolean wouldChange(String mnemonic, Map<String, String> executionInfo) {
    for (Expression expr : matchingExpressions(mnemonic)) {
      if (expr.remove()) {
        if (executionInfo.containsKey(expr.key())) {
          return true;
        }
      } else {
        String val = executionInfo.get(expr.key());
        if (val == null || !val.isEmpty()) {
          return true;
        }
      }
    }
    return false;
  }

  /** Checks whether any modifier in the list would actually change the given map. */
  public static boolean wouldChange(
      List<ExecutionInfoModifier> executionInfoList,
      String mnemonic,
      Map<String, String> executionInfo) {
    for (int i = 0; i < executionInfoList.size(); i++) {
      if (executionInfoList.get(i).wouldChange(mnemonic, executionInfo)) {
        return true;
      }
    }
    return false;
  }

  /** Applies {@code executionInfoList} to the given {@code executionInfo}. */
  public static void apply(
      List<ExecutionInfoModifier> executionInfoList,
      String mnemonic,
      Map<String, String> executionInfo) {
    for (int i = 0; i < executionInfoList.size(); i++) {
      executionInfoList.get(i).apply(mnemonic, executionInfo);
    }
  }

  /** Modifies the given map of {@code executionInfo} to add or remove the keys for this option. */
  void apply(String mnemonic, Map<String, String> executionInfo) {
    for (Expression expr : matchingExpressions(mnemonic)) {
      if (expr.remove()) {
        executionInfo.remove(expr.key());
      } else {
        executionInfo.put(expr.key(), "");
      }
    }
  }
}
