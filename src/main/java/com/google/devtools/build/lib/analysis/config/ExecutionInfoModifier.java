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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents a list of regexes over mnemonics and changes to add or remove keys, parsed from a
 * {@code --modify_execution_info} option.
 */
public final class ExecutionInfoModifier {

  private static final Pattern MODIFIER_PATTERN =
      Pattern.compile("^(?<pattern>.+)=(?<sign>[+-])(?<key>.+)$");

  private final ImmutableList<Expression> expressions;
  private final String option;

  @AutoValue
  abstract static class Expression {
    abstract Pattern pattern();

    abstract boolean remove();

    abstract String key();
  }

  private ExecutionInfoModifier(String option, ImmutableList<Expression> expressions) {
    this.expressions = expressions;
    this.option = option;
  }

  /** Constructs an instance of ExecutionInfoModifier by parsing an option string. */
  public static class Converter
      implements com.google.devtools.common.options.Converter<ExecutionInfoModifier> {
    @Override
    public ExecutionInfoModifier convert(String input) throws OptionsParsingException {

      String[] expressionSpecs = input.split(",");
      if (expressionSpecs.length == 0) {
        throw new OptionsParsingException("expected 'regex=[+-]key,regex=[+-]key,...'", input);
      }
      ImmutableList.Builder<Expression> expressionBuilder = ImmutableList.builder();
      for (String spec : expressionSpecs) {
        Matcher specMatcher = MODIFIER_PATTERN.matcher(spec);
        if (!specMatcher.matches()) {
          throw new OptionsParsingException(
              String.format("malformed expression '%s'", spec), input);
        }
        expressionBuilder.add(
            new AutoValue_ExecutionInfoModifier_Expression(
                new RegexPatternConverter().convert(specMatcher.group("pattern")),
                specMatcher.group("sign").equals("-"),
                specMatcher.group("key")));
      }
      return new ExecutionInfoModifier(input, expressionBuilder.build());
    }

    @Override
    public String getTypeDescription() {
      return "regex=[+-]key,regex=[+-]key,...";
    }
  }

  /**
   * Determines whether the given {@code mnemonic} (e.g. "CppCompile") matches any of the patterns.
   */
  public boolean matches(String mnemonic) {
    return expressions.stream().anyMatch(expr -> expr.pattern().matcher(mnemonic).matches());
  }

  /** Modifies the given map of {@code executionInfo} to add or remove the keys for this option. */
  void apply(String mnemonic, Map<String, String> executionInfo) {
    for (Expression expr : expressions) {
      if (expr.pattern().matcher(mnemonic).matches()) {
        if (expr.remove()) {
          executionInfo.remove(expr.key());
        } else {
          executionInfo.put(expr.key(), "");
        }
      }
    }
  }

  @Override
  public String toString() {
    return option;
  }
}
