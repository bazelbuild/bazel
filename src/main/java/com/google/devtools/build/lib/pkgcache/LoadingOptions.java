// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import net.starlark.java.syntax.BinaryOperatorExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.UnaryOperatorExpression;

import javax.annotation.Nullable;
import java.util.List;
import java.util.Set;

/**
 * Options that affect how command-line target patterns are resolved to individual targets.
 */
public class LoadingOptions extends OptionsBase {
  @Option(
    name = "build_tests_only",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If specified, only *_test and test_suite rules will be built and other targets specified "
            + "on the command line will be ignored. By default everything that was requested "
            + "will be built."
  )
  public boolean buildTestsOnly;

  @Option(
    name = "compile_one_dependency",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Compile a single dependency of the argument files. This is useful for syntax checking "
            + "source files in IDEs, for example, by rebuilding a single target that depends on "
            + "the source file to detect errors as early as possible in the edit/build/test cycle. "
            + "This argument affects the way all non-flag arguments are interpreted; instead of "
            + "being targets to build they are source filenames.  For each source filename "
            + "an arbitrary target that depends on it will be built."
  )
  public boolean compileOneDependency;

  @Option(
    name = "build_tag_filters",
    converter = TagFilterConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A boolean formula over rule tags that selects the targets to build. "
            + "Example: '(server or client) and not experimental'. "
            + "See --test_tag_filters for details. This option "
            + "does not affect the set of tests executed with the 'test' command; those are be "
            + "governed by the test filtering options, for example '--test_tag_filters'"
  )
  public TagFilter buildTagFilter;

  @Option(
    name = "test_tag_filters",
    converter = TagFilterConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A boolean formula over test tags that selects the tests to run. "
            + "Example: '(release or smoke) and not flaky'. Formulas have the form: "
            + "expr = IDENT | expr 'or' expr | expr 'and' expr | 'not' expr | '(' expr ')'. "
            + "IDENT is defined as a Starlark identifier, with the same reserved words. "
            + "Comma-separated list of tags are also supported: tag[,tag]*. Each tag can be optionally "
            + "preceded with '-' to specify excluded tags."
  )
  public TagFilter testTagFilter;

  /**
   * TagFilterConverter parses a tag filter, which can either be a boolean expression or a comma-separated list.
   *
   * Declared public to make it accessible to reflection.
   */
  public static class TagFilterConverter implements Converter<TagFilter> {
    @Override
    @Nullable
    public TagFilter convert(String input) throws OptionsParsingException {
      return new TagFilter(input);
    }

    @Override
    public String getTypeDescription() {
      return "A boolean formula over test tags (e.g. '(release or smoke) and not flaky')";
    }
  }

  @Option(
    name = "test_size_filters",
    converter = TestSize.TestSizeFilterConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test sizes. Each size can be optionally "
            + "preceded with '-' to specify excluded sizes. Only those test targets will be "
            + "found that contain at least one included size and do not contain any excluded "
            + "sizes. This option affects --build_tests_only behavior and the test command."
  )
  public Set<TestSize> testSizeFilterSet;

  @Option(
    name = "test_timeout_filters",
    converter = TestTimeout.TestTimeoutFilterConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test timeouts. Each timeout can be "
            + "optionally preceded with '-' to specify excluded timeouts. Only those test "
            + "targets will be found that contain at least one included timeout and do not "
            + "contain any excluded timeouts. This option affects --build_tests_only behavior "
            + "and the test command."
  )
  public Set<TestTimeout> testTimeoutFilterSet;

  @Option(
    name = "test_lang_filters",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies a comma-separated list of test languages. Each language can be "
            + "optionally preceded with '-' to specify excluded languages. Only those "
            + "test targets will be found that are written in the specified languages. "
            + "The name used for each language should be the same as the language prefix in the "
            + "*_test rule, e.g. one of 'cc', 'java', 'py', etc. "
            + "This option affects --build_tests_only behavior and the test command."
  )
  public List<String> testLangFilterList;

  @Option(
    name = "build_manual_tests",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Forces test targets tagged 'manual' to be built. 'manual' tests are excluded from "
            + "processing. This option forces them to be built (but not executed)."
  )
  public boolean buildManualTests;

  @Deprecated
  @Option(
    name = "experimental_skyframe_target_pattern_evaluator",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Use the Skyframe-based target pattern evaluator; implies "
            + "--experimental_interleave_loading_and_analysis."
  )
  public boolean useSkyframeTargetPatternEvaluator;

  @Option(
    name = "expand_test_suites",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
    help = "Expand test_suite targets into their constituent tests before analysis. "
        + "When this flag is turned on (the default), negative target patterns will apply "
        + "to the tests belonging to the test suite, otherwise they will not. "
        + "Turning off this flag is useful when top-level aspects are applied at command line: "
        + "then they can analyze test_suite targets."
  )
  public boolean expandTestSuites;

  /** BooleanFormulaConverter parses a boolean formula to an Expression.
   * Declared public to make it accessible to reflection. */
  public static class BooleanFormulaConverter implements Converter<Expression> {
    @Override
    @Nullable public Expression convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return null;
      }
      if (isLegacySyntax(input)) {
        input = convertLegacySyntaxToBooleanFormula(input);
      }
      try {
        Expression result = Expression.parse(ParserInput.fromLines(input));
        validate(result);
        return result;
      } catch (SyntaxError.Exception e) {
        throw new OptionsParsingException("Failed to parse expression: " + e.getMessage() + " input: " + input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "A Boolean formula over test tags (e.g. '(release or smoke) and not flaky')";
    }

    private static boolean isLegacySyntax(String input) {
      return input.contains(",") || input.contains("-");
    }

    /**
     * Converts a string containing tags separated by comma to a boolean formula.
     */
    private String convertLegacySyntaxToBooleanFormula(String input) throws OptionsParsingException {
      return input.replaceAll(" *, *-", " and not ")
              .replaceAll(" *, *", " or ")
              .replace("-", " not ").trim();
    }

    /**
     * Throws an OptionsParsingException when the given boolean formula does not follow the grammar:
     * expr = IDENT | expr 'or' expr | expr 'and' expr | 'not' expr | '(' expr ')'
     */
    private void validate(Expression expression) throws OptionsParsingException {
      switch (expression.kind()) {
        case IDENTIFIER:
          break;
        case BINARY_OPERATOR:
          BinaryOperatorExpression boe = (BinaryOperatorExpression) expression;
          if (boe.getOperator() != TokenKind.OR && boe.getOperator() != TokenKind.AND) {
            throw new OptionsParsingException(String.format("invalid Boolean operator: %s (want 'and' or 'or')",
                boe.getOperator()));
          }
          validate(boe.getX());
          validate(boe.getY());
          break;
        case UNARY_OPERATOR:
          UnaryOperatorExpression uoe = (UnaryOperatorExpression) expression;
          if (uoe.getOperator() != TokenKind.NOT) {
            throw new OptionsParsingException(String.format("invalid Boolean operator: %s (want 'not')",
                uoe.getOperator()));
          }
          validate(uoe.getX());
          break;
        default:
          throw new OptionsParsingException("invalid Boolean operator");
      }
    }
  }
}
