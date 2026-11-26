// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.query2.engine.QueryParser.ParserOptions;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommonQueryOptions}. */
@RunWith(JUnit4.class)
public class CommonQueryOptionsTest {

  @Test
  public void defaultLineTerminatorIsNewline() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse();
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isFalse();
    assertThat(options.getLineTerminator()).isEqualTo("\n");
  }

  @Test
  public void nullOptionExpandsToLineTerminatorNull() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--null");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isTrue();
    assertThat(options.getLineTerminator()).isEqualTo("\0");
  }

  @Test
  public void lineTerminatorNullOverridesNullOption() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--null", "--line_terminator_null=false");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isFalse();
    assertThat(options.getLineTerminator()).isEqualTo("\n");
  }

  @Test
  public void defaultMaxQueryExpressionDepthIsUnlimited() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse();
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getMaxQueryExpressionDepth()).isEqualTo(-1);
    assertThat(options.toParserOptions()).isEqualTo(ParserOptions.DEFAULT);
  }

  @Test
  public void maxQueryExpressionDepthSetsParserOptions() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--max_query_expression_depth=5");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getMaxQueryExpressionDepth()).isEqualTo(5);
    assertThat(options.toParserOptions())
        .isEqualTo(ParserOptions.builder().setMaxQueryDepth(5).build());
  }

  @Test
  public void maxQueryExpressionDepthZeroIsAllowed() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--max_query_expression_depth=0");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getMaxQueryExpressionDepth()).isEqualTo(0);
    assertThat(options.toParserOptions())
        .isEqualTo(ParserOptions.builder().setMaxQueryDepth(0).build());
  }

  @Test
  public void maxQueryExpressionDepthNegativeOneResetsToUnlimited() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--max_query_expression_depth=5", "--max_query_expression_depth=-1");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getMaxQueryExpressionDepth()).isEqualTo(-1);
    assertThat(options.toParserOptions()).isEqualTo(ParserOptions.DEFAULT);
  }

  @Test
  public void maxQueryExpressionDepthBelowNegativeOneThrows() {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> parser.parse("--max_query_expression_depth=-2"));
    assertThat(e).hasMessageThat().contains("'-2' should be >= -1");
  }

  @Test
  public void maxQueryExpressionDepthNonIntegerThrows() {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    assertThrows(
        OptionsParsingException.class, () -> parser.parse("--max_query_expression_depth=abc"));
  }

  @Test
  public void toParserOptionsRejectsInvalidProgrammaticDepth() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse();
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    options.setMaxQueryExpressionDepth(-2);
    assertThrows(IllegalArgumentException.class, options::toParserOptions);
  }
}
