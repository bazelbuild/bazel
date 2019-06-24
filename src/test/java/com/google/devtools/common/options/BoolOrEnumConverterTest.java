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

package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link BoolOrEnumConverter}.
 */
@RunWith(JUnit4.class)
public class BoolOrEnumConverterTest {

  private enum CompilationMode {
    DBG, OPT
  }

  private static class CompilationModeConverter
    extends BoolOrEnumConverter<CompilationMode> {

    public CompilationModeConverter() {
      super(CompilationMode.class, "compilation mode",
          CompilationMode.DBG, CompilationMode.OPT);
    }
  }

  /**
   * The test options for the CompilationMode hybrid converter.
   */
  public static class CompilationModeTestOptions extends OptionsBase {
    @Option(
      name = "compile_mode",
      converter = CompilationModeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "dbg"
    )
    public CompilationMode compileMode;
  }

  @Test
  public void converterFromEnum() throws Exception {
    CompilationModeConverter converter = new CompilationModeConverter();
    assertThat(converter.convert("dbg")).isEqualTo(CompilationMode.DBG);
    assertThat(converter.convert("opt")).isEqualTo(CompilationMode.OPT);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("none"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Not a valid compilation mode: 'none' (should be dbg or opt)");
    assertThat(converter.getTypeDescription()).isEqualTo("dbg or opt");
  }

  @Test
  public void convertFromBooleanValues() throws Exception {
    String[] falseValues = new String[]{"false", "0"};
    String[] trueValues = new String[]{"true", "1"};
    CompilationModeConverter converter = new CompilationModeConverter();

    for (String falseValue : falseValues) {
      assertThat(converter.convert(falseValue)).isEqualTo(CompilationMode.OPT);
    }

    for (String trueValue : trueValues) {
      assertThat(converter.convert(trueValue)).isEqualTo(CompilationMode.DBG);
    }
  }

  @Test
  public void prefixedWithNo() throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(CompilationModeTestOptions.class).build();
    parser.parse("--nocompile_mode");
    CompilationModeTestOptions options =
        parser.getOptions(CompilationModeTestOptions.class);
    assertThat(options.compileMode).isNotNull();
    assertThat(options.compileMode).isEqualTo(CompilationMode.OPT);
  }

  @Test
  public void missingValueAsBoolConversion() throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(CompilationModeTestOptions.class).build();
    parser.parse("--compile_mode");
    CompilationModeTestOptions options =
        parser.getOptions(CompilationModeTestOptions.class);
    assertThat(options.compileMode).isNotNull();
    assertThat(options.compileMode).isEqualTo(CompilationMode.DBG);
  }

}
