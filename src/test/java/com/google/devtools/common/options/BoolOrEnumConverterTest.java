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

import static com.google.devtools.common.options.OptionsParser.newOptionsParser;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

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
    @Option(name = "compile_mode",
            converter = CompilationModeConverter.class,
            defaultValue = "dbg")
    public CompilationMode compileMode;
  }

  @Test
  public void converterFromEnum() throws Exception {
    CompilationModeConverter converter = new CompilationModeConverter();
    assertEquals(CompilationMode.DBG, converter.convert("dbg"));
    assertEquals(CompilationMode.OPT, converter.convert("opt"));

    try {
      converter.convert("none");
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Not a valid compilation mode: 'none' (should be dbg or opt)", e.getMessage());
    }
    assertEquals("dbg or opt", converter.getTypeDescription());
  }

  @Test
  public void convertFromBooleanValues() throws Exception {
    String[] falseValues = new String[]{"false", "0"};
    String[] trueValues = new String[]{"true", "1"};
    CompilationModeConverter converter = new CompilationModeConverter();

    for (String falseValue : falseValues) {
      assertEquals(CompilationMode.OPT, converter.convert(falseValue));
    }

    for (String trueValue : trueValues) {
      assertEquals(CompilationMode.DBG, converter.convert(trueValue));
    }
  }

  @Test
  public void prefixedWithNo() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(CompilationModeTestOptions.class);
    parser.parse("--nocompile_mode");
    CompilationModeTestOptions options =
        parser.getOptions(CompilationModeTestOptions.class);
    assertNotNull(options.compileMode);
    assertEquals(CompilationMode.OPT, options.compileMode);
  }

  @Test
  public void missingValueAsBoolConversion() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(CompilationModeTestOptions.class);
    parser.parse("--compile_mode");
    CompilationModeTestOptions options =
        parser.getOptions(CompilationModeTestOptions.class);
    assertNotNull(options.compileMode);
    assertEquals(CompilationMode.DBG, options.compileMode);
  }

}
