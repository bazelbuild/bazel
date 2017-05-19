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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link EnumConverter}.
 */
@RunWith(JUnit4.class)
public class EnumConverterTest {

  private enum CompilationMode {
    DBG, OPT
  }

  private static class CompilationModeConverter
    extends EnumConverter<CompilationMode> {

    public CompilationModeConverter() {
      super(CompilationMode.class, "compilation mode");
    }
  }

  @Test
  public void converterForEnumWithTwoValues() throws Exception {
    CompilationModeConverter converter = new CompilationModeConverter();
    assertEquals(CompilationMode.DBG, converter.convert("dbg"));
    assertEquals(CompilationMode.OPT, converter.convert("opt"));
    try {
      converter.convert("none");
      fail();
    } catch(OptionsParsingException e) {
      assertEquals("Not a valid compilation mode: 'none' (should be dbg or opt)", e.getMessage());
    }
    assertEquals("dbg or opt", converter.getTypeDescription());
  }

  private enum Fruit {
    Apple, Banana, Cherry
  }

  private static class FruitConverter extends EnumConverter<Fruit> {

    public FruitConverter() {
      super(Fruit.class, "fruit");
    }
  }

  @Test
  public void typeDescriptionForEnumWithThreeValues() throws Exception {
    FruitConverter converter = new FruitConverter();
    // We always use lowercase in the user-visible messages:
    assertEquals("apple, banana or cherry",
                 converter.getTypeDescription());
  }

  @Test
  public void converterIsCaseInsensitive() throws Exception {
    FruitConverter converter = new FruitConverter();
    assertSame(Fruit.Banana, converter.convert("bAnANa"));
  }

  // Regression test: lists of enum using a subclass of EnumConverter don't work
  private static class AlphabetEnumConverter extends EnumConverter<AlphabetEnum> {
    public AlphabetEnumConverter() {
      super(AlphabetEnum.class, "alphabet enum");
    }
  }

  private static enum AlphabetEnum {
    ALPHA, BRAVO, CHARLY, DELTA, ECHO
  }

  public static class EnumListTestOptions extends OptionsBase {
    @Option(name = "goo",
            allowMultiple = true,
            converter = AlphabetEnumConverter.class,
            defaultValue = "null")
    public List<AlphabetEnum> goo;
  }

  @Test
  public void enumList() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(EnumListTestOptions.class);
    parser.parse("--goo=ALPHA", "--goo=BRAVO");
    EnumListTestOptions options = parser.getOptions(EnumListTestOptions.class);
    assertNotNull(options.goo);
    assertEquals(2, options.goo.size());
    assertEquals(AlphabetEnum.ALPHA, options.goo.get(0));
    assertEquals(AlphabetEnum.BRAVO, options.goo.get(1));
  }

}
