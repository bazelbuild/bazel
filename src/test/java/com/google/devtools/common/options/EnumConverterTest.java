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

package com.google.devtools.common.options;

import static com.google.devtools.common.options.OptionsParser.newOptionsParser;

import junit.framework.TestCase;

import java.util.List;

/**
 * A test for {@link EnumConverter}.
 */
public class EnumConverterTest extends TestCase {

  private enum CompilationMode {
    DBG, OPT
  }

  private static class CompilationModeConverter
    extends EnumConverter<CompilationMode> {

    public CompilationModeConverter() {
      super(CompilationMode.class, "compilation mode");
    }
  }

  public void testConverterForEnumWithTwoValues() throws Exception {
    CompilationModeConverter converter = new CompilationModeConverter();
    assertEquals(converter.convert("dbg"), CompilationMode.DBG);
    assertEquals(converter.convert("opt"), CompilationMode.OPT);
    try {
      converter.convert("none");
      fail();
    } catch(OptionsParsingException e) {
      assertEquals(e.getMessage(),
                   "Not a valid compilation mode: 'none' (should be dbg or opt)");
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

  public void testTypeDescriptionForEnumWithThreeValues() throws Exception {
    FruitConverter converter = new FruitConverter();
    // We always use lowercase in the user-visible messages:
    assertEquals("apple, banana or cherry",
                 converter.getTypeDescription());
  }

  public void testConverterIsCaseInsensitive() throws Exception {
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

  public void testEnumList() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(EnumListTestOptions.class);
    parser.parse("--goo=ALPHA", "--goo=BRAVO");
    EnumListTestOptions options = parser.getOptions(EnumListTestOptions.class);
    assertNotNull(options.goo);
    assertEquals(2, options.goo.size());
    assertEquals(AlphabetEnum.ALPHA, options.goo.get(0));
    assertEquals(AlphabetEnum.BRAVO, options.goo.get(1));
  }

}
