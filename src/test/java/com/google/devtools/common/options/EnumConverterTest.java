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
    assertThat(converter.convert("dbg")).isEqualTo(CompilationMode.DBG);
    assertThat(converter.convert("opt")).isEqualTo(CompilationMode.OPT);
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("none"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Not a valid compilation mode: 'none' (should be dbg or opt)");
    assertThat(converter.getTypeDescription()).isEqualTo("dbg or opt");
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
    assertThat(converter.getTypeDescription()).isEqualTo("apple, banana or cherry");
  }

  @Test
  public void converterIsCaseInsensitive() throws Exception {
    FruitConverter converter = new FruitConverter();
    assertThat(converter.convert("bAnANa")).isSameInstanceAs(Fruit.Banana);
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
    @Option(
      name = "goo",
      allowMultiple = true,
      converter = AlphabetEnumConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public List<AlphabetEnum> goo;
  }

  @Test
  public void enumList() throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(EnumListTestOptions.class).build();
    parser.parse("--goo=ALPHA", "--goo=BRAVO");
    EnumListTestOptions options = parser.getOptions(EnumListTestOptions.class);
    assertThat(options.goo).isNotNull();
    assertThat(options.goo).hasSize(2);
    assertThat(options.goo.get(0)).isEqualTo(AlphabetEnum.ALPHA);
    assertThat(options.goo.get(1)).isEqualTo(AlphabetEnum.BRAVO);
  }

}
