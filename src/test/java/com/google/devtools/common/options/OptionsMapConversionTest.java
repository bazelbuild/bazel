// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Field;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for converting {@link OptionsBase} subclass instances to and from maps. */
@RunWith(JUnit4.class)
public class OptionsMapConversionTest {

  private static Map<String, Object> keysToStrings(Map<Field, Object> map) {
    Map<String, Object> result = new LinkedHashMap<>();
    for (Map.Entry<Field, Object> entry : map.entrySet()) {
      OptionDefinition optionDefinition = OptionDefinition.extractOptionDefinition(entry.getKey());
      result.put(optionDefinition.getOptionName(), entry.getValue());
    }
    return result;
  }

  private static Map<Field, Object> keysToFields(
      Class<? extends OptionsBase> optionsClass, Map<String, Object> map) {
    OptionsData data = OptionsParser.getOptionsDataInternal(optionsClass);
    Map<Field, Object> result = new LinkedHashMap<>();
    for (Map.Entry<String, Object> entry : map.entrySet()) {
      OptionDefinition optionDefinition = data.getOptionDefinitionFromName(entry.getKey());
      result.put(optionDefinition.getField(), entry.getValue());
    }
    return result;
  }

  /** Dummy options base class. */
  public static class FooOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo;
  }

  /** Dummy options derived class. */
  public static class BazOptions extends FooOptions {
    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "true"
    )
    public boolean bar;

    @Option(
      name = "baz",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "5"
    )
    public int baz;
  }

  @Test
  public void toMap_Basic() {
    FooOptions foo = Options.getDefaults(FooOptions.class);
    assertThat(keysToStrings(OptionsParser.toMap(FooOptions.class, foo)))
        .containsExactly("foo", false);
  }

  @Test
  public void asMap_Basic() {
    FooOptions foo = Options.getDefaults(FooOptions.class);
    assertThat(foo.asMap())
        .containsExactly("foo", false);
  }

  @Test
  public void toMap_Inheritance() {
    BazOptions baz = Options.getDefaults(BazOptions.class);
    assertThat(keysToStrings(OptionsParser.toMap(BazOptions.class, baz)))
        .containsExactly("foo", false, "bar", true, "baz", 5);
  }

  @Test
  public void asMap_Inheritance() {
    // Static type is base class, dynamic type is derived. We still get the derived fields.
    FooOptions foo = Options.getDefaults(BazOptions.class);
    assertThat(foo.asMap())
        .containsExactly("foo", false, "bar", true, "baz", 5);
  }

  @Test
  public void toMap_InheritanceBaseFieldsOnly() {
    BazOptions baz = Options.getDefaults(BazOptions.class);
    assertThat(keysToStrings(OptionsParser.toMap(FooOptions.class, baz)))
        .containsExactly("foo", false);
  }

  /**
   * Dummy options class for checking alphabetizing.
   *
   * <p>Note that field name order differs from option name order.
   */
  public static class AlphaOptions extends OptionsBase {

    @Option(
      name = "c",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int v;

    @Option(
      name = "d",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int w;

    @Option(
      name = "a",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int x;

    @Option(
      name = "e",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int y;

    @Option(
      name = "b",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int z;
  }

  @Test
  public void toMap_AlphabeticalOrder() {
    AlphaOptions alpha = Options.getDefaults(AlphaOptions.class);
    assertThat(keysToStrings(OptionsParser.toMap(AlphaOptions.class, alpha)))
        .containsExactly("a", 0, "b", 0, "c", 0, "d", 0, "e", 0).inOrder();
  }

  @Test
  public void asMap_AlphabeticalOrder() {
    AlphaOptions alpha = Options.getDefaults(AlphaOptions.class);
    assertThat(alpha.asMap())
        .containsExactly("a", 0, "b", 0, "c", 0, "d", 0, "e", 0).inOrder();
  }

  @Test
  public void fromMap_Basic() {
    Map<String, Object> map = ImmutableMap.<String, Object>of("foo", true);
    Map<Field, Object> fieldMap = keysToFields(FooOptions.class, map);
    FooOptions foo = OptionsParser.fromMap(FooOptions.class, fieldMap);
    assertThat(foo.foo).isTrue();
  }

  /** Dummy subclass of foo. */
  public static class SubFooAOptions extends FooOptions {
    @Option(
      name = "a",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean a;
  }

  /** Dummy subclass of foo. */
  public static class SubFooBOptions extends FooOptions {
    @Option(
      name = "b1",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean b1;

    @Option(
      name = "b2",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean b2;
  }

  @Test
  public void fromMap_FailsOnWrongKeys() {
    Map<String, Object> map = ImmutableMap.<String, Object>of("foo", true, "a", false);
    Map<Field, Object> fieldMap = keysToFields(SubFooAOptions.class, map);
    try {
      OptionsParser.fromMap(SubFooBOptions.class, fieldMap);
      fail("Should have failed due to the given map's fields not matching the ones on the class");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Map keys do not match fields of options class; extra map keys: {'a'}; "
                  + "extra options class options: {'b1', 'b2'}");
    }
  }

  @Test
  public void fromMap_FailsOnWrongTypes() {
    Map<String, Object> map = ImmutableMap.<String, Object>of("foo", 5);
    Map<Field, Object> fieldMap = keysToFields(SubFooAOptions.class, map);
    try {
      OptionsParser.fromMap(FooOptions.class, fieldMap);
      fail("Should have failed due to trying to assign a field value with the wrong type");
    } catch (IllegalArgumentException e) {
      assertThat(e)
          .hasMessageThat()
          .matches("Can not set boolean field .*\\.foo to java\\.lang\\.Integer");
    }
  }

  @Test
  public void fromMap_Inheritance() {
    Map<String, Object> map = ImmutableMap.<String, Object>of("foo", true, "bar", true, "baz", 3);
    Map<Field, Object> fieldMap = keysToFields(BazOptions.class, map);
    BazOptions baz = OptionsParser.fromMap(BazOptions.class, fieldMap);
    assertThat(baz.foo).isTrue();
    assertThat(baz.bar).isTrue();
    assertThat(baz.baz).isEqualTo(3);
  }
}
