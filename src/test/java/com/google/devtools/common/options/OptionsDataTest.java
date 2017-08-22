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

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IsolatedOptionsData} and {@link OptionsData}. */
@RunWith(JUnit4.class)
public class OptionsDataTest {

  private static IsolatedOptionsData construct(Class<? extends OptionsBase> optionsClass)
      throws OptionsParser.ConstructionException {
    return IsolatedOptionsData.from(ImmutableList.<Class<? extends OptionsBase>>of(optionsClass));
  }

  private static IsolatedOptionsData construct(
      Class<? extends OptionsBase> optionsClass1,
      Class<? extends OptionsBase> optionsClass2)
      throws OptionsParser.ConstructionException {
    return IsolatedOptionsData.from(
        ImmutableList.<Class<? extends OptionsBase>>of(optionsClass1, optionsClass2));
  }

  private static IsolatedOptionsData construct(
      Class<? extends OptionsBase> optionsClass1,
      Class<? extends OptionsBase> optionsClass2,
      Class<? extends OptionsBase> optionsClass3)
      throws OptionsParser.ConstructionException {
    return IsolatedOptionsData.from(
        ImmutableList.<Class<? extends OptionsBase>>of(
            optionsClass1, optionsClass2, optionsClass3));
  }

  /** Dummy options class. */
  public static class ExampleNameConflictOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1"
    )
    public int foo;

    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "I should conflict with foo"
    )
    public String anotherFoo;
  }

  @Test
  public void testNameConflictInSingleClass() {
    try {
      construct(ExampleNameConflictOptions.class);
      fail("foo should conflict with the previous flag foo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e).hasMessageThat().contains("Duplicate option name, due to option: --foo");
    }
  }

  /** Dummy options class. */
  public static class ExampleIntegerFooOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "5"
    )
    public int foo;
  }

  /** Dummy options class. */
  public static class ExampleBooleanFooOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo;
  }

  @Test
  public void testNameConflictInTwoClasses() {
    try {
      construct(ExampleIntegerFooOptions.class, ExampleBooleanFooOptions.class);
      fail("foo should conflict with the previous flag foo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e).hasMessageThat().contains("Duplicate option name, due to option: --foo");
    }
  }

  /** Dummy options class. */
  public static class ExamplePrefixFooOptions extends OptionsBase {
    @Option(
      name = "nofoo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean noFoo;
  }

  @Test
  public void testBooleanPrefixNameConflict() {
    // Try the same test in both orders, the parser should fail if the overlapping flag is defined
    // before or after the boolean flag introduces the alias.
    try {
      construct(ExampleBooleanFooOptions.class, ExamplePrefixFooOptions.class);
      fail("nofoo should conflict with the previous flag foo, "
         + "since foo, as a boolean flag, can be written as --nofoo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Duplicate option name, due to option --nofoo, it "
                  + "conflicts with a negating alias for boolean flag --foo");
    }

    try {
      construct(ExamplePrefixFooOptions.class, ExampleBooleanFooOptions.class);
      fail("nofoo should conflict with the previous flag foo, "
         + "since foo, as a boolean flag, can be written as --nofoo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Duplicate option name, due to boolean option alias: --nofoo");
    }
  }

  /** Dummy options class. */
  public static class ExampleBarWasNamedFooOption extends OptionsBase {
    @Option(
      name = "bar",
      oldName = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean bar;
  }

  @Test
  public void testBooleanAliasWithOldNameConflict() {
    // Try the same test in both orders, the parser should fail if the overlapping flag is defined
    // before or after the boolean flag introduces the alias.
    try {
      construct(ExamplePrefixFooOptions.class, ExampleBarWasNamedFooOption.class);
      fail("nofoo should conflict with the previous flag foo, "
         + "since foo, as a boolean flag, can be written as --nofoo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Duplicate option name, due to boolean option alias: --nofoo");
    }
  }

  /** Dummy options class. */
  public static class ExampleBarWasNamedNoFooOption extends OptionsBase {
    @Option(
      name = "bar",
      oldName = "nofoo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean bar;
  }

  @Test
  public void testBooleanWithOldNameAsAliasOfBooleanConflict() {
    // Try the same test in both orders, the parser should fail if the overlapping flag is defined
    // before or after the boolean flag introduces the alias.
    try {
      construct(ExampleBooleanFooOptions.class, ExampleBarWasNamedNoFooOption.class);
      fail("nofoo, the old name for bar, should conflict with the previous flag foo, "
         + "since foo, as a boolean flag, can be written as --nofoo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Duplicate option name, due to old option name --nofoo, it conflicts with a "
                  + "negating alias for boolean flag --foo");
    }
  }

  /** Dummy options class. */
  public static class OldNameConflictExample extends OptionsBase {
    @Option(
      name = "new_name",
      oldName = "old_name",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultValue"
    )
    public String flag1;

    @Option(
      name = "old_name",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultValue"
    )
    public String flag2;
  }

  @Test
  public void testOldNameConflict() {
    try {
      construct(OldNameConflictExample.class);
      fail("old_name should conflict with the flag already named old_name");
    } catch (DuplicateOptionDeclarationException expected) {
    }
  }

  /** Dummy options class. */
  public static class StringConverter implements Converter<String> {
    @Override
    public String convert(String input) {
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "a string";
    }
  }

  /** Dummy options class. */
  public static class InvalidOptionConverter extends OptionsBase {
    @Option(
      name = "foo",
      converter = StringConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1"
    )
    public Integer foo;
  }

  @Test
  public void errorForInvalidOptionConverter() throws Exception {
    try {
      construct(InvalidOptionConverter.class);
    } catch (ConstructionException e) {
      // Expected exception
      return;
    }
    fail();
  }

  /** Dummy options class. */
  public static class InvalidListOptionConverter extends OptionsBase {
    @Option(
      name = "foo",
      converter = StringConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1",
      allowMultiple = true
    )
    public List<Integer> foo;
  }

  @Test
  public void errorForInvalidListOptionConverter() throws Exception {
    try {
      construct(InvalidListOptionConverter.class);
    } catch (ConstructionException e) {
      // Expected exception
      return;
    }
    fail();
  }

  /** Dummy options class using deprecated category. */
  public static class InvalidUndocumentedCategory extends OptionsBase {
    @Option(
      name = "experimental_foo",
      category = "undocumented",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "true"
    )
    public boolean experimentalFoo;
  }

  @Test
  public void invalidUndocumentedCategoryFails() {
    try {
      construct(InvalidUndocumentedCategory.class);
      fail();
    } catch (ConstructionException e) {
      // Expected exception
      assertThat(e).hasMessageThat().contains(
          "Documentation level is no longer read from the option category.");
      assertThat(e).hasMessageThat().contains("undocumented");
      assertThat(e).hasMessageThat().contains("experimental_foo");
    }
  }

  /**
   * Dummy options class.
   *
   * <p>Option name order is different from field name order.
   * 
   * <p>There are four fields to increase the likelihood of a non-deterministic order being noticed.
   */
  public static class FieldNamesDifferOptions extends OptionsBase {

    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int aFoo;

    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int bBar;

    @Option(
      name = "baz",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int cBaz;

    @Option(
      name = "qux",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int dQux;
  }

  /** Dummy options class. */
  public static class EndOfAlphabetOptions extends OptionsBase {
    @Option(
      name = "X",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int x;

    @Option(
      name = "Y",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int y;
  }

  /** Dummy options class. */
  public static class ReverseOrderedOptions extends OptionsBase {
    @Option(
      name = "C",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int c;

    @Option(
      name = "B",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int b;

    @Option(
      name = "A",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0"
    )
    public int a;
  }

  @Test
  public void optionsClassesIsOrdered() throws Exception {
    IsolatedOptionsData data = construct(
        FieldNamesDifferOptions.class,
        EndOfAlphabetOptions.class,
        ReverseOrderedOptions.class);
    assertThat(data.getOptionsClasses()).containsExactly(
        FieldNamesDifferOptions.class,
        EndOfAlphabetOptions.class,
        ReverseOrderedOptions.class).inOrder();
  }

  @Test
  public void getAllNamedFieldsIsOrdered() throws Exception {
    IsolatedOptionsData data = construct(
        FieldNamesDifferOptions.class,
        EndOfAlphabetOptions.class,
        ReverseOrderedOptions.class);
    ArrayList<String> names = new ArrayList<>();
    for (Map.Entry<String, OptionDefinition> entry : data.getAllNamedFields()) {
      names.add(entry.getKey());
    }
    assertThat(names).containsExactly(
        "bar", "baz", "foo", "qux", "X", "Y", "A", "B", "C").inOrder();
  }

  private List<String> getOptionNames(Iterable<OptionDefinition> fields) {
    ArrayList<String> result = new ArrayList<>();
    for (OptionDefinition optionDefinition : fields) {
      result.add(optionDefinition.getOptionName());
    }
    return result;
  }

  @Test
  public void getFieldsForClassIsOrdered() throws Exception {
    IsolatedOptionsData data = construct(
        FieldNamesDifferOptions.class,
        EndOfAlphabetOptions.class,
        ReverseOrderedOptions.class);
    assertThat(getOptionNames(data.getOptionDefinitionsFromClass(FieldNamesDifferOptions.class)))
        .containsExactly("bar", "baz", "foo", "qux")
        .inOrder();
    assertThat(getOptionNames(data.getOptionDefinitionsFromClass(EndOfAlphabetOptions.class)))
        .containsExactly("X", "Y")
        .inOrder();
    assertThat(getOptionNames(data.getOptionDefinitionsFromClass(ReverseOrderedOptions.class)))
        .containsExactly("A", "B", "C")
        .inOrder();
  }

  /** Dummy options class. */
  public static class InvalidExpansionOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1"
    )
    public int foo;

    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1",
      expansion = {"--foo=42"}
    )
    public int bar;
  }

  @Test
  public void staticExpansionOptionsShouldNotHaveValues() {
    try {
      construct(InvalidExpansionOptions.class);
      fail();
    } catch (ConstructionException e) {
      // Expected exception
      assertThat(e).hasMessageThat().contains(
          "Option bar is an expansion flag with a static expansion, but does not have Void type.");
    }
  }

  /** Dummy options class. */
  public static class ValidExpansionOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1"
    )
    public int foo;

    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansion = {"--foo=42"}
    )
    public Void bar;
  }

  @Test
  public void staticExpansionOptionsCanBeVoidType() {
    construct(ValidExpansionOptions.class);
  }
}
