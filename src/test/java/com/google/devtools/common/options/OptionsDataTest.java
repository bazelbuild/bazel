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
import java.util.List;
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
      Class<? extends OptionsBase> optionsClass1, Class<? extends OptionsBase> optionsClass2)
      throws OptionsParser.ConstructionException {
    return IsolatedOptionsData.from(
        ImmutableList.<Class<? extends OptionsBase>>of(optionsClass1, optionsClass2));
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleNameConflictOptions extends OptionsBase {
    @Option(
      name = "foo",
      defaultValue = "1"
    )
    public int foo;

    @Option(
      name = "foo",
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
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to option: --foo");
    }
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleIntegerFooOptions extends OptionsBase {
    @Option(
      name = "foo",
      defaultValue = "5"
    )
    public int foo;
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleBooleanFooOptions extends OptionsBase {
    @Option(
      name = "foo",
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
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to option: --foo");
    }
  }

  /** Dummy comment (linter suppression) */
  public static class ExamplePrefixFooOptions extends OptionsBase {
    @Option(
      name = "nofoo",
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
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to option --nofoo, it "
          + "conflicts with a negating alias for boolean flag --foo");
    }

    try {
      construct(ExamplePrefixFooOptions.class, ExampleBooleanFooOptions.class);
      fail("nofoo should conflict with the previous flag foo, "
         + "since foo, as a boolean flag, can be written as --nofoo");
    } catch (DuplicateOptionDeclarationException e) {
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to boolean option alias: --nofoo");
    }
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleBarWasNamedFooOption extends OptionsBase {
    @Option(
      name = "bar",
      oldName = "foo",
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
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to boolean option alias: --nofoo");
    }
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleBarWasNamedNoFooOption extends OptionsBase {
    @Option(
      name = "bar",
      oldName = "nofoo",
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
      assertThat(e.getMessage()).contains(
          "Duplicate option name, due to old option name --nofoo, it conflicts with a "
          + "negating alias for boolean flag --foo");
    }
  }

  /** Dummy comment (linter suppression) */
  public static class OldNameConflictExample extends OptionsBase {
    @Option(
      name = "new_name",
      oldName = "old_name",
      defaultValue = "defaultValue"
    )
    public String flag1;

    @Option(
      name = "old_name",
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

  /** Dummy comment (linter suppression) */
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

  /** Dummy comment (linter suppression) */
  public static class InvalidOptionConverter extends OptionsBase {
    @Option(
      name = "foo",
      converter = StringConverter.class,
      defaultValue = "1"
    )
    public Integer foo;
  }

  @Test
  public void errorForInvalidOptionConverter() throws Exception {
    try {
      construct(InvalidOptionConverter.class);
    } catch (AssertionError e) {
      // Expected exception
      return;
    }
    fail();
  }

  /** Dummy comment (linter suppression) */
  public static class InvalidListOptionConverter extends OptionsBase {
    @Option(
      name = "foo",
      converter = StringConverter.class,
      defaultValue = "1",
      allowMultiple = true
    )
    public List<Integer> foo;
  }

  @Test
  public void errorForInvalidListOptionConverter() throws Exception {
    try {
      construct(InvalidListOptionConverter.class);
    } catch (AssertionError e) {
      // Expected exception
      return;
    }
    fail();
  }
}
