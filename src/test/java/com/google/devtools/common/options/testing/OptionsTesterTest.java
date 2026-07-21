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

package com.google.devtools.common.options.testing;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the OptionsTester. */
@RunWith(JUnit4.class)
public final class OptionsTesterTest {

  @Test
  public void optionAnnotationCheck_PassesWhenAllOptionsAnnotated() throws Exception {
    new OptionsTester(OptionAnnotationCheckAllOptionsAnnotated.class).testAllOptions();
  }

  /** Test options class for optionAnnotationCheck_PassesWhenAllOptionsAnnotated. */
  @OptionsClass
  public abstract static class BaseAllOptionsAnnotated extends OptionsBase {
    @Option(
        name = "public_inherited_option_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public abstract String getInheritedOptione();
  }

  /** Test options class for optionAnnotationCheck_PassesWhenAllOptionsAnnotated. */
  @OptionsClass
  public abstract static class OptionAnnotationCheckAllOptionsAnnotated
      extends BaseAllOptionsAnnotated {
    @Option(
        name = "public_declared_option_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public abstract String getPublicOption();

    @Option(
        name = "other_public_declared_option_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public abstract String getPublicOption2();
  }

  public static final class TestConverter extends Converter.Contextless<String> {
    @Override
    public String convert(String input) {
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "a string";
    }
  }

  @Test
  public void defaultTestCheck_PassesIfAllDefaultsTestedIgnoringNullAndAllowMultiple() {
    new OptionsTester(DefaultTestCheck.class)
        .testAllDefaultValuesTestedBy(
            new ConverterTesterMap.Builder()
                .add(
                    new ConverterTester(TestConverter.class, /*conversionContext=*/ null)
                        .addEqualityGroup("testedDefault", "otherTestedDefault"))
                .build());
  }

  /** Test options class for defaultTestCheck_PassesIfAllDefaultsTested. */
  @OptionsClass
  public abstract static class DefaultTestCheck extends OptionsBase {
    @Option(
        name = "tested_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "testedDefault")
    public abstract String getTestedOption();

    @Option(
        name = "option_implicitly_using_default_converter",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "implicitConverterDefault")
    public abstract String getImplicitConverterOption();

    @Option(
        name = "other_tested_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "otherTestedDefault")
    public abstract String getOtherTestedOption();

    @Option(
        name = "option_with_null_default",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null")
    public abstract String getNullDefaultOption();

    @Option(
        name = "allowMultiple_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null",
        allowMultiple = true)
    public abstract List<String> getAllowMultipleOption();
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsPresentButValueIsNotTested() {
    try {
      new OptionsTester(DefaultTestCheckUntestedOption.class)
          .testAllDefaultValuesTestedBy(
              new ConverterTesterMap.Builder()
                  .add(
                      new ConverterTester(TestConverter.class, /* conversionContext= */ null)
                          .addEqualityGroup("testedDefault"))
                  .build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("getUntestedOption");
      assertThat(expected).hasMessageThat().contains("untestedDefault");
      return;
    }
    fail("test is expected to have failed");
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsent() {
    try {
      new OptionsTester(DefaultTestCheckUntestedOption.class)
          .testAllDefaultValuesTestedBy(new ConverterTesterMap.Builder().build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("TestConverter");
      return;
    }
    fail("test is expected to have failed");
  }

  /**
   * Test options class for defaultTestCheck_FailsIfTesterIsPresentButValueIsNotTested and
   * defaultTestCheck_FailsIfTesterIsAbsent.
   */
  @OptionsClass
  public abstract static class DefaultTestCheckUntestedOption extends OptionsBase {
    @Option(
        name = "untested_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "untestedDefault")
    public abstract String getUntestedOption();
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsentEvenForNullDefault() {
    try {
      new OptionsTester(DefaultTestCheckUntestedNullOption.class)
          .testAllDefaultValuesTestedBy(new ConverterTesterMap.Builder().build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("TestConverter");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for defaultTestCheck_FailsIfTesterIsAbsentEvenForNullDefault. */
  @OptionsClass
  public abstract static class DefaultTestCheckUntestedNullOption extends OptionsBase {
    @Option(
        name = "untested_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null")
    public abstract String getUntestedNullOption();
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsentEvenForAllowMultiple() {
    try {
      new OptionsTester(DefaultTestCheckUntestedMultipleOption.class)
          .testAllDefaultValuesTestedBy(new ConverterTesterMap.Builder().build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("TestConverter");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for defaultTestCheck_FailsIfTesterIsAbsentEvenForAllowMultiple. */
  @OptionsClass
  public abstract static class DefaultTestCheckUntestedMultipleOption extends OptionsBase {
    @Option(
        name = "untested_option",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null",
        allowMultiple = true)
    public abstract List<String> getUntestedMultipleOption();
  }
}
