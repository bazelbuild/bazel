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
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the OptionsTester. */
@RunWith(JUnit4.class)
public final class OptionsTesterTest {

  @Test
  public void optionAnnotationCheck_PassesWhenAllFieldsAnnotated() throws Exception {
    new OptionsTester(OptionAnnotationCheckAllFieldsAnnotated.class)
        .testAllInstanceFieldsAnnotatedWithOption();
  }

  /** Test options class for optionAnnotationCheck_PassesWhenAllFieldsAnnotated. */
  public static class BaseAllFieldsAnnotated extends OptionsBase {
    @Option(
        name = "public_inherited_field_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public String inheritedField;
  }

  /** Test options class for optionAnnotationCheck_PassesWhenAllFieldsAnnotated. */
  public static final class OptionAnnotationCheckAllFieldsAnnotated extends BaseAllFieldsAnnotated {
    @Option(
        name = "public_declared_field_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public String publicField;

    @Option(
        name = "other_public_declared_field_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public String publicField2;
  }

  @Test
  public void optionAnnotationCheck_AllowsUnAnnotatedStaticFields() throws Exception {
    new OptionsTester(OptionAnnotationCheckUnAnnotatedStaticField.class)
        .testAllInstanceFieldsAnnotatedWithOption();
  }

  /** Test options class for optionAnnotationCheck_AllowsUnAnnotatedStaticFields. */
  public static class BaseUnAnnotatedStaticField extends OptionsBase {
    public static String parentClassUnAnnotatedStaticField;
  }

  /** Test options class for optionAnnotationCheck_AllowsUnAnnotatedStaticFields. */
  public static final class OptionAnnotationCheckUnAnnotatedStaticField
      extends BaseUnAnnotatedStaticField {
    @SuppressWarnings("unused")
    private static String privateDeclaredUnAnnotatedStaticField;
  }

  @Test
  public void optionAnnotationCheck_FailsWhenFieldNotAnnotated() throws Exception {
    try {
      new OptionsTester(OptionAnnotationCheckUnAnnotatedField.class)
          .testAllInstanceFieldsAnnotatedWithOption();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("unAnnotatedField");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for optionAnnotationCheck_FailsWhenFieldNotAnnotated. */
  public static class OptionAnnotationCheckUnAnnotatedField extends OptionsBase {
    public String unAnnotatedField;
  }

  @Test
  public void optionAnnotationCheck_FailsForUnAnnotatedFieldsInSuperclass() throws Exception {
    try {
      new OptionsTester(OptionAnnotationCheckInheritedUnAnnotatedField.class)
          .testAllInstanceFieldsAnnotatedWithOption();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("unAnnotatedField");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for optionAnnotationCheck_FailsForUnAnnotatedFieldsInSuperclass. */
  public static final class OptionAnnotationCheckInheritedUnAnnotatedField
      extends OptionAnnotationCheckUnAnnotatedField {}

  @Test
  public void optionAnnotationCheck_FailsForPrivateUnAnnotatedField() throws Exception {
    try {
      new OptionsTester(OptionAnnotationCheckPrivateUnAnnotatedField.class)
          .testAllInstanceFieldsAnnotatedWithOption();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("privateUnAnnotatedField");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for optionAnnotationCheck_FailsForPrivateUnAnnotatedField. */
  public static final class OptionAnnotationCheckPrivateUnAnnotatedField extends OptionsBase {
    @SuppressWarnings("unused")
    private String privateUnAnnotatedField;
  }

  /** Test options class for optionAccessCheck_PassesWhenAllFieldsPublicNotStaticNotFinal. */
  public static class BaseAllFieldsPublicNotStaticNotFinal extends OptionsBase {
    @Option(
        name = "public_inherited_field_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public String inheritedField;
  }

  /** Test options class for optionAccessCheck_PassesWhenAllFieldsPublicNotStaticNotFinal. */
  public static final class OptionAccessCheckAllFieldsPublicNotStaticNotFinal
      extends BaseAllFieldsPublicNotStaticNotFinal {
    @Option(
        name = "public_declared_field_with_annotation",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defaultFoo")
    public String annotatedField;
  }

  /** Test options class for optionAccessCheck_IgnoresNonAnnotatedFields. */
  public static class BaseNonAnnotatedFields extends OptionsBase {
    protected static String parentClassUnAnnotatedStaticField;
    final String parentClassUnAnnotatedFinalField = "";
  }

  /** Test options class for optionAccessCheck_IgnoresNonAnnotatedFields. */
  public static final class OptionAccessCheckNonAnnotatedFields extends BaseNonAnnotatedFields {
    @SuppressWarnings("unused")
    private static String privateDeclaredUnAnnotatedStaticFinalField;

    public static final String PUBLIC_DECLARED_UN_ANNOTATED_STATIC_FIELD = "";
    String protectedDeclaredField;
  }

  /** Test converter class for testing testAllDefaultValuesTestedBy. */
  public static final class TestConverter implements Converter<String> {
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
                    new ConverterTester(TestConverter.class)
                        .addEqualityGroup("testedDefault", "otherTestedDefault"))
                .build());
  }

  /** Test options class for defaultTestCheck_PassesIfAllDefaultsTested. */
  public static final class DefaultTestCheck extends OptionsBase {
    @Option(
        name = "tested_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "testedDefault")
    public String testedField;

    @Option(
        name = "field_implicitly_using_default_converter",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "implicitConverterDefault")
    public String implicitConverterField;

    @Option(
        name = "other_tested_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "otherTestedDefault")
    public String otherTestedField;

    @Option(
        name = "field_with_null_default",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null")
    public String nullDefaultField;

    @Option(
        name = "allowMultiple_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null",
        allowMultiple = true)
    public List<String> allowMultipleField;
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsPresentButValueIsNotTested() {
    try {
      new OptionsTester(DefaultTestCheckUntestedField.class)
          .testAllDefaultValuesTestedBy(
              new ConverterTesterMap.Builder()
                  .add(new ConverterTester(TestConverter.class).addEqualityGroup("testedDefault"))
                  .build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("untestedField");
      assertThat(expected).hasMessageThat().contains("untestedDefault");
      return;
    }
    fail("test is expected to have failed");
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsent() {
    try {
      new OptionsTester(DefaultTestCheckUntestedField.class)
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
  public static final class DefaultTestCheckUntestedField extends OptionsBase {
    @Option(
        name = "untested_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "untestedDefault")
    public String untestedField;
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsentEvenForNullDefault() {
    try {
      new OptionsTester(DefaultTestCheckUntestedNullField.class)
          .testAllDefaultValuesTestedBy(new ConverterTesterMap.Builder().build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("TestConverter");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for defaultTestCheck_FailsIfTesterIsAbsentEvenForNullDefault. */
  public static final class DefaultTestCheckUntestedNullField extends OptionsBase {
    @Option(
        name = "untested_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null")
    public String untestedNullField;
  }

  @Test
  public void defaultTestCheck_FailsIfTesterIsAbsentEvenForAllowMultiple() {
    try {
      new OptionsTester(DefaultTestCheckUntestedMultipleField.class)
          .testAllDefaultValuesTestedBy(new ConverterTesterMap.Builder().build());
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("TestConverter");
      return;
    }
    fail("test is expected to have failed");
  }

  /** Test options class for defaultTestCheck_FailsIfTesterIsAbsentEvenForAllowMultiple. */
  public static final class DefaultTestCheckUntestedMultipleField extends OptionsBase {
    @Option(
        name = "untested_field",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = TestConverter.class,
        defaultValue = "null",
        allowMultiple = true)
    public List<String> untestedMultipleField;
  }
}
