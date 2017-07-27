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

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * A tester to validate certain useful properties of OptionsBase subclasses. These are not required
 * for parsing options in these classes, but can be helpful for e.g. ensuring that equality is not
 * violated.
 */
public final class OptionsTester {

  private final Class<? extends OptionsBase> optionsClass;

  public OptionsTester(Class<? extends OptionsBase> optionsClass) {
    this.optionsClass = optionsClass;
  }

  private static ImmutableList<Field> getAllFields(Class<? extends OptionsBase> optionsClass) {
    ImmutableList.Builder<Field> builder = ImmutableList.builder();
    Class<? extends OptionsBase> current = optionsClass;
    while (!OptionsBase.class.equals(current)) {
      builder.add(current.getDeclaredFields());
      // the input extends OptionsBase and we haven't seen OptionsBase yet, so this must also extend
      // (or be) OptionsBase
      @SuppressWarnings("unchecked")
      Class<? extends OptionsBase> superclass =
          (Class<? extends OptionsBase>) current.getSuperclass();
      current = superclass;
    }
    return builder.build();
  }

  /**
   * Tests that there are no non-Option instance fields. Fields not annotated with @Option will not
   * be considered for equality.
   */
  public OptionsTester testAllInstanceFieldsAnnotatedWithOption() {
    for (Field field : getAllFields(optionsClass)) {
      if (!Modifier.isStatic(field.getModifiers())) {
        assertWithMessage(
                field
                    + " is missing an @Option annotation; it will not be considered for equality.")
            .that(field.getAnnotation(Option.class))
            .isNotNull();
      }
    }
    return this;
  }

  /** Tests that there are no non-public fields which would interfere with option parsing. */
  public OptionsTester testAllOptionFieldsPublic() {
    for (Field field : getAllFields(optionsClass)) {
      if (field.isAnnotationPresent(Option.class)) {
        assertWithMessage(
                field
                    + " is Option-annotated, but is not public; it will not be considered as part"
                    + " of the options. Change the visibility to public.")
            .that(Modifier.isPublic(field.getModifiers()))
            .isTrue();
      }
      if (Modifier.isStatic(field.getModifiers()) || Modifier.isFinal(field.getModifiers())) {
        assertWithMessage(
                field
                    + " is Option-annotated, but is either static or final; it cannot be properly"
                    + " set by the option parser. Remove either the annotation or the modifier(s).")
            .that(field.getAnnotation(Option.class))
            .isNull();
      }
    }
    return this;
  }

  /**
   * Tests that the default values of this class were part of the test data for the appropriate
   * ConverterTester, ensuring that the defaults at least obey proper equality semantics.
   *
   * <p>The default converters are not tested in this way.
   *
   * <p>Note that testConvert is not actually run on the ConverterTesters; it is expected that they
   * are run elsewhere.
   */
  public OptionsTester testAllDefaultValuesTestedBy(ConverterTesterMap testers) {
    ImmutableListMultimap.Builder<Class<? extends Converter<?>>, Field> converterClassesBuilder =
        ImmutableListMultimap.builder();
    for (Field field : getAllFields(optionsClass)) {
      Option option = field.getAnnotation(Option.class);
      if (option != null && !Converter.class.equals(option.converter())) {
        @SuppressWarnings("unchecked") // converter is rawtyped; see comment on Option.converter()
        Class<? extends Converter<?>> converter =
            (Class<? extends Converter<?>>) option.converter();
        converterClassesBuilder.put(converter, field);
      }
    }
    ImmutableListMultimap<Class<? extends Converter<?>>, Field> converterClasses =
        converterClassesBuilder.build();
    for (Class<? extends Converter<?>> converter : converterClasses.keySet()) {
      assertWithMessage(
              "Converter " + converter.getCanonicalName() + " has no corresponding ConverterTester")
          .that(testers)
          .containsKey(converter);
      for (Field field : converterClasses.get(converter)) {
        Option option = field.getAnnotation(Option.class);
        if (!option.allowMultiple() && !"null".equals(option.defaultValue())) {
          assertWithMessage(
                  "Default value \""
                      + option.defaultValue()
                      + "\" on "
                      + field
                      + " is not tested in the corresponding ConverterTester for "
                      + converter.getCanonicalName())
              .that(testers.get(converter).hasTestForInput(option.defaultValue()))
              .isTrue();
        }
      }
    }
    return this;
  }
}
