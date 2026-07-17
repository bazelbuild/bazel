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
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.lang.reflect.AccessibleObject;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
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

  private static ImmutableList<Method> getAllMethods(Class<? extends OptionsBase> optionsClass) {
    ImmutableList.Builder<Method> builder = ImmutableList.builder();
    Class<? extends OptionsBase> current = optionsClass;
    while (!OptionsBase.class.equals(current)) {
      builder.add(current.getDeclaredMethods());
      Class<? extends OptionsBase> superclass =
          current.getSuperclass().asSubclass(OptionsBase.class);
      current = superclass;
    }
    return builder.build();
  }

  /**
   * Tests that there are no non-Option instance fields. Fields not annotated with @Option will not
   * be considered for equality.
   */
  @CanIgnoreReturnValue
  public OptionsTester testAllOptions() {
    for (Field field : getAllFields(optionsClass)) {
      if (!Modifier.isStatic(field.getModifiers())) {
        assertWithMessage(
                "%s is missing an @Option annotation; it will not be considered for equality.",
                field)
            .that(field.getAnnotation(Option.class))
            .isNotNull();
      }
    }
    for (Method method : getAllMethods(optionsClass)) {
      if (Modifier.isAbstract(method.getModifiers()) && method.getName().startsWith("get")) {
        assertWithMessage(
                "%s is missing an @Option annotation; it will not be considered for equality.",
                method)
            .that(method.getAnnotation(Option.class))
            .isNotNull();
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
  @CanIgnoreReturnValue
  public OptionsTester testAllDefaultValuesTestedBy(ConverterTesterMap testers) {
    ImmutableListMultimap.Builder<Class<? extends Converter<?>>, AccessibleObject>
        converterClassesBuilder = ImmutableListMultimap.builder();
    for (Field field : getAllFields(optionsClass)) {
      Option option = field.getAnnotation(Option.class);
      if (option != null && !Converter.class.equals(option.converter())) {
        @SuppressWarnings("unchecked") // converter is rawtyped; see comment on Option.converter()
        Class<? extends Converter<?>> converter =
            (Class<? extends Converter<?>>) option.converter();
        converterClassesBuilder.put(converter, field);
      }
    }
    for (Method method : getAllMethods(optionsClass)) {
      Option option = method.getAnnotation(Option.class);
      if (option != null && !Converter.class.equals(option.converter())) {
        @SuppressWarnings("unchecked") // converter is rawtyped; see comment on Option.converter()
        Class<? extends Converter<?>> converter =
            (Class<? extends Converter<?>>) option.converter();
        converterClassesBuilder.put(converter, method);
      }
    }
    ImmutableListMultimap<Class<? extends Converter<?>>, AccessibleObject> converterClasses =
        converterClassesBuilder.build();
    for (Class<? extends Converter<?>> converter : converterClasses.keySet()) {
      assertWithMessage(
              "Converter %s has no corresponding ConverterTester", converter.getCanonicalName())
          .that(testers)
          .containsKey(converter);
      for (AccessibleObject member : converterClasses.get(converter)) {
        Option option = member.getAnnotation(Option.class);
        if (option != null && !option.allowMultiple() && !option.defaultValue().equals("null")) {
          assertWithMessage(
                  "Default value \"%s\" on %s is not tested in the corresponding ConverterTester"
                      + " for %s",
                  option.defaultValue(), member, converter.getCanonicalName())
              .that(testers.get(converter).hasTestForInput(option.defaultValue()))
              .isTrue();
        }
      }
    }
    return this;
  }
}
