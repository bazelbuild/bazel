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

import java.lang.reflect.Field;
import java.lang.reflect.Type;

/**
 * Knowledge about how the option parser should get type information about options and set their
 * values that are defined by fields.
 */
public class FieldOptionDefinition extends OptionDefinition {

  /**
   * If the {@code field} is annotated with the appropriate @{@link Option} annotation, returns the
   * {@code OptionDefinition} for that option. Otherwise, throws a {@link NotAnOptionException}.
   *
   * <p>These values are cached in the {@link OptionsData} layer and should be accessed through
   * {@link OptionsParser#getOptionDefinitions(Class)}.
   */
  static FieldOptionDefinition extractOptionDefinition(Field field) {
    Option annotation = field == null ? null : field.getAnnotation(Option.class);
    if (annotation == null) {
      throw new NotAnOptionException(field);
    }
    return new FieldOptionDefinition(field, annotation);
  }

  private final Field field;

  private FieldOptionDefinition(Field field, Option optionAnnotation) {
    super(optionAnnotation);
    this.field = field;
    this.field.setAccessible(true);
  }

  /** Returns the underlying {@code field} for this {@code OptionDefinition}. */
  protected Field getField() {
    return field;
  }

  @Override
  public <C extends OptionsBase> Class<? extends C> getDeclaringClass(Class<C> baseClass) {
    Class<?> declaringClass = field.getDeclaringClass();
    if (!baseClass.isAssignableFrom(declaringClass)) {
      throw new IllegalStateException(
          String.format(
              "Declaring class %s is not assignable from requested base class %s",
              declaringClass, baseClass));
    }
    @SuppressWarnings("unchecked") // This should be safe based on the previous check.
    Class<? extends C> castClass = (Class<? extends C>) declaringClass;
    return castClass;
  }

  @Override
  public Object getRawValue(OptionsBase optionsBase) {
    try {
      return field.get(optionsBase);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
          String.format(
              "Unexpected illegal access trying to fetch value for field %s in options %s: ",
              this.getOptionName(), optionsBase),
          e);
    }
  }

  @Override
  public void setValue(OptionsBase optionsBase, Object value) {
    try {
      field.set(optionsBase, value);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Couldn't set " + this.getOptionName(), e);
    }
  }

  @Override
  public boolean isDeprecated() {
    return field.isAnnotationPresent(Deprecated.class);
  }

  @Override
  public Class<?> getType() {
    return field.getType();
  }

  @Override
  protected Type getSingularType() {
    return field.getGenericType();
  }

  /**
   * {@link FieldOptionDefinition} is really a wrapper around a {@link Field} that caches
   * information obtained through reflection. Checking that the fields they represent are equal is
   * sufficient to check that two {@link FieldOptionDefinition} objects are equal.
   */
  @Override
  public boolean equals(Object object) {
    if (!(object instanceof FieldOptionDefinition)) {
      return false;
    }
    FieldOptionDefinition otherOption = (FieldOptionDefinition) object;
    return field.equals(otherOption.field);
  }

  @Override
  public int hashCode() {
    return field.hashCode();
  }

  @Override
  public String getMemberName() {
    return field.getName();
  }

  @Override
  public String toString() {
    return String.format("option '--%s'", getOptionName());
  }
}
