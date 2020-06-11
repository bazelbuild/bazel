/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.Arrays;
import java.util.Objects;
import org.junit.runners.model.FrameworkMethod;

/**
 * Represents a {@link ParameterValueSource}-annotated method with its parameter values (either
 * fully or partially) derivable from the associated value source annotation. The test runner will
 * parameterize a @Test method with N {@link ParameterValueSource} annotations into N {@link
 * ValueSourceAnnotatedMethod}s for testing, and provides value bindings to all {@link
 * FromParameterValueSource}-annotated parameters from a {@link ParameterValueSource} instance in
 * sequence.
 */
public final class ValueSourceAnnotatedMethod extends FrameworkMethod {

  /** One single bundle for parameter value resolution. */
  private final ParameterValueSource parameterValueSource;

  public static ValueSourceAnnotatedMethod create(
      Method method, ParameterValueSource parameterValueSource) {
    return new ValueSourceAnnotatedMethod(method, parameterValueSource);
  }

  private ValueSourceAnnotatedMethod(Method method, ParameterValueSource parameterValueSource) {
    super(method);
    this.parameterValueSource = parameterValueSource;
  }

  ImmutableMap<Parameter, Object> getResolvableParameters() {
    String[] providedParamValues = parameterValueSource.value();
    int i = 0;
    ImmutableMap.Builder<Parameter, Object> resolvableParameterValues = ImmutableMap.builder();
    for (Parameter parameter : getMethod().getParameters()) {
      if (parameter.isAnnotationPresent(FromParameterValueSource.class)) {
        resolvableParameterValues.put(
            parameter, parsePrimitive(providedParamValues[i++], parameter.getType()));
      }
    }
    return resolvableParameterValues.build();
  }

  @Override
  public String getName() {
    // Appends distinct suffixes to support test filters.
    return super.getName() + "_" + Arrays.toString(parameterValueSource.value());
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (obj instanceof ValueSourceAnnotatedMethod) {
      ValueSourceAnnotatedMethod that = (ValueSourceAnnotatedMethod) obj;
      return this.parameterValueSource.equals(that.parameterValueSource)
          && this.getMethod().equals(that.getMethod());
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(getMethod(), parameterValueSource);
  }

  @Override
  public String toString() {
    return super.toString() + "_" + Arrays.toString(parameterValueSource.value());
  }

  private static Object parsePrimitive(String text, Class<?> targetType) {
    if (targetType == String.class) {
      return text;
    }
    if (targetType == int.class) {
      return Integer.parseInt(text);
    }
    if (targetType == boolean.class) {
      return Boolean.parseBoolean(text);
    }
    if (targetType == byte.class) {
      return Byte.parseByte(text);
    }
    if (targetType == char.class) {
      checkState(text.length() == 1, "Expected a single character for char type");
      return text.charAt(0);
    }
    if (targetType == short.class) {
      return Short.parseShort(text);
    }
    if (targetType == double.class) {
      return Double.parseDouble(text);
    }
    if (targetType == float.class) {
      return Float.parseFloat(text);
    }
    if (targetType == long.class) {
      return Long.parseLong(text);
    }
    throw new UnsupportedOperationException("Expected a primitive type: " + text);
  }
}
