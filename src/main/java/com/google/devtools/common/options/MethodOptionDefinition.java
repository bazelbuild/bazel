// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.Locale;
import javax.annotation.Nullable;

/**
 * Knowledge about how the option parser should get type information about options and set their
 * values that are defined by methods.
 */
public class MethodOptionDefinition extends OptionDefinition {

  /** Returns an {@code MethodOptionDefinition} for the given method. */
  @Nullable
  static MethodOptionDefinition extractOptionDefinition(Method method) {
    Option annotation = method.getAnnotation(Option.class);
    if (annotation == null) {
      return null;
    }
    Class<?> declaringClass = method.getDeclaringClass();
    @SuppressWarnings("unchecked") // Can't do generics with reflection
    Class<? extends OptionsBase> optionsClass = (Class<? extends OptionsBase>) declaringClass;
    return new MethodOptionDefinition(method, annotation, getImplClass(optionsClass));
  }

  /** Returns the generated implementation class for the given options class. */
  public static Class<? extends OptionsBase> getImplClass(
      Class<? extends OptionsBase> optionsClass) {
    Verify.verify(optionsClass.isAnnotationPresent(OptionsClass.class));
    String packageName = optionsClass.getPackage().getName();
    String simpleName = optionsClass.getSimpleName();
    String implClassName = packageName + "." + simpleName + "Impl";
    try {
      return Class.forName(implClassName).asSubclass(OptionsBase.class);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e); // The annotation processor should have been run
    }
  }

  /**
   * Returns an {@code MethodOptionDefinition} for the given method name in the given class.
   *
   * <p>This is intended to be used by the generated implementation classes.
   */
  public static MethodOptionDefinition get(
      Class<? extends OptionsBase> optionsClass, String methodName) {
    try {
      Class<?> methodsClass;
      Class<? extends OptionsBase> implClass;
      if (optionsClass.isAnnotationPresent(OptionsClass.class)) {
        methodsClass = optionsClass;
        implClass = getImplClass(optionsClass);
      } else {
        throw new IllegalStateException(optionsClass + " is not an @OptionsClass");
      }

      Method method = methodsClass.getDeclaredMethod(methodName);
      Option result = method.getAnnotation(Option.class);
      if (result == null) {
        throw new IllegalStateException(methodName + " is not an @Option");
      }
      return new MethodOptionDefinition(method, result, implClass);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    }
  }

  private final Method method;
  private final Field field;

  private MethodOptionDefinition(
      Method method, Option optionAnnotation, Class<? extends OptionsBase> implClass) {
    super(optionAnnotation);
    this.method = method;
    try {
      String methodName = method.getName();
      Verify.verify(methodName.startsWith("get")); // Enforced by the annotation processor
      String fieldName =
          methodName.substring(3, 4).toLowerCase(Locale.ROOT) + methodName.substring(4);
      this.field = implClass.getDeclaredField(fieldName);
      this.field.setAccessible(true);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public <C extends OptionsBase> Class<? extends C> getDeclaringClass(Class<C> baseClass) {
    // The implementation class is not technically the "declaring" class, but it's the one that is
    // referenced everywhere, so this is what needs to be returned. In particular, that's the one
    // that needs to be passed to getOptions().
    Class<?> methodClass = method.getDeclaringClass();
    Preconditions.checkArgument(baseClass.isAssignableFrom(methodClass));
    @SuppressWarnings("unchecked") // This should be safe based on the previous check.
    Class<? extends C> castClass = (Class<? extends C>) methodClass;
    return castClass;
  }

  @Override
  public Object getRawValue(OptionsBase optionsBase) {
    try {
      return method.invoke(optionsBase);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void setValue(OptionsBase optionsBase, Object value) {
    try {
      field.set(optionsBase, value);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public boolean isDeprecated() {
    return method.isAnnotationPresent(Deprecated.class);
  }

  @Override
  public Class<?> getType() {
    return method.getReturnType();
  }

  @Override
  protected Type getSingularType() {
    return method.getGenericReturnType();
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof MethodOptionDefinition that)) {
      return false;
    }
    return this.method.equals(that.method);
  }

  @Override
  public int hashCode() {
    return method.hashCode();
  }

  @Override
  public String getMemberName() {
    return method.getName();
  }
}
