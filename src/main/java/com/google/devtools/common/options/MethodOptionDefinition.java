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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Knowledge about how the option parser should get type information about options and set their
 * values that are defined by methods.
 */
public class MethodOptionDefinition extends OptionDefinition {

  /** Returns an {@code MethodOptionDefinition} for the given method. */
  @Nullable
  static MethodOptionDefinition extractOptionDefinition(
      Method method, Class<? extends OptionsBase> optionsClass) {
    Option annotation = method.getAnnotation(Option.class);
    if (annotation == null) {
      return null;
    }
    return new MethodOptionDefinition(method, annotation);
  }

  /** Returns the generated implementation class for the given options class. */
  public static Class<? extends OptionsBase> getImplClass(
      Class<? extends OptionsBase> optionsClass) {
    Verify.verify(optionsClass.isAnnotationPresent(OptionsClass.class));
    String packageName = optionsClass.getPackage().getName();
    String className = optionsClass.getName().substring(packageName.length() + 1);
    String implClassName = packageName + "." + className.replace('$', '_') + "Impl";
    try {
      return Class.forName(implClassName, true, optionsClass.getClassLoader())
          .asSubclass(OptionsBase.class);
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
      if (!optionsClass.isAnnotationPresent(OptionsClass.class)) {
        throw new IllegalStateException(optionsClass + " is not an @OptionsClass");
      }
      Method method = optionsClass.getMethod(methodName);
      Option result = method.getAnnotation(Option.class);
      if (result == null) {
        throw new IllegalStateException(methodName + " is not an @Option");
      }
      return new MethodOptionDefinition(method, result);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    }
  }

  private final Method method;
  private final String fieldName;

  // This is needed because options classes sometimes inherit from each other. In this case, the
  // field storing the value can be on multiple classes (e.g. if FooOptions inherits from
  // BarOptions, the implementation of both will have fields corresponding to the options in
  // BarOptions)
  private final ConcurrentMap<Class<? extends OptionsBase>, Field> fieldCache =
      new ConcurrentHashMap<>();

  private MethodOptionDefinition(Method method, Option optionAnnotation) {
    super(optionAnnotation);
    this.method = method;
    String methodName = method.getName();
    Verify.verify(methodName.startsWith("get")); // Enforced by the annotation processor
    this.fieldName = methodName.substring(3, 4).toLowerCase(Locale.ROOT) + methodName.substring(4);
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

  private Field getField(Class<? extends OptionsBase> optionsClass) {
    try {
      Class<? extends OptionsBase> implClass =
          getImplClass(optionsClass.asSubclass(OptionsBase.class));
      Field f = implClass.getDeclaredField(fieldName);
      f.setAccessible(true);
      return f;
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(
          "Could not find field " + fieldName + " in implementation of " + optionsClass, e);
    }
  }

  @Override
  public void setValue(OptionsBase optionsBase, Object value) {
    Field field =
        fieldCache.computeIfAbsent(
            optionsBase.getOptionsClass().asSubclass(OptionsBase.class), this::getField);
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

  @Override
  public String toString() {
    return String.format("option '--%s'", getOptionName());
  }
}
