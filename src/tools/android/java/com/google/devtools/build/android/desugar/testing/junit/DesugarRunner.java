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

import static java.util.Collections.max;
import static java.util.Collections.min;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.android.desugar.testing.junit.SourceCompilationUnit.SourceCompilationException;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.FrameworkMethod;
import org.junit.runners.model.InitializationError;
import org.junit.runners.model.Statement;

/**
 * A custom test runner that powers desugar tests.
 *
 * <p>The test runner works the same way as {@link org.junit.runners.JUnit4}, except that,
 * <li>Allows each {@link Test} method to declare injectable parameters, whose values will be
 *     populated at runtime through {@link DesugarRule}.
 * <li>Performs a precondition check against {@link JdkSuppress} annotation if present. A test
 *     passes vacuously without execution, if the precondition check fails.
 */
public final class DesugarRunner extends BlockJUnit4ClassRunner {

  private static final int JAVA_RUNTIME_VERSION = JdkVersion.getJavaRuntimeVersion();

  private static final ImmutableSet<Class<? extends Annotation>> SUPPORTED_ANNOTATIONS =
      ImmutableSet.<Class<? extends Annotation>>builder()
          .addAll(RuntimeEntityResolver.SUPPORTED_QUALIFIERS)
          .add(FromParameterValueSource.class)
          .build();

  public DesugarRunner(Class<?> testClassLiteral) throws InitializationError {
    super(testClassLiteral);
  }

  @Override
  protected void validateTestMethods(List<Throwable> errors) {
    validatePublicVoidMethodsWithInjectableArgs(Test.class, /* isStatic= */ false, errors);
  }

  @Override
  protected boolean isIgnored(FrameworkMethod child) {
    return isJdkSuppressed(child) || super.isIgnored(child);
  }

  private boolean isJdkSuppressed(FrameworkMethod child) {
    JdkSuppress jdkSuppress = getJdkSuppress(child);
    return jdkSuppress != null
        && (JAVA_RUNTIME_VERSION < jdkSuppress.minJdkVersion()
            || JAVA_RUNTIME_VERSION > jdkSuppress.maxJdkVersion());
  }

  private JdkSuppress getJdkSuppress(FrameworkMethod child) {
    JdkSuppress jdkSuppress = child.getAnnotation(JdkSuppress.class);
    return jdkSuppress != null ? jdkSuppress : getTestClass().getAnnotation(JdkSuppress.class);
  }

  @Override
  protected List<FrameworkMethod> computeTestMethods() {
    List<FrameworkMethod> frameworkMethods = super.computeTestMethods();
    // Generates one method for each @ParameterValueSource instance.
    ImmutableList.Builder<FrameworkMethod> expandedFrameworkMethods = ImmutableList.builder();
    for (FrameworkMethod frameworkMethod : frameworkMethods) {
      Method reflectMethod = frameworkMethod.getMethod();
      if (reflectMethod.isAnnotationPresent(ParameterValueSourceSet.class)) {
        ParameterValueSourceSet paramValues =
            reflectMethod.getDeclaredAnnotation(ParameterValueSourceSet.class);
        for (ParameterValueSource paramValueBundle : paramValues.value()) {
          expandedFrameworkMethods.add(
              ValueSourceAnnotatedMethod.create(frameworkMethod.getMethod(), paramValueBundle));
        }
      } else if (reflectMethod.isAnnotationPresent(ParameterValueSource.class)) {
        ParameterValueSource paramValueBundle =
            reflectMethod.getDeclaredAnnotation(ParameterValueSource.class);
        expandedFrameworkMethods.add(
            ValueSourceAnnotatedMethod.create(frameworkMethod.getMethod(), paramValueBundle));
      } else {
        expandedFrameworkMethods.add(frameworkMethod);
      }
    }
    return expandedFrameworkMethods.build();
  }

  /**
   * In replacement of {@link BlockJUnit4ClassRunner#validatePublicVoidNoArgMethods} for @Test
   * method signature validation.
   */
  private void validatePublicVoidMethodsWithInjectableArgs(
      Class<? extends Annotation> annotation, boolean isStatic, List<Throwable> errors) {
    List<FrameworkMethod> methods = getTestClass().getAnnotatedMethods(annotation);
    for (FrameworkMethod eachTestMethod : methods) {
      eachTestMethod.validatePublicVoid(isStatic, errors);
      validateInjectableParameters(eachTestMethod, errors);
      validateParameterValueSource(eachTestMethod, errors);
    }
  }

  private static void validateInjectableParameters(
      FrameworkMethod eachTestMethod, List<Throwable> errors) {
    for (Parameter parameter : eachTestMethod.getMethod().getParameters()) {
      if (Arrays.stream(parameter.getDeclaredAnnotations())
          .map(Annotation::annotationType)
          .noneMatch(SUPPORTED_ANNOTATIONS::contains)) {
        errors.add(
            new Exception(
                String.format(
                    "Expected the parameter (%s) in @Test method (%s) annotated with one of"
                        + " the supported qualifiers %s to be injectable.",
                    parameter, eachTestMethod, SUPPORTED_ANNOTATIONS)));
      }
    }
  }

  private static void validateParameterValueSource(
      FrameworkMethod eachTestMethod, List<Throwable> errors) {
    Method reflectMethod = eachTestMethod.getMethod();
    long numOfParamsWithValueRequest =
        Arrays.stream(reflectMethod.getParameters())
            .filter(parameter -> parameter.isAnnotationPresent(FromParameterValueSource.class))
            .count();
    if (numOfParamsWithValueRequest > 0) {
      List<ParameterValueSource> parameterValueSources = new ArrayList<>();
      if (reflectMethod.isAnnotationPresent(ParameterValueSource.class)) {
        parameterValueSources.add(reflectMethod.getDeclaredAnnotation(ParameterValueSource.class));
      }
      if (reflectMethod.isAnnotationPresent(ParameterValueSourceSet.class)) {
        Collections.addAll(
            parameterValueSources,
            reflectMethod.getDeclaredAnnotation(ParameterValueSourceSet.class).value());
      }
      for (ParameterValueSource parameterValueSource : parameterValueSources) {
        int valueSourceLength = parameterValueSource.value().length;
        if (valueSourceLength != numOfParamsWithValueRequest) {
          errors.add(
              new Exception(
                  String.format(
                      "Parameter value source bundle (%s) and @FromParameterValueSource-annotated"
                          + " parameters in Method (%s) mismatch in length.",
                      parameterValueSource, eachTestMethod)));
        }
      }
    }
  }

  @Override
  protected Statement methodInvoker(FrameworkMethod method, Object test) {
    DesugarRule desugarRule =
        Iterables.getOnlyElement(
            getTestClass().getAnnotatedFieldValues(test, Rule.class, DesugarRule.class));

    // Compile source Java files before desugar transformation.
    try {
      desugarRule.compileSourceInputs();
    } catch (IOException | SourceCompilationException e) {
      throw new IllegalStateException(e);
    }

    JdkSuppress jdkSuppress = getJdkSuppress(method);
    if (jdkSuppress != null) {
      ImmutableMap<String, Integer> inputClassFileMajorVersions =
          desugarRule.getInputClassFileMajorVersionMap();
      ImmutableCollection<Integer> classFileVersions = inputClassFileMajorVersions.values();
      if (min(classFileVersions) < jdkSuppress.minJdkVersion()
          || max(classFileVersions) > jdkSuppress.maxJdkVersion()) {
        return new VacuousSuccess(
            String.format(
                "@Test method (%s) passed vacuously without execution: All or part of the input"
                    + " class file versions (%s) does not meet the declared prerequisite version"
                    + " range (%s) for testing.",
                method, inputClassFileMajorVersions, jdkSuppress));
      }
    }

    try {
      desugarRule.executeDesugarTransformation();
      desugarRule.injectTestInstanceFields();
      Method reflectMethod = method.getMethod();

      ImmutableMap.Builder<Parameter, Object> parameterBindingsBuilder = ImmutableMap.builder();
      // Load bindings from annotations.
      if (reflectMethod.isAnnotationPresent(ParameterValueSourceSet.class)
          || reflectMethod.isAnnotationPresent(ParameterValueSource.class)) {
        parameterBindingsBuilder.putAll(
            ((ValueSourceAnnotatedMethod) method).getResolvableParameters());
      }
      // Load bindings from desugar rule.
      parameterBindingsBuilder.putAll(desugarRule.getResolvableParameters(reflectMethod));

      ImmutableMap<Parameter, Object> parameterBindings = parameterBindingsBuilder.build();
      Parameter[] parameters = reflectMethod.getParameters();
      int parameterCount = parameters.length;
      Object[] resolvedParameterValues = new Object[parameterCount];
      for (int i = 0; i < parameterCount; i++) {
        resolvedParameterValues[i] = parameterBindings.get(parameters[i]);
      }
      return new InvokeMethodWithParams(method, test, resolvedParameterValues);
    } catch (Throwable throwable) {
      throw new IllegalStateException(throwable);
    }
  }

  /**
   * Used as in replacement of {@link org.junit.internal.runners.statements.InvokeMethod} for @Test
   * method execution.
   */
  static class InvokeMethodWithParams extends Statement {
    private final FrameworkMethod testMethod;
    private final Object target;
    private final Object[] params;

    InvokeMethodWithParams(FrameworkMethod testMethod, Object target, Object... params) {
      this.testMethod = testMethod;
      this.target = target;
      this.params = params;
    }

    @Override
    public void evaluate() throws Throwable {
      testMethod.invokeExplosively(target, params);
    }
  }

  /**
   * Used as a statement implementation indicating that a test will pass without the execution of
   * test method body.
   */
  static final class VacuousSuccess extends Statement {

    private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

    private final String reason;

    VacuousSuccess(String reason) {
      this.reason = reason;
    }

    @Override
    public void evaluate() throws Throwable {
      logger.atWarning().log(reason);
    }

    @Override
    public String toString() {
      return getClass().getSimpleName() + ":" + reason;
    }
  }
}
