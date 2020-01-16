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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.android.desugar.testing.junit.SourceCompilationUnit.SourceCompilationException;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Parameter;
import java.util.Arrays;
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

  /**
   * In replacement of {@link BlockJUnit4ClassRunner#validatePublicVoidNoArgMethods} for @Test
   * method signature validation.
   */
  private void validatePublicVoidMethodsWithInjectableArgs(
      Class<? extends Annotation> annotation, boolean isStatic, List<Throwable> errors) {
    List<FrameworkMethod> methods = getTestClass().getAnnotatedMethods(annotation);
    for (FrameworkMethod eachTestMethod : methods) {
      eachTestMethod.validatePublicVoid(isStatic, errors);
      for (Parameter parameter : eachTestMethod.getMethod().getParameters()) {
        if (Arrays.stream(parameter.getDeclaredAnnotations())
            .map(Annotation::annotationType)
            .noneMatch(RuntimeEntityResolver.SUPPORTED_QUALIFIERS::contains)) {
          errors.add(
              new Exception(
                  String.format(
                      "Expected the parameter (%s) in @Test method (%s) annotated with one of"
                          + " the supported qualifiers %s to be injectable.",
                      parameter, eachTestMethod, RuntimeEntityResolver.SUPPORTED_QUALIFIERS)));
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
      Object[] resolvedParameterValues =
          desugarRule.resolveParameters(method.getMethod().getParameters());
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
