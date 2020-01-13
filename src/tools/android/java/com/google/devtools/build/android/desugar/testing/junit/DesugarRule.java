/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoAnnotation;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.AnnotatedElement;
import java.lang.reflect.Field;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/** A JUnit4 Rule that desugars an input jar file and load the transformed jar to JVM. */
public final class DesugarRule implements TestRule {

  static final ClassLoader BASE_CLASS_LOADER = ClassLoader.getSystemClassLoader().getParent();

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;

  private final ImmutableList<Field> injectableFields;
  private final RuntimeEntityResolver runtimeEntityResolver;

  /**
   * The entry point to create a {@link DesugarRule}.
   *
   * @param testInstance The <code>this</code> reference of the JUnit test class.
   * @param testInstanceLookup The lookup object from the test class, i.e.<code>
   *     MethodHandles.lookup()</code>
   */
  public static DesugarRuleBuilder builder(Object testInstance, Lookup testInstanceLookup) {
    return new DesugarRuleBuilder(testInstance, testInstanceLookup);
  }

  DesugarRule(
      Object testInstance,
      Lookup testInstanceLookup,
      ImmutableList<Field> injectableFields,
      RuntimeEntityResolver runtimeEntityResolver) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;
    this.injectableFields = injectableFields;
    this.runtimeEntityResolver = runtimeEntityResolver;
  }

  ImmutableSet<Integer> getInputClassFileMajorVersions() {
    try {
      return runtimeEntityResolver.getInputClassFileMajorVersions();
    } catch (Throwable throwable) {
      throw new IllegalStateException(throwable);
    }
  }

  @Override
  public Statement apply(Statement base, Description description) {
    return new Statement() {
      @Override
      public void evaluate() throws Throwable {
        before();
        base.evaluate();
      }
    };
  }

  private void before() throws Throwable {
    for (Field field : injectableFields) {
      MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
      fieldSetter.invoke(testInstance, resolve(field, field.getType()));
    }
  }

  public Object resolve(AnnotatedElement param, Class<?> type) {
    try {
      return runtimeEntityResolver.resolve(param, type);
    } catch (Throwable throwable) {
      throw new AssertionError(throwable);
    }
  }

  @AutoAnnotation
  static DynamicClassLiteral createDynamicClassLiteral(String value, int round) {
    return new AutoAnnotation_DesugarRule_createDynamicClassLiteral(value, round);
  }

}
