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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoAnnotation;
import com.google.common.collect.ImmutableList;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.zip.ZipEntry;
import javax.inject.Inject;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/** A JUnit4 Rule that desugars an input jar file and load the transformed jar to JVM. */
public final class DesugarRule implements TestRule {

  static final ClassLoader BASE_CLASS_LOADER = ClassLoader.getSystemClassLoader().getParent();

  private final Object testInstance;
  private final MethodHandles.Lookup testInstanceLookup;

  private final ImmutableList<Field> injectableClassLiterals;
  private final ImmutableList<Field> injectableAsmNodes;
  private final ImmutableList<Field> injectableMethodHandles;
  private final ImmutableList<Field> injectableZipEntries;

  private final Map<DynamicClassLiteral, Class<?>> dynamicClassLiterals;
  private final Map<AsmNode, Object> asmNodes;
  private final Map<RuntimeMethodHandle, MethodHandle> runtimeMethodHandles;
  private final Map<RuntimeZipEntry, ZipEntry> runtimeZipEntries;

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
      ImmutableList<Field> injectableAsmNodes,
      Object testInstance,
      Lookup testInstanceLookup,
      ImmutableList<Field> injectableClassLiterals,
      ImmutableList<Field> injectableMethodHandles,
      ImmutableList<Field> injectableZipEntries,
      Path androidRuntimeJar,
      Path jacocoAgentJar,
      Map<DynamicClassLiteral, Class<?>> dynamicClassLiterals,
      Map<AsmNode, Object> asmNodes,
      Map<RuntimeMethodHandle, MethodHandle> runtimeMethodHandles,
      Map<RuntimeZipEntry, ZipEntry> runtimeZipEntries) {
    this.testInstance = testInstance;
    this.testInstanceLookup = testInstanceLookup;

    this.injectableClassLiterals = injectableClassLiterals;
    this.injectableAsmNodes = injectableAsmNodes;
    this.injectableMethodHandles = injectableMethodHandles;
    this.injectableZipEntries = injectableZipEntries;
    this.dynamicClassLiterals = dynamicClassLiterals;
    this.asmNodes = asmNodes;
    this.runtimeMethodHandles = runtimeMethodHandles;
    this.runtimeZipEntries = runtimeZipEntries;

    checkState(Files.exists(androidRuntimeJar));
    checkState(Files.exists(jacocoAgentJar));
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
    for (Field field : injectableClassLiterals) {
      DynamicClassLiteral dynamicClassLiteralRequest =
          field.getDeclaredAnnotation(DynamicClassLiteral.class);
      Class<?> classLiteral = dynamicClassLiterals.get(dynamicClassLiteralRequest);
      MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
      fieldSetter.invoke(testInstance, classLiteral);
    }

    for (Field field : injectableAsmNodes) {
      Object asmNode = asmNodes.get(field.getDeclaredAnnotation(AsmNode.class));
      MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
      fieldSetter.invoke(testInstance, asmNode);
    }

    for (Field field : injectableMethodHandles) {
      MethodHandle methodHandle =
          runtimeMethodHandles.get(field.getDeclaredAnnotation(RuntimeMethodHandle.class));
      MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
      fieldSetter.invoke(testInstance, methodHandle);
    }

    for (Field field : injectableZipEntries) {
      ZipEntry zipEntry = runtimeZipEntries.get(field.getDeclaredAnnotation(RuntimeZipEntry.class));
      MethodHandle fieldSetter = testInstanceLookup.unreflectSetter(field);
      fieldSetter.invoke(testInstance, zipEntry);
    }
  }

  @AutoAnnotation
  static DynamicClassLiteral createDynamicClassLiteral(String value, int round) {
    return new AutoAnnotation_DesugarRule_createDynamicClassLiteral(value, round);
  }

  static ImmutableList<Field> findAllInjectableFieldsWithQualifier(
      Class<?> testClass, Class<? extends Annotation> annotationClass) {
    ImmutableList.Builder<Field> fields = ImmutableList.builder();
    for (Class<?> currentClass = testClass;
        currentClass != null;
        currentClass = currentClass.getSuperclass()) {
      for (Field field : currentClass.getDeclaredFields()) {
        if (field.isAnnotationPresent(Inject.class) && field.isAnnotationPresent(annotationClass)) {
          fields.add(field);
        }
      }
    }
    return fields.build();
  }
}
