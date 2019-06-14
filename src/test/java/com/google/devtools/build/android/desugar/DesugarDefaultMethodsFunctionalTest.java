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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethodParameters;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethodWithParam;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethodWithParam.MethodInvocations;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Variant of {@link DesugarJava8FunctionalTest} that expects default and static interface methods
 * to be desugared
 */
@RunWith(JUnit4.class)
public final class DesugarDefaultMethodsFunctionalTest extends DesugarJava8FunctionalTest {

  public DesugarDefaultMethodsFunctionalTest() {
    super(/*expectBridgesFromSeparateTarget*/ true, /*expectDefaultMethods*/ false);
  }

  @Test
  public void staticMethodsAreAbsentFromInterfaceBytecode() {
    List<Method> staticMethods =
        Arrays.stream(InterfaceMethodParameters.class.getDeclaredMethods())
            .filter(method -> Modifier.isStatic(method.getModifiers()))
            .collect(Collectors.toList());

    assertThat(staticMethods).isEmpty();
  }

  @Test
  public void defaultMethodsAreAbsentFromInterfaceBytecode() {
    List<Method> defaultMethods =
        Arrays.stream(InterfaceMethodParameters.class.getDeclaredMethods())
            .filter(Method::isDefault)
            .collect(Collectors.toList());

    assertThat(defaultMethods).isEmpty();
  }

  @Test
  public void invokeInterfaceStaticMethod() {
    assertThat(InterfaceMethodWithParam.MethodInvocations.simpleComputeStatic(1234))
        .isEqualTo(1234L);
  }

  @Test
  public void invokeInterfaceDefaultMethod() {
    assertThat(InterfaceMethodWithParam.MethodInvocations.simpleComputeDefault(1234))
        .isEqualTo(1234L);
  }

  @Test
  public void parameterNamesArePreservedOnDesugaredDefaultMethods() throws Exception {
    Method desugaredDefaultMethod = MethodInvocations.inspectDesugaredDefaultMethod();

    assertThat(desugaredDefaultMethod.getParameters()[0].getName())
        .isEqualTo("v12525d61e4b10b3e27bc280dd61e56728e3e8c27");
  }

  @Test
  public void parameterAnnotationsArePreservedOnDesugaredDefaultMethods() throws Exception {
    Method desugaredDefaultMethod = MethodInvocations.inspectDesugaredDefaultMethod();

    Annotation[][] parameterAnnotations = desugaredDefaultMethod.getParameterAnnotations();

    assertThat(parameterAnnotations).hasLength(1);
    assertThat(parameterAnnotations[0]).hasLength(1);
    assertThat(parameterAnnotations[0][0]).isInstanceOf(InterfaceMethodWithParam.Foo.class);
  }

  @Test
  public void methodAnnotationsArePreservedOnDesugaredDefaultMethods() throws Exception {
    Method method = MethodInvocations.inspectDesugaredDefaultMethod();

    List<Annotation> annotations = Arrays.asList(method.getAnnotations());

    Annotation parameterAnnotation = Iterables.getOnlyElement(annotations);
    assertThat(parameterAnnotation).isInstanceOf(InterfaceMethodWithParam.Foo.class);

    InterfaceMethodWithParam.Foo fooInstance = (InterfaceMethodWithParam.Foo) parameterAnnotation;
    assertThat(fooInstance.value()).isEqualTo("custom-attr-value-2");
  }

  @Test
  public void methodReturnTypeAnnotationsArePreservedOnDesugaredDefaultMethods() throws Exception {
    Method method = MethodInvocations.inspectDesugaredDefaultMethod();

    List<Annotation> annotations = Arrays.asList(method.getAnnotatedReturnType().getAnnotations());

    assertThat(Iterables.getOnlyElement(annotations))
        .isInstanceOf(InterfaceMethodWithParam.TyFoo.class);
  }

  @Test
  public void typeAnnotationsArePreservedOnDesugaredDefaultMethods() throws Exception {
    Method desugaredDefaultMethod = MethodInvocations.inspectDesugaredDefaultMethod();

    List<Parameter> parameters = Arrays.asList(desugaredDefaultMethod.getParameters());
    List<Annotation> typeAnnotations =
        Arrays.asList(Iterables.getOnlyElement(parameters).getAnnotatedType().getAnnotations());

    assertThat(Iterables.getOnlyElement(typeAnnotations))
        .isInstanceOf(InterfaceMethodWithParam.TyFoo.class);
  }

  /** Test for b/129719629. */
  @Test
  public void parameterNamesArePreservedOnStaticMethodCompanions() throws Exception {
    Method method = InterfaceMethodWithParam.MethodInvocations.inspectCompanionOfStaticMethod();

    assertThat(method.getParameters()).hasLength(1);
    assertThat(method.getParameters()[0].getName())
        .isEqualTo("v4897b02fddeda3bb31bc15b3cad0f6febc61508");
  }

  @Test
  public void parameterAnnotationsArePreservedOnStaticMethodCompanions() throws Exception {
    Method method = InterfaceMethodWithParam.MethodInvocations.inspectCompanionOfStaticMethod();
    Annotation[][] parameterAnnotations = method.getParameterAnnotations();
    assertThat(parameterAnnotations).hasLength(1);
    assertThat(parameterAnnotations[0]).hasLength(1);
    assertThat(parameterAnnotations[0][0]).isInstanceOf(InterfaceMethodWithParam.Foo.class);
  }

  @Test
  public void methodParameterAnnotationsArePreservedOnStaticMethodCompanions() throws Exception {
    Method method = InterfaceMethodWithParam.MethodInvocations.inspectCompanionOfStaticMethod();
    Annotation[] parameterAnnotations = method.getAnnotations();
    assertThat(parameterAnnotations).hasLength(1);
    assertThat(parameterAnnotations[0]).isInstanceOf(InterfaceMethodWithParam.Foo.class);
  }

  @Test
  public void typeAnnotationsArePreservedOnStaticMethodCompanions() throws Exception {
    Method method = InterfaceMethodWithParam.MethodInvocations.inspectCompanionOfStaticMethod();
    List<Parameter> parameterAnnotations = Arrays.asList(method.getParameters());

    List<Annotation> typeAnnoations =
        Arrays.asList(
            Iterables.getOnlyElement(parameterAnnotations).getAnnotatedType().getAnnotations());

    assertThat(Iterables.getOnlyElement(typeAnnoations))
        .isInstanceOf(InterfaceMethodWithParam.TyFoo.class);
  }

  @Test
  public void methodReturnTypeAnnotationsArePreservedOnStaticMethodCompanions() throws Exception {
    Method method = InterfaceMethodWithParam.MethodInvocations.inspectCompanionOfStaticMethod();
    List<Annotation> returnTypeAnnotations =
        Arrays.asList(method.getAnnotatedReturnType().getAnnotations());

    assertThat(Iterables.getOnlyElement(returnTypeAnnotations))
        .isInstanceOf(InterfaceMethodWithParam.TyFoo.class);
  }

  /** Test for b/129719629. */
  @Test
  public void parameterNamesAreOmittedOnDefaultMethodCompanions() throws Exception {
    Method method =
        InterfaceMethodWithParam.MethodInvocations.inspectCompanionMethodOfDefaultMethod();

    assertThat(method.getParameters()).hasLength(2);
    assertThat(method.getParameters()[1].getName())
        .isNotEqualTo("v12525d61e4b10b3e27bc280dd61e56728e3e8c27");
  }

  @Test
  public void parameterAnnotationsAreOmittedOnDefaultMethodCompanions() throws Exception {
    Method method =
        InterfaceMethodWithParam.MethodInvocations.inspectCompanionMethodOfDefaultMethod();

    assertThat(method.getParameterAnnotations()).hasLength(2);
    assertThat(method.getParameterAnnotations()[0]).isEmpty();
    assertThat(method.getParameterAnnotations()[1]).isEmpty();
  }

  @Test
  public void methodAnnotationsArePreservedOnDefaultMethodCompanions() throws Exception {
    Method method =
        InterfaceMethodWithParam.MethodInvocations.inspectCompanionMethodOfDefaultMethod();

    List<Annotation> annotations = Arrays.asList(method.getAnnotations());
    Annotation methodAnnotation = Iterables.getOnlyElement(annotations);
    assertThat(methodAnnotation).isInstanceOf(InterfaceMethodWithParam.Foo.class);

    InterfaceMethodWithParam.Foo fooInstance = (InterfaceMethodWithParam.Foo) methodAnnotation;
    assertThat(fooInstance.value()).isEqualTo("custom-attr-value-2");
  }

  @Test
  public void methodReturnTypeAnnotationsAreOmittedOnDefaultMethodCompanions() throws Exception {
    Method method =
        InterfaceMethodWithParam.MethodInvocations.inspectCompanionMethodOfDefaultMethod();

    Annotation[] methodReturnTypeAnnotations = method.getAnnotatedReturnType().getAnnotations();

    assertThat(methodReturnTypeAnnotations).isEmpty();
  }

  @Test
  public void parameterTypeAnnotationsAreOmittedOnDefaultMethodCompanion() throws Exception {
    Method method =
        InterfaceMethodWithParam.MethodInvocations.inspectCompanionMethodOfDefaultMethod();

    List<Parameter> parameterAnnotations = Arrays.asList(method.getParameters());

    assertThat(parameterAnnotations).hasSize(2);
    assertThat(parameterAnnotations.get(0).getAnnotatedType().getAnnotations()).isEmpty();
    assertThat(parameterAnnotations.get(1).getAnnotatedType().getAnnotations()).isEmpty();
  }
}
