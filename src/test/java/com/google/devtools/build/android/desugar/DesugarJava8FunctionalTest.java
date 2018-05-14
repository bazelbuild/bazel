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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.testdata.java8.AnnotationsOfDefaultMethodsShouldBeKept.AnnotatedInterface;
import com.google.devtools.build.android.desugar.testdata.java8.AnnotationsOfDefaultMethodsShouldBeKept.SomeAnnotation;
import com.google.devtools.build.android.desugar.testdata.java8.ConcreteDefaultInterfaceWithLambda;
import com.google.devtools.build.android.desugar.testdata.java8.ConcreteOverridesDefaultWithLambda;
import com.google.devtools.build.android.desugar.testdata.java8.DefaultInterfaceMethodWithStaticInitializer;
import com.google.devtools.build.android.desugar.testdata.java8.DefaultInterfaceWithBridges;
import com.google.devtools.build.android.desugar.testdata.java8.DefaultMethodFromSeparateJava8Target;
import com.google.devtools.build.android.desugar.testdata.java8.DefaultMethodFromSeparateJava8TargetOverridden;
import com.google.devtools.build.android.desugar.testdata.java8.DefaultMethodTransitivelyFromSeparateJava8Target;
import com.google.devtools.build.android.desugar.testdata.java8.FunctionWithDefaultMethod;
import com.google.devtools.build.android.desugar.testdata.java8.FunctionalInterfaceWithInitializerAndDefaultMethods;
import com.google.devtools.build.android.desugar.testdata.java8.GenericDefaultInterfaceWithLambda;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceMethod;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceWithDefaultMethod;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceWithDuplicateMethods.ClassWithDuplicateMethods;
import com.google.devtools.build.android.desugar.testdata.java8.InterfaceWithInheritedMethods;
import com.google.devtools.build.android.desugar.testdata.java8.Java7InterfaceWithBridges;
import com.google.devtools.build.android.desugar.testdata.java8.Named;
import com.google.devtools.build.android.desugar.testdata.java8.TwoInheritedDefaultMethods;
import com.google.devtools.build.android.desugar.testdata.java8.VisibilityTestClass;
import java.lang.annotation.Annotation;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test that exercises classes in the {@code testdata_java8} package. This is meant to be run
 * against a desugared version of those classes, which in turn exercise various desugaring features.
 */
@RunWith(JUnit4.class)
public class DesugarJava8FunctionalTest extends DesugarFunctionalTest {

  public DesugarJava8FunctionalTest() {
    this(true, true);
  }

  protected DesugarJava8FunctionalTest(
      boolean expectBridgesFromSeparateTarget, boolean expectDefaultMethods) {
    super(expectBridgesFromSeparateTarget, expectDefaultMethods);
  }

  @Test
  public void testLambdaInDefaultMethod() {
    assertThat(new ConcreteDefaultInterfaceWithLambda().defaultWithLambda())
        .containsExactly("0", "1")
        .inOrder();
  }

  @Test
  public void testLambdaInDefaultCallsInterfaceMethod() {
    assertThat(new ConcreteDefaultInterfaceWithLambda().defaultCallsInterfaceMethod())
        .containsExactly("1", "2")
        .inOrder();
  }

  @Test
  public void testOverrideLambdaInDefault() {
    assertThat(new ConcreteOverridesDefaultWithLambda().defaultWithLambda())
        .containsExactly("2", "3")
        .inOrder();
  }

  @Test
  public void testLambdaInDefaultCallsOverrideMethod() {
    assertThat(new ConcreteOverridesDefaultWithLambda().defaultCallsInterfaceMethod())
        .containsExactly("3", "4")
        .inOrder();
  }

  @Test
  public void testDefaultInterfaceMethodReference() {
    InterfaceMethod methodrefUse = new InterfaceMethod.Concrete();
    List<String> dest =
        methodrefUse.defaultMethodReference(ImmutableList.of("Sergey", "Larry", "Alex"));
    assertThat(dest).containsExactly("Sergey");
  }

  @Test
  public void testStaticInterfaceMethodReference() {
    InterfaceMethod methodrefUse = new InterfaceMethod.Concrete();
    List<String> dest =
        methodrefUse.staticMethodReference(ImmutableList.of("Sergey", "Larry", "Alex"));
    assertThat(dest).containsExactly("Alex");
  }

  @Test
  public void testLambdaCallsDefaultMethod() {
    InterfaceMethod methodrefUse = new InterfaceMethod.Concrete();
    List<String> dest =
        methodrefUse.lambdaCallsDefaultMethod(ImmutableList.of("Sergey", "Larry", "Alex"));
    assertThat(dest).containsExactly("Sergey");
  }

  @Test
  public void testBootclasspathMethodInvocations() {
    InterfaceMethod concrete = new InterfaceMethod.Concrete();
    assertThat(concrete.defaultInvokingBootclasspathMethods("Larry")).isEqualTo("Larry");
  }

  @Test
  public void testStaticMethodsInInterface_explicitAndLambdaBody() {
    List<Long> result = FunctionWithDefaultMethod.DoubleInts.add(ImmutableList.of(7, 39, 8), 3);
    assertThat(result).containsExactly(10L, 42L, 11L).inOrder();
  }

  @Test
  public void testOverriddenDefaultMethod_inHandwrittenClass() {
    FunctionWithDefaultMethod<Integer> doubler = new FunctionWithDefaultMethod.DoubleInts();
    assertThat(doubler.apply(7)).isEqualTo(14);
    assertThat(doubler.twice(7)).isEqualTo(35);
  }

  @Test
  public void testOverriddenDefaultMethod_inHandwrittenSuperclass() {
    FunctionWithDefaultMethod<Integer> doubler = new FunctionWithDefaultMethod.DoubleInts2();
    assertThat(doubler.apply(7)).isEqualTo(14);
    assertThat(doubler.twice(7)).isEqualTo(35);
  }

  @Test
  public void testInheritedDefaultMethod_inLambda() {
    FunctionWithDefaultMethod<Integer> doubler =
        FunctionWithDefaultMethod.DoubleInts.doubleLambda();
    assertThat(doubler.apply(7)).isEqualTo(14);
    assertThat(doubler.twice(7)).isEqualTo(28);
  }

  @Test
  public void testDefaultMethodReference_onLambda() {
    FunctionWithDefaultMethod<Integer> plus6 = FunctionWithDefaultMethod.DoubleInts.incTwice(3);
    assertThat(plus6.apply(18)).isEqualTo(24);
    assertThat(plus6.twice(18)).isEqualTo(30);
  }

  @Test
  public void testDefaultMethodReference_onHandwrittenClass() {
    FunctionWithDefaultMethod<Integer> times5 = FunctionWithDefaultMethod.DoubleInts.times5();
    assertThat(times5.apply(6)).isEqualTo(30);
    assertThat(times5.twice(6)).isEqualTo(150); // Irrelevant that DoubleInts overrides twice()
  }

  @Test
  public void testStaticInterfaceMethodReferenceReturned() {
    Function<Integer, FunctionWithDefaultMethod<Integer>> factory =
        FunctionWithDefaultMethod.DoubleInts.incFactory();
    assertThat(factory.apply(6).apply(7)).isEqualTo(13);
    assertThat(factory.apply(6).twice(7)).isEqualTo(19);
  }

  @Test
  public void testSuperDefaultMethodInvocation() {
    assertThat(new TwoInheritedDefaultMethods().name()).isEqualTo("One:Two");
    assertThat(new Named.DefaultName().name()).isEqualTo("DefaultName-once");
    assertThat(new Named.DefaultNameSubclass().name()).isEqualTo("DefaultNameSubclass-once-twice");
  }

  @Test
  public void testInheritedPreferredOverDefault() throws Exception {
    assertThat(new Named.ExplicitName("hello").name()).isEqualTo("hello");
    // Make sure AbstractName remains abstract, despite default method from implemented interface
    assertThat(Modifier.isAbstract(Named.AbstractName.class.getMethod("name").getModifiers()))
        .isTrue();
  }

  @Test
  public void testRedefinedDefaultMethod() throws Exception {
    assertThat(new InterfaceWithDefaultMethod.Version2().version()).isEqualTo(2);
  }

  @Test
  public void testDefaultMethodRedefinedInSubclass() throws Exception {
    assertThat(new InterfaceWithDefaultMethod.AlsoVersion2().version()).isEqualTo(2);
  }

  @Test
  public void testDefaultMethodVisibility() {
    assertThat(new VisibilityTestClass().m()).isEqualTo(42);
  }

  /** Test for b/38302860 */
  @Test
  public void testAnnotationsOfDefaultMethodsAreKept() throws Exception {
    {
      Annotation[] annotations = AnnotatedInterface.class.getAnnotations();
      assertThat(annotations).hasLength(1);
      assertThat(annotations[0]).isInstanceOf(SomeAnnotation.class);
      assertThat(((SomeAnnotation) annotations[0]).value()).isEqualTo(1);
    }
    {
      Annotation[] annotations =
          AnnotatedInterface.class.getMethod("annotatedAbstractMethod").getAnnotations();
      assertThat(annotations).hasLength(1);
      assertThat(annotations[0]).isInstanceOf(SomeAnnotation.class);
      assertThat(((SomeAnnotation) annotations[0]).value()).isEqualTo(2);
    }
    {
      Annotation[] annotations =
          AnnotatedInterface.class.getMethod("annotatedDefaultMethod").getAnnotations();
      assertThat(annotations).hasLength(1);
      assertThat(annotations[0]).isInstanceOf(SomeAnnotation.class);
      assertThat(((SomeAnnotation) annotations[0]).value()).isEqualTo(3);
    }
  }
  /** Test for b/38308515 */
  @Test
  public void testDefaultAndStaticMethodNameClash() {
    final ClassWithDuplicateMethods instance = new ClassWithDuplicateMethods();
    assertThat(instance.getZero()).isEqualTo(0);
    assertThat(instance.getZeroFromStaticInterfaceMethod()).isEqualTo(1);
  }

  /**
   * Test for b/38257037
   *
   * <p>Note that, we intentionally suppress unchecked warnings, because we expect some
   * ClassCastException to test bridge methods.
   */
  @SuppressWarnings("unchecked")
  @Test
  public void testBridgeAndDefaultMethods() {
    {
      DefaultInterfaceWithBridges object = new DefaultInterfaceWithBridges();
      Integer one = 1;
      assertThat(object.copy(one)).isEqualTo(one);
      assertThat(object.copy((Number) one)).isEqualTo(one);
      assertThrows(ClassCastException.class, () -> object.copy(Double.valueOf(1)));

      assertThat(object.getNumber()).isInstanceOf(Double.class);
      assertThat(object.getNumber()).isEqualTo(Double.valueOf(2.3d));
      assertThat(object.getDouble()).isEqualTo(Double.valueOf(2.3d));
    }
    {
      Java7InterfaceWithBridges.ClassAddTwo testObject =
          new Java7InterfaceWithBridges.ClassAddTwo();
      assertThat(testObject.add(Integer.valueOf(2))).isEqualTo(4);

      @SuppressWarnings("rawtypes")
      Java7InterfaceWithBridges top = testObject;
      assertThat(top.add(Integer.valueOf(2))).isEqualTo(4);
      assertThrows(ClassCastException.class, () -> top.add(new Object()));
      assertThrows(ClassCastException.class, () -> top.add(Double.valueOf(1)));
      assertThrows(ClassCastException.class, () -> top.add(Long.valueOf(1)));

      @SuppressWarnings("rawtypes")
      Java7InterfaceWithBridges.LevelOne levelOne = testObject;
      assertThat(levelOne.add(Integer.valueOf(2))).isEqualTo(4);
      assertThrows(ClassCastException.class, () -> top.add(new Object()));
      assertThrows(ClassCastException.class, () -> top.add(Double.valueOf(1)));
      assertThrows(ClassCastException.class, () -> top.add(Long.valueOf(1)));

      @SuppressWarnings("rawtypes")
      Java7InterfaceWithBridges.LevelOne levelTwo = testObject;
      assertThat(levelTwo.add(Integer.valueOf(2))).isEqualTo(4);
      assertThrows(ClassCastException.class, () -> levelTwo.add(Double.valueOf(1)));
      assertThrows(ClassCastException.class, () -> levelTwo.add(Long.valueOf(1)));
    }
    {
      GenericDefaultInterfaceWithLambda.ClassTwo testObject =
          new GenericDefaultInterfaceWithLambda.ClassTwo();

      assertThat(testObject.increment(Integer.valueOf(0))).isEqualTo(1);
      assertThat(testObject.toString(Integer.valueOf(0))).isEqualTo("0");
      assertThat(testObject.getBaseValue()).isEqualTo(Integer.valueOf(0));

      assertThat(testObject.toList(0)).isEmpty();
      assertThat(testObject.toList(1)).containsExactly(0).inOrder();
      assertThat(testObject.toList(2)).containsExactly(0, 1).inOrder();

      assertThat(((Function<Integer, ArrayList<Integer>>) testObject.toListSupplier()).apply(0))
          .isEmpty();
      assertThat(((Function<Integer, ArrayList<Integer>>) testObject.toListSupplier()).apply(1))
          .containsExactly(0)
          .inOrder();
      assertThat(((Function<Integer, ArrayList<Integer>>) testObject.toListSupplier()).apply(2))
          .containsExactly(0, 1)
          .inOrder();

      assertThat(testObject.convertToStringList(ImmutableList.of(0)))
          .containsExactly("0")
          .inOrder();
      assertThat(testObject.convertToStringList(ImmutableList.of(0, 1)))
          .containsExactly("0", "1")
          .inOrder();

      @SuppressWarnings("rawtypes")
      GenericDefaultInterfaceWithLambda top = testObject;
      assertThrows(ClassCastException.class, () -> top.increment(Long.valueOf(1)));
      assertThrows(ClassCastException.class, () -> top.toString(Long.valueOf(1)));
      assertThat(top.increment(Integer.valueOf(0))).isEqualTo(1);
      assertThat(top.toString(Integer.valueOf(0))).isEqualTo("0");
      assertThat(top.getBaseValue()).isEqualTo(Integer.valueOf(0));

      assertThat(top.toList(0)).isEmpty();
      assertThat(top.toList(1)).containsExactly(0).inOrder();
      assertThat(top.toList(2)).containsExactly(0, 1).inOrder();

      assertThat(((Function<Integer, ArrayList<Integer>>) top.toListSupplier()).apply(0)).isEmpty();
      assertThat(((Function<Integer, ArrayList<Integer>>) top.toListSupplier()).apply(1))
          .containsExactly(0)
          .inOrder();
      assertThat(((Function<Integer, ArrayList<Integer>>) top.toListSupplier()).apply(2))
          .containsExactly(0, 1)
          .inOrder();

      assertThat(top.convertToStringList(ImmutableList.of(0))).containsExactly("0").inOrder();
      assertThat(top.convertToStringList(ImmutableList.of(0, 1)))
          .containsExactly("0", "1")
          .inOrder();
    }
    {
      @SuppressWarnings("rawtypes")
      GenericDefaultInterfaceWithLambda testObject =
          new GenericDefaultInterfaceWithLambda.ClassThree();
      assertThat(testObject.getBaseValue()).isEqualTo(Long.valueOf(0));
      assertThat(testObject.increment(Long.valueOf(0))).isEqualTo(Long.valueOf(0 + 1));
      assertThat(testObject.toString(Long.valueOf(0))).isEqualTo(Long.valueOf(0).toString());
      assertThrows(ClassCastException.class, () -> testObject.increment(Integer.valueOf(0)));
      assertThrows(ClassCastException.class, () -> testObject.toString(Integer.valueOf(0)));
      assertThat(testObject.toList(2)).containsExactly(Long.valueOf(0), Long.valueOf(1)).inOrder();
      assertThat(testObject.convertToStringList(testObject.toList(1))).containsExactly("0");
      assertThat(((Function<Integer, ArrayList<Long>>) testObject.toListSupplier()).apply(2))
          .containsExactly(Long.valueOf(0), Long.valueOf(1));
    }
  }

  /**
   * Test for b/62047432.
   *
   * <p>When desugaring a functional interface with an executable clinit and default methods, we
   * erase the body of clinit to avoid executing it during desugaring. This test makes sure that all
   * the constants defined in the interface are still there after desugaring.
   */
  @Test
  public void testFunctionalInterfaceWithExecutableClinitStillWorkAfterDesugar() {
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.CONSTANT.length("").convert())
        .isEqualTo(0);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.CONSTANT.length("1").convert())
        .isEqualTo(1);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.BOOLEAN).isFalse();
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.CHAR).isEqualTo('h');
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.BYTE).isEqualTo(0);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.SHORT).isEqualTo(0);

    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.INT).isEqualTo(0);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.FLOAT).isEqualTo(0f);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.LONG).isEqualTo(0);
    assertThat(FunctionalInterfaceWithInitializerAndDefaultMethods.DOUBLE).isEqualTo(0d);
  }

  /** Test for b/38255926. */
  @Test
  public void testDefaultMethodInitializationOrder() {
    {
      assertThat(new DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetOne.C().sum())
          .isEqualTo(11); // To trigger loading the class C and its super interfaces.
      assertThat(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetOne
                  .getRealInitializationOrder())
          .isEqualTo(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetOne
                  .getExpectedInitializationOrder());
    }
    {
      assertThat(new DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetTwo.C().sum())
          .isEqualTo(3);
      assertThat(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetTwo
                  .getRealInitializationOrder())
          .isEqualTo(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetTwo
                  .getExpectedInitializationOrder());
    }
    {
      assertThat(new DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetThree.C().sum())
          .isEqualTo(11);
      assertThat(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetThree
                  .getRealInitializationOrder())
          .isEqualTo(
              DefaultInterfaceMethodWithStaticInitializer.TestInterfaceSetThree
                  .getExpectedInitializationOrder());
    }
  }

  /**
   * Tests that default methods on the classpath are correctly handled. We'll also verify the
   * metadata that's emitted for this case to make sure the binary-wide double-check for correct
   * desugaring of default and static interface methods keeps working (b/65645388).
   */
  @Test
  public void testDefaultMethodsInSeparateTarget() {
    assertThat(new DefaultMethodFromSeparateJava8Target().dflt()).isEqualTo("dflt");
    assertThat(new DefaultMethodTransitivelyFromSeparateJava8Target().dflt()).isEqualTo("dflt");
    assertThat(new DefaultMethodFromSeparateJava8TargetOverridden().dflt()).isEqualTo("override");
  }

  /** Regression test for b/73355452 */
  @Test
  public void testSuperCallToInheritedDefaultMethod() {
    assertThat(new InterfaceWithInheritedMethods.Impl().name()).isEqualTo("Base");
    assertThat(new InterfaceWithInheritedMethods.Impl().suffix()).isEqualTo("!");
  }
}
