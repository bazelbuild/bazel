// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.functional;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DynamicClassLiteral;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.nio.file.Paths;
import javax.inject.Inject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for accessing private constructors from another class within a nest. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class ConstructorInNestTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addInputs(Paths.get(System.getProperty("input_jar")))
          .setWorkingJavaPackage(
              "com.google.devtools.build.android.desugar.nest.functional.testsrc.simpleunit.constructor")
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .enableIterativeTransformation(3)
          .build();

  @Inject
  @DynamicClassLiteral("ConstructorNest$ConstructorServiceMate")
  private Class<?> mate;

  @Inject
  @DynamicClassLiteral("ConstructorNest$NestCC")
  private Class<?> companion;

  @Inject
  @DynamicClassLiteral("ConstructorNest")
  private Class<?> invoker;

  @Test
  public void companionClassIsPresent() {
    assertThat(companion).isNotNull();
  }

  @Test
  public void companionClassHierarchy() {
    assertThat(companion.getEnclosingClass()).isEqualTo(invoker);
    assertThat(companion.getEnclosingConstructor()).isNull();
    assertThat(companion.getEnclosingMethod()).isNull();
  }

  @Test
  public void companionClassModifiers() {
    assertThat(companion.isSynthetic()).isTrue();
    assertThat(companion.isMemberClass()).isTrue();
    assertThat(Modifier.isAbstract(companion.getModifiers())).isTrue();
    assertThat(Modifier.isStatic(companion.getModifiers())).isTrue();
  }

  @Test
  public void constructorBridgeGeneration() {
    assertThat(mate.getDeclaredConstructors()).hasLength(4);
  }

  @Test
  public void zeroArgConstructorBridge() throws Exception {
    Constructor<?> constructor = mate.getDeclaredConstructor(companion);
    assertThat(constructor.getModifiers() & 0x7).isEqualTo(0);
  }

  @Test
  public void multiArgConstructorBridge() throws Exception {
    Constructor<?> constructor = mate.getDeclaredConstructor(long.class, int.class, companion);

    assertThat(Modifier.isPublic(constructor.getModifiers())).isFalse();
    assertThat(Modifier.isPrivate(constructor.getModifiers())).isFalse();
    assertThat(Modifier.isProtected(constructor.getModifiers())).isFalse();
  }

  @Test
  public void createFromEmptyArgConstructor() throws Exception {
    assertThat(invoker.getDeclaredMethod("createFromZeroArgConstructor").invoke(null))
        .isEqualTo(30L);
  }

  @Test
  public void createFromMultiArgConstructor() throws Exception {
    assertThat(
            invoker
                .getDeclaredMethod("createFromMultiArgConstructor", long.class, int.class)
                .invoke(null, 20L, 30))
        .isEqualTo(50L);
  }
}
