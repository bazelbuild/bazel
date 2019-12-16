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

import com.google.common.flags.Flag;
import com.google.common.flags.FlagSpec;
import com.google.common.flags.Flags;
import com.google.devtools.build.android.desugar.DesugarRule;
import com.google.devtools.build.android.desugar.DesugarRule.LoadClass;
import com.google.testing.junit.junit4.api.TestArgs;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for accessing private constructors from another class within a nest. */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class ConstructorInNestTest {

  @FlagSpec(
      help = "The root directory of source root directory for desugar testing.",
      name = "test_jar_constructor")
  private static final Flag<String> testJar = Flag.nullString();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addRuntimeInputs(testJar.getNonNull())
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .enableIterativeTransformation(3)
          .build();

  @LoadClass("simpleunit.constructor.ConstructorNest$ConstructorServiceMate")
  private Class<?> mate;

  @LoadClass("simpleunit.constructor.ConstructorNest$NestCC")
  private Class<?> companion;

  @LoadClass("simpleunit.constructor.ConstructorNest")
  private Class<?> invoker;

  @BeforeClass
  public static void setUpFlags() throws Exception {
    Flags.parse(TestArgs.get());
  }

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
