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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.flags.Flag;
import com.google.common.flags.FlagSpec;
import com.google.common.flags.Flags;
import com.google.testing.junit.junit4.api.TestArgs;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.FieldNode;
import org.objectweb.asm.tree.MethodNode;

/** The test for {@link DesugarRule}. */
@RunWith(JUnit4.class)
public class DesugarRuleTest {

  @FlagSpec(help = "The input jar to be processed under desugar operations.", name = "input_jar")
  private static final Flag<String> inputJar = Flag.nullString();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .enableIterativeTransformation(3)
          .addInputs(Paths.get(inputJar.getNonNull()))
          .build();

  @LoadClass(
      "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$InterfaceSubjectToDesugar")
  private Class<?> interfaceSubjectToDesugarClassRound1;

  @LoadClass(
      value =
          "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$InterfaceSubjectToDesugar",
      round = 2)
  private Class<?> interfaceSubjectToDesugarClassRound2;

  @LoadClass(
      "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$InterfaceSubjectToDesugar$$CC")
  private Class<?> interfaceSubjectToDesugarCompanionClassRound1;

  @LoadClass(
      value =
          "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$InterfaceSubjectToDesugar$$CC",
      round = 2)
  private Class<?> interfaceSubjectToDesugarCompanionClassRound2;

  @LoadZipEntry(
      value =
          "com/google/devtools/build/android/desugar/testing/junit/DesugarRuleTestTarget$InterfaceSubjectToDesugar$$CC.class",
      round = 1)
  private ZipEntry interfaceSubjectToDesugarZipEntryRound1;

  @LoadZipEntry(
      value =
          "com/google/devtools/build/android/desugar/testing/junit/DesugarRuleTestTarget$InterfaceSubjectToDesugar$$CC.class",
      round = 2)
  private ZipEntry interfaceSubjectToDesugarZipEntryRound2;

  @LoadAsmNode(
      className = "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget")
  private ClassNode desugarRuleTestTargetClassNode;

  @LoadAsmNode(
      className =
          "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$Alpha",
      memberName = "twoIntSum")
  private MethodNode twoIntSum;

  @LoadAsmNode(
      className =
          "com.google.devtools.build.android.desugar.testing.junit.DesugarRuleTestTarget$Alpha",
      memberName = "multiplier",
      memberDescriptor = "J")
  private FieldNode multiplier;

  @BeforeClass
  public static void parseFlags() throws Exception {
    Flags.parse(TestArgs.get());
  }

  @Test
  public void staticMethodsAreMovedFromOriginatingClass() {
    assertThrows(
        NoSuchMethodException.class,
        () -> interfaceSubjectToDesugarClassRound1.getDeclaredMethod("staticMethod"));
  }

  @Test
  public void staticMethodsAreMovedFromOriginatingClass_desugarTwice() {
    assertThrows(
        NoSuchMethodException.class,
        () -> interfaceSubjectToDesugarClassRound2.getDeclaredMethod("staticMethod"));
  }

  @Test
  public void staticMethodsAreMovedToCompanionClass() {
    assertThat(
            Arrays.stream(interfaceSubjectToDesugarCompanionClassRound1.getDeclaredMethods())
                .map(Method::getName))
        .contains("staticMethod$$STATIC$$");
  }

  @Test
  public void nestMembers() {
    assertThat(desugarRuleTestTargetClassNode.nestMembers)
        .contains(
            "com/google/devtools/build/android/desugar/testing/junit/DesugarRuleTestTarget$InterfaceSubjectToDesugar");
  }

  @Test
  public void idempotencyOperation() {
    assertThat(interfaceSubjectToDesugarZipEntryRound1.getCrc())
        .isEqualTo(interfaceSubjectToDesugarZipEntryRound2.getCrc());
  }

  @Test
  public void classLoaders_sameInstanceInSameRound() {
    assertThat(interfaceSubjectToDesugarClassRound1.getClassLoader())
        .isSameInstanceAs(interfaceSubjectToDesugarCompanionClassRound1.getClassLoader());
    assertThat(interfaceSubjectToDesugarClassRound2.getClassLoader())
        .isSameInstanceAs(interfaceSubjectToDesugarCompanionClassRound2.getClassLoader());
  }

  @Test
  public void injectFieldNodes() {
    assertThat(twoIntSum.desc).isEqualTo("(II)I");
  }

  @Test
  public void injectMethodNodes() {
    assertThat(multiplier.desc).isEqualTo("J");
  }
}
