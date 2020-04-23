/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typeannotation;

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.android.desugar.testing.junit.DesugarTestHelpers.filterInstructions;
import static org.objectweb.asm.tree.AbstractInsnNode.TYPE_INSN;

import com.google.devtools.build.android.desugar.testing.junit.AsmNode;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRule;
import com.google.devtools.build.android.desugar.testing.junit.DesugarRunner;
import com.google.devtools.build.android.desugar.testing.junit.JdkSuppress;
import com.google.devtools.build.android.desugar.testing.junit.JdkVersion;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Objects;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.objectweb.asm.tree.MethodNode;

/** Tests for desugaring annotations. */
@RunWith(DesugarRunner.class)
@JdkSuppress(minJdkVersion = JdkVersion.V11)
public class LocalTypeAnnotationUseTest {

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, MethodHandles.lookup())
          .addSourceInputsFromJvmFlag("input_srcs")
          .addJavacOptions("-source 11", "-target 11")
          .setWorkingJavaPackage("com.google.devtools.build.android.desugar.typeannotation")
          .addCommandOptions("desugar_try_with_resources_if_needed", "true")
          .build();

  @Test
  public void localNonTypeAnnotationsWithRuntimeRetention_discardedByJavac(
      @AsmNode(className = "AnnotationUser", memberName = "localNonTypeAnnotations", round = 0)
          MethodNode before) {
    assertThat(before.visibleLocalVariableAnnotations).isNull();
    assertThat(
            Arrays.stream(before.instructions.toArray())
                .map(insn -> insn.visibleTypeAnnotations)
                .allMatch(Objects::isNull))
        .isTrue();
  }

  @Test
  public void localVarWithTypeUseAnnotation(
      @AsmNode(
              className = "AnnotationUser",
              memberName = "localVarWithTypeUseAnnotation",
              round = 0)
          MethodNode before,
      @AsmNode(
              className = "AnnotationUser",
              memberName = "localVarWithTypeUseAnnotation",
              round = 1)
          MethodNode after) {
    assertThat(before.visibleLocalVariableAnnotations).isNotEmpty();
    assertThat(after.visibleLocalVariableAnnotations).isNull();
  }

  @Test
  public void instructionTypeUseAnnotation(
      @AsmNode(className = "AnnotationUser", memberName = "instructionTypeUseAnnotation", round = 0)
          MethodNode before,
      @AsmNode(className = "AnnotationUser", memberName = "instructionTypeUseAnnotation", round = 1)
          MethodNode after) {
    assertThat(getOnlyElement(filterInstructions(before, TYPE_INSN)).visibleTypeAnnotations)
        .isNotEmpty();
    assertThat(getOnlyElement(filterInstructions(after, TYPE_INSN)).visibleTypeAnnotations)
        .isNull();
  }

  @Test
  public void tryCatchTypeAnnotation(
      @AsmNode(className = "AnnotationUser", memberName = "tryCatchTypeAnnotation", round = 0)
          MethodNode before,
      @AsmNode(className = "AnnotationUser", memberName = "tryCatchTypeAnnotation", round = 1)
          MethodNode after) {
    assertThat(getOnlyElement(before.tryCatchBlocks).visibleTypeAnnotations).isNotEmpty();
    assertThat(getOnlyElement(after.tryCatchBlocks).visibleTypeAnnotations).isNull();
  }
}
