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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;

/** Static utilities that facilities desugar testing. */
public class DesugarTestHelpers {

  private DesugarTestHelpers() {}

  /**
   * A helper method that reads file paths into an array from the JVM flag value associated with
   * {@code jvmFlagKey}.
   */
  public static Path[] getRuntimePathsFromJvmFlag(String jvmFlagKey) {
    String jvmPropertyValue =
        checkNotNull(
            System.getProperty(jvmFlagKey),
            "Expected JVM Option to be specified: -D<%s>=<YourValue>",
            jvmFlagKey);
    return Splitter.on(" ").trimResults().splitToList(jvmPropertyValue).stream()
        .map(Paths::get)
        .toArray(Path[]::new);
  }

  /** Find all matched method invocation instructions in the given method. */
  public static ImmutableList<MethodInvocationSite> findMethodInvocationSites(
      MethodNode enclosingMethod,
      String methodOwnerRegex,
      String methodNameRegex,
      String methodDescRegex) {
    Predicate<String> methodOwnerPredicate = Pattern.compile(methodOwnerRegex).asPredicate();
    Predicate<String> methodNamePredicate = Pattern.compile(methodNameRegex).asPredicate();
    Predicate<String> methodDescPredicate = Pattern.compile(methodDescRegex).asPredicate();
    return filterInstructions(enclosingMethod, AbstractInsnNode.METHOD_INSN).stream()
        .map(node -> (MethodInsnNode) node)
        .filter(node -> methodOwnerPredicate.test(node.owner))
        .filter(node -> methodNamePredicate.test(node.name))
        .filter(node -> methodDescPredicate.test(node.desc))
        .map(
            node ->
                MethodInvocationSite.builder()
                    .setInvocationKind(MemberUseKind.fromValue(node.getOpcode()))
                    .setMethod(MethodKey.create(ClassName.create(node.owner), node.name, node.desc))
                    .setIsInterface(node.itf)
                    .build())
        .collect(toImmutableList());
  }

  public static ImmutableList<AbstractInsnNode> filterInstructions(
      MethodNode enclosingMethod, int instructionType) {
    return Arrays.stream(enclosingMethod.instructions.toArray())
        .filter(node -> node.getType() == instructionType)
        .collect(toImmutableList());
  }
}
