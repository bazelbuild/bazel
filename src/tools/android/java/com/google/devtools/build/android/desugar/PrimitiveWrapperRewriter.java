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

import static com.google.devtools.build.android.desugar.langmodel.ClassName.IN_PROCESS_LABEL_STRIPPER;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.util.concurrent.atomic.AtomicInteger;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/** This class rewrites (or removes) some trivial primitive wrapper methods. */
public class PrimitiveWrapperRewriter extends ClassVisitor {

  private static final ClassName PRIMITIVE_HASH_CODE_OWNER =
      ClassName.create("com/google/devtools/build/android/desugar/runtime/PrimitiveHashcode");

  private static final ImmutableMap<MethodInvocationSite, MethodInvocationSite>
      RUNTIME_LIB_IMPL_REPLACEMENTS =
          ImmutableMap.<MethodInvocationSite, MethodInvocationSite>builder()
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(
                              ClassName.create("java/lang/Integer"), "hashCode", "(I)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "identityAsHashCode", "(I)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(ClassName.create("java/lang/Short"), "hashCode", "(S)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "identityAsHashCode", "(S)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(ClassName.create("java/lang/Byte"), "hashCode", "(B)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "identityAsHashCode", "(B)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(
                              ClassName.create("java/lang/Character"), "hashCode", "(C)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "identityAsHashCode", "(C)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(ClassName.create("java/lang/Float"), "hashCode", "(F)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "hashCode", "(F)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(
                              ClassName.create("java/lang/Boolean"), "hashCode", "(Z)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "hashCode", "(Z)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(ClassName.create("java/lang/Long"), "hashCode", "(J)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "hashCode", "(J)I"))
                      .setIsInterface(false)
                      .build())
              .put(
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(
                          MethodKey.create(
                              ClassName.create("java/lang/Double"), "hashCode", "(D)I"))
                      .setIsInterface(false)
                      .build(),
                  MethodInvocationSite.builder()
                      .setInvocationKind(MemberUseKind.INVOKESTATIC)
                      .setMethod(MethodKey.create(PRIMITIVE_HASH_CODE_OWNER, "hashCode", "(D)I"))
                      .setIsInterface(false)
                      .build())
              .build();

  /**
   * The counter to record the times of desugar runtime library's hashcode implementation is
   * invoked.
   */
  private final AtomicInteger numOfPrimitiveHashCodeInvoked;

  public PrimitiveWrapperRewriter(ClassVisitor cv, AtomicInteger numOfPrimitiveHashCodeInvoked) {
    super(Opcodes.ASM8, cv);
    this.numOfPrimitiveHashCodeInvoked = numOfPrimitiveHashCodeInvoked;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null ? null : new PrimitiveWrapperMethodVisitor(visitor);
  }

  private class PrimitiveWrapperMethodVisitor extends MethodVisitor {

    PrimitiveWrapperMethodVisitor(MethodVisitor visitor) {
      super(Opcodes.ASM8, visitor);
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String desc, boolean isInterface) {
      MethodInvocationSite verbatimInvocationSite =
          MethodInvocationSite.create(opcode, owner, name, desc, isInterface)
              .acceptTypeMapper(IN_PROCESS_LABEL_STRIPPER);
      MethodInvocationSite replacementSite =
          RUNTIME_LIB_IMPL_REPLACEMENTS.get(verbatimInvocationSite);
      if (replacementSite != null) {
        if (replacementSite.name().equals("identityAsHashCode")) {
          return; // skip: use original primitive value as its hash (b/147139686)
        }
        numOfPrimitiveHashCodeInvoked.incrementAndGet();
        replacementSite.accept(this); // (b/147139686)
        return;
      }
      super.visitMethodInsn(opcode, owner, name, desc, isInterface);
    }
  }
}
