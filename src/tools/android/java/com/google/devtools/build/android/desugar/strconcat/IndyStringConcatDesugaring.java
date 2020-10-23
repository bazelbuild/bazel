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

package com.google.devtools.build.android.desugar.strconcat;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.android.desugar.langmodel.LangModelHelper.visitPushInstr;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberUseCounter;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodInvocationSite;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.util.Arrays;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** Desugars indy string concatenations by replacement with string builders. */
public final class IndyStringConcatDesugaring extends ClassVisitor {

  public static final MethodInvocationSite INVOKE_JDK11_STRING_CONCAT =
      MethodInvocationSite.builder()
          .setInvocationKind(MemberUseKind.INVOKEDYNAMIC)
          .setMethod(
              MethodKey.create(
                  ClassName.create("java/lang/invoke/StringConcatFactory"),
                  "makeConcatWithConstants",
                  "(Ljava/lang/invoke/MethodHandles$Lookup;"
                      + "Ljava/lang/String;"
                      + "Ljava/lang/invoke/MethodType;"
                      + "Ljava/lang/String;"
                      + "[Ljava/lang/Object;)"
                      + "Ljava/lang/invoke/CallSite;"))
          .setIsInterface(false)
          .build();

  private static final MethodInvocationSite INVOKE_STRING_CONCAT_REPLACEMENT_METHOD =
      MethodInvocationSite.builder()
          .setInvocationKind(MemberUseKind.INVOKESTATIC)
          .setMethod(
              MethodKey.create(
                  ClassName.create(
                      "com/google/devtools/build/android/desugar/runtime/StringConcats"),
                  "concat",
                  "([Ljava/lang/Object;"
                      + "Ljava/lang/String;"
                      + "[Ljava/lang/Object;)"
                      + "Ljava/lang/String;"))
          .setIsInterface(false)
          .build();

  private final ClassMemberUseCounter classMemberUseCounter;

  public IndyStringConcatDesugaring(
      ClassMemberUseCounter classMemberUseCounter, ClassVisitor classVisitor) {
    super(Opcodes.ASM8, classVisitor);
    this.classMemberUseCounter = classMemberUseCounter;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null
        ? null
        : new IndifiedStringConcatInvocationConverter(classMemberUseCounter, api, mv);
  }

  static final class IndifiedStringConcatInvocationConverter extends MethodVisitor {

    private final ClassMemberUseCounter classMemberUseCounter;

    IndifiedStringConcatInvocationConverter(
        ClassMemberUseCounter classMemberUseCounter, int api, MethodVisitor methodVisitor) {
      super(api, methodVisitor);
      this.classMemberUseCounter = classMemberUseCounter;
    }

    @Override
    public void visitInvokeDynamicInsn(
        String name,
        String descriptor,
        Handle bootstrapMethodHandle,
        Object... bootstrapMethodArguments) {
      MethodInvocationSite bootstrapMethodInvocation =
          MethodInvocationSite.builder()
              .setInvocationKind(MemberUseKind.INVOKEDYNAMIC)
              .setMethod(
                  MethodKey.create(
                      ClassName.create(bootstrapMethodHandle.getOwner()),
                      bootstrapMethodHandle.getName(),
                      bootstrapMethodHandle.getDesc()))
              .setIsInterface(false)
              .build();
      if (INVOKE_JDK11_STRING_CONCAT.equals(bootstrapMethodInvocation)) {
        // Increment the counter for the bootstrap method invocation of
        // StringConcatFactory#makeConcatWithConstants
        classMemberUseCounter.incrementMemberUseCount(bootstrapMethodInvocation);

        LangModelHelper.collapseStackValuesToObjectArray(
            this,
            ImmutableList.of(LangModelHelper::anyPrimitiveToStringInvocationSite),
            Arrays.stream(Type.getArgumentTypes(descriptor))
                .map(ClassName::create)
                .collect(toImmutableList()),
            ImmutableList.of());

        String recipe = (String) bootstrapMethodArguments[0];
        visitLdcInsn(recipe);

        // Stores the constants into an array.
        visitPushInstr(mv, bootstrapMethodArguments.length - 1);
        visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/Object");
        for (int i = 1; i < bootstrapMethodArguments.length; i++) {
          visitPushInstr(mv, i - 1);
          visitLdcInsn(bootstrapMethodArguments[i]);
          visitInsn(Opcodes.AASTORE);
        }

        INVOKE_STRING_CONCAT_REPLACEMENT_METHOD.accept(mv);
        return;
      }
      super.visitInvokeDynamicInsn(
          name, descriptor, bootstrapMethodHandle, bootstrapMethodArguments);
    }
  }
}
