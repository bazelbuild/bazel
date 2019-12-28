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

import static com.google.devtools.build.android.desugar.langmodel.LangModelHelper.getTypeSizeAlignedOpcode;
import static com.google.devtools.build.android.desugar.langmodel.LangModelHelper.isPrimitive;
import static com.google.devtools.build.android.desugar.langmodel.LangModelHelper.toBoxedType;
import static com.google.devtools.build.android.desugar.langmodel.LangModelHelper.visitPushInstr;

import com.google.devtools.build.android.desugar.langmodel.ClassMemberUse;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberUseCounter;
import com.google.devtools.build.android.desugar.langmodel.MemberUseKind;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** Desugars indified string concatenations to use string builders. */
public final class IndyStringConcatDesugaring extends ClassVisitor {

  public static final ClassMemberUse INVOKE_JDK11_STRING_CONCAT =
      ClassMemberUse.create(
          MethodKey.create(
              "java/lang/invoke/StringConcatFactory",
              "makeConcatWithConstants",
              "(Ljava/lang/invoke/MethodHandles$Lookup;"
                  + "Ljava/lang/String;"
                  + "Ljava/lang/invoke/MethodType;"
                  + "Ljava/lang/String;"
                  + "[Ljava/lang/Object;)"
                  + "Ljava/lang/invoke/CallSite;"),
          MemberUseKind.INVOKEDYNAMIC);

  private static final ClassMemberUse INVOKE_STRING_CONCAT_REPLACEMENT_METHOD =
      ClassMemberUse.create(
          MethodKey.create(
              "com/google/devtools/build/android/desugar/runtime/StringConcats",
              "concat",
              "([Ljava/lang/Object;"
                  + "Ljava/lang/String;"
                  + "[Ljava/lang/Object;)"
                  + "Ljava/lang/String;"),
          MemberUseKind.INVOKESTATIC);

  private final ClassMemberUseCounter classMemberUseCounter;

  public IndyStringConcatDesugaring(
      ClassMemberUseCounter classMemberUseCounter, ClassVisitor classVisitor) {
    super(Opcodes.ASM7, classVisitor);
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
      ClassMemberUse bootstrapMethodInvocation =
          ClassMemberUse.create(
              MethodKey.create(
                  bootstrapMethodHandle.getOwner(),
                  bootstrapMethodHandle.getName(),
                  bootstrapMethodHandle.getDesc()),
              MemberUseKind.INVOKEDYNAMIC);
      if (INVOKE_JDK11_STRING_CONCAT.equals(bootstrapMethodInvocation)) {
        // Increment the counter for the bootstrap method invocation of
        // StringConcatFactory#makeConcatWithConstants
        classMemberUseCounter.incrementMemberUseCount(bootstrapMethodInvocation);

        // Creates an array of java/lang/Object to store the values on top of the operand stack that
        // are subject to string concatenation.
        Type[] typesOfValuesOnOperandStack = Type.getArgumentTypes(descriptor);
        int numOfValuesOnOperandStack = typesOfValuesOnOperandStack.length;
        visitPushInstr(mv, numOfValuesOnOperandStack);
        visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/Object");

        // To preserve the order of the operands to be string-concatenated, we slot the values on
        // the top of the stack to the end of the array.
        for (int i = numOfValuesOnOperandStack - 1; i >= 0; i--) {
          Type operandType = typesOfValuesOnOperandStack[i];
          // Pre-duplicates the array reference for next loop iteration use.
          // Post-operation stack bottom to top:
          //     ..., value_i-1, arrayref, value_i, arrayref.
          visitInsn(getTypeSizeAlignedOpcode(Opcodes.DUP_X1, operandType));

          // Pushes the array index and adjusts the order of the values on stack top in the order
          // of <bottom/> arrayref, index, value <top/> before emitting an aastore instruction.
          // https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-6.html#jvms-6.5.aastore
          // Post-operation stack bottom to top:
          //     ..., value_i-1, arrayref, value_i, arrayref, i.
          visitPushInstr(mv, i);
          // Cross-duplicates the array reference and index.
          // Post-operation stack bottom to top:
          //     ..., value_i-1, arrayref, arrayref, i, value_i, arrayref, i.
          visitInsn(getTypeSizeAlignedOpcode(Opcodes.DUP2_X1, operandType));
          // Pops arrayref, index, leaving the stack top as value_i.
          // Post-operation stack bottom to top:
          //     ..., value_i-1, arrayref, arrayref, i, value_i.
          visitInsn(Opcodes.POP2);

          if (isPrimitive(operandType)) {
            // Explicitly computes the string value of primitive types, so that they can be stored
            // in the Object[] array.
            // Post-operation stack bottom to top:
            //     ..., value_i-1, arrayref, arrayref, i, processed_value_i.
            Type boxedType = toBoxedType(operandType);
            visitMethodInsn(
                Opcodes.INVOKESTATIC,
                boxedType.getInternalName(),
                "toString",
                Type.getMethodDescriptor(Type.getType(String.class), operandType),
                /* isInterface= */ false);
          }
          // Post-operation stack bottom to top:
          //     ..., value_i-1, arrayref.
          visitInsn(Opcodes.AASTORE);
        }
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

        INVOKE_STRING_CONCAT_REPLACEMENT_METHOD.acceptClassMethodInsn(this);
        return;
      }
      super.visitInvokeDynamicInsn(
          name, descriptor, bootstrapMethodHandle, bootstrapMethodArguments);
    }
  }
}
