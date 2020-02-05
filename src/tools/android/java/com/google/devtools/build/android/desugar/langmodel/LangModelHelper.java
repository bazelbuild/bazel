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

package com.google.devtools.build.android.desugar.langmodel;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import java.util.Collection;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** A utility class for the desguaring of nest-based access control classes. */
public final class LangModelHelper {

  /**
   * The primitive type as specified at
   * https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-2.html#jvms-2.3
   */
  private static final ImmutableMap<Type, Type> PRIMITIVES_TO_BOXED_TYPES =
      ImmutableMap.<Type, Type>builder()
          .put(Type.INT_TYPE, Type.getObjectType("java/lang/Integer"))
          .put(Type.BOOLEAN_TYPE, Type.getObjectType("java/lang/Boolean"))
          .put(Type.BYTE_TYPE, Type.getObjectType("java/lang/Byte"))
          .put(Type.CHAR_TYPE, Type.getObjectType("java/lang/Character"))
          .put(Type.SHORT_TYPE, Type.getObjectType("java/lang/Short"))
          .put(Type.DOUBLE_TYPE, Type.getObjectType("java/lang/Double"))
          .put(Type.FLOAT_TYPE, Type.getObjectType("java/lang/Float"))
          .put(Type.LONG_TYPE, Type.getObjectType("java/lang/Long"))
          .build();

  /**
   * The lookup table for dup instructional opcodes. The row key is the count of words on stack top
   * to be duplicated. The column key is gap in word count between the original word section on
   * stack top and the post-duplicated word section underneath. See from
   *
   * <p>https://docs.oracle.com/javase/specs/jvms/se11/html/jvms-6.html#jvms-6.5.dup
   */
  private static final ImmutableTable<Integer, Integer, Integer> DIRECT_DUP_OPCODES =
      ImmutableTable.<Integer, Integer, Integer>builder()
          .put(1, 0, Opcodes.DUP)
          .put(2, 0, Opcodes.DUP2)
          .put(1, 1, Opcodes.DUP_X1)
          .put(2, 1, Opcodes.DUP2_X1)
          .put(1, 2, Opcodes.DUP_X2)
          .put(2, 2, Opcodes.DUP2_X2)
          .build();

  /** Whether the given type is a primitive type */
  public static boolean isPrimitive(Type type) {
    return PRIMITIVES_TO_BOXED_TYPES.containsKey(type);
  }

  public static Type toBoxedType(Type primitiveType) {
    return PRIMITIVES_TO_BOXED_TYPES.get(primitiveType);
  }

  /**
   * Returns the operation code for pop operations with a single instruction support by their type
   * sizes on stack top
   */
  public static int getTypeSizeAlignedPopOpcode(ImmutableList<Type> elementsToPop) {
    int totalWordsToPop = getTotalWords(elementsToPop);
    switch (totalWordsToPop) {
      case 1:
        return Opcodes.POP;
      case 2:
        return Opcodes.POP2;
      default:
        throw new IllegalStateException(
            String.format(
                "Expected either 1 or 2 words to be popped, but actually requested to pop (%d)"
                    + " words from <top/>%s...<bottom/>",
                totalWordsToPop, elementsToPop));
    }
  }

  /**
   * Returns the operation code for gap-free dup operations with a single instruction support by
   * their type sizes on stack top.
   */
  public static int getTypeSizeAlignedDupOpcode(ImmutableList<Type> elementsToDup) {
    return getTypeSizeAlignedDupOpcode(elementsToDup, ImmutableList.of());
  }

  /**
   * Returns the operation code for dup operations with a single instruction support by their type
   * sizes on stack top and the underneath gap size in words.
   */
  public static int getTypeSizeAlignedDupOpcode(
      ImmutableList<Type> elementsToDup, ImmutableList<Type> elementsToSkipBeforeInsertion) {
    int wordsToDup = getTotalWords(elementsToDup);
    int wordsToSkip = getTotalWords(elementsToSkipBeforeInsertion);
    Integer opCode = DIRECT_DUP_OPCODES.get(wordsToDup, wordsToSkip);
    if (opCode != null) {
      return opCode;
    }
    throw new IllegalStateException(
        String.format(
            "Expected either 1 or 2 words to be duplicated with a offset gap of {0, 1, 2} words,"
                + " but actually requested to duplicate (%d) words with an offset gap of (%d)"
                + " words, <top/>%s|%s...<bottom/>",
            wordsToDup, wordsToSkip, elementsToDup, elementsToSkipBeforeInsertion));
  }

  /**
   * Returns the word count summation of of the given type collection. It takes 1 word for category
   * 1 computational type, and 2 words for category 2 computational type.
   *
   * <p>https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-2.html#jvms-2.11.1
   */
  private static int getTotalWords(Collection<Type> types) {
    return types.stream().mapToInt(Type::getSize).sum();
  }

  /**
   * A checker on whether the give class is eligible as an inner class by its class internal name.
   *
   * <p>Note: The reliable source of truth is to check the InnerClasses attribute. However, the
   * attribute may have not been visited yet.
   */
  public static boolean isEligibleAsInnerClass(String className) {
    return className.contains("$");
  }

  /**
   * Whether the referenced class member is a in-nest distinct class access within the given
   * enclosing method.
   */
  public static boolean isCrossMateRefInNest(
      ClassMemberKey referencedMember, MethodKey enclosingMethod) {
    String enclosingClassName = enclosingMethod.owner();
    String referencedMemberName = referencedMember.owner();
    return (isEligibleAsInnerClass(enclosingClassName)
            || isEligibleAsInnerClass(referencedMemberName))
        && !referencedMemberName.equals(enclosingClassName);
  }

  /** Emits efficient instructions for a given integer push operation. */
  public static void visitPushInstr(MethodVisitor mv, final int value) {
    if (value >= -1 && value <= 5) {
      mv.visitInsn(Opcodes.ICONST_0 + value);
    } else if (value >= Byte.MIN_VALUE && value <= Byte.MAX_VALUE) {
      mv.visitIntInsn(Opcodes.BIPUSH, value);
    } else if (value >= Short.MIN_VALUE && value <= Short.MAX_VALUE) {
      mv.visitIntInsn(Opcodes.SIPUSH, value);
    } else {
      mv.visitLdcInsn(value);
    }
  }

  private LangModelHelper() {}
}
