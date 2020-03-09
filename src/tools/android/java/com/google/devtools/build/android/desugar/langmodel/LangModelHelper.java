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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Streams;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** A utility class for the desguaring of nest-based access control classes. */
public final class LangModelHelper {

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
      ClassMemberKey<?> referencedMember, MethodKey enclosingMethod) {
    String enclosingClassName = enclosingMethod.ownerName();
    String referencedMemberName = referencedMember.ownerName();
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

  /**
   * Emits bytecode to allocate a new object array, stores the bottom values in the operand stack to
   * the new array, and replaces the bottom values with a reference to the newly-allocated array.
   *
   * <p>Operand Stack:
   * <li>Before instructions: [Stack Top]..., value_n, value_n-1, ..., value_2, value_1, value_0
   * <li>After instructions: [Stack Top] ..., value_n, arrayref
   *
   *     <p>where n is the size of {@code expectedTypesOnOperandStack} and is expected to be equal
   *     to array length referenced by arrayref
   *
   * @param mv The current method visitor that is visiting the class.
   * @param mappers Applies to an operand stack value if tested positive on {@code filter}.
   * @param expectedTypesOnOperandStack The expected types at the bottom of the operand stack. The
   *     end of the list corresponds to the the bottom of the operand stack.
   */
  public static ImmutableList<ClassName> collapseStackValuesToObjectArray(
      MethodVisitor mv,
      ImmutableList<Function<ClassName, Optional<MethodInvocationSite>>> mappers,
      ImmutableList<ClassName> expectedTypesOnOperandStack) {
    // Creates an array of java/lang/Object to store the values on top of the operand stack that
    // are subject to string concatenation.
    int numOfValuesOnOperandStack = expectedTypesOnOperandStack.size();
    visitPushInstr(mv, numOfValuesOnOperandStack);
    mv.visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/Object");

    // To preserve the order of the operands to be string-concatenated, we slot the values on
    // the top of the stack to the end of the array.
    List<ClassName> actualTypesOnObjectArray = new ArrayList<>(expectedTypesOnOperandStack);
    for (int i = numOfValuesOnOperandStack - 1; i >= 0; i--) {
      ClassName operandTypeName = expectedTypesOnOperandStack.get(i);
      Type operandType = operandTypeName.toAsmObjectType();
      // Pre-duplicates the array reference for next loop iteration use.
      // Post-operation stack bottom to top:
      //     ..., value_i-1, arrayref, value_i, arrayref.
      mv.visitInsn(
          getTypeSizeAlignedDupOpcode(
              ImmutableList.of(Type.getType(Object.class)), ImmutableList.of(operandType)));

      // Pushes the array index and adjusts the order of the values on stack top in the order
      // of <bottom/> arrayref, index, value <top/> before emitting an aastore instruction.
      // https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-6.html#jvms-6.5.aastore
      // Post-operation stack bottom to top:
      //     ..., value_i-1, arrayref, value_i, arrayref, i.
      visitPushInstr(mv, i);
      // Cross-duplicates the array reference and index.
      // Post-operation stack bottom to top:
      //     ..., value_i-1, arrayref, arrayref, i, value_i, arrayref, i.
      mv.visitInsn(
          getTypeSizeAlignedDupOpcode(
              ImmutableList.of(Type.getType(Object.class), Type.getType(int.class)),
              ImmutableList.of(operandType)));

      // Pops arrayref, index, leaving the stack top as value_i.
      // Post-operation stack bottom to top:
      //     ..., value_i-1, arrayref, arrayref, i, value_i.
      mv.visitInsn(
          getTypeSizeAlignedPopOpcode(
              ImmutableList.of(Type.getType(Object.class), Type.getType(int.class))));

      int targetArrayIndex = i;
      mappers.stream()
          .map(mapper -> mapper.apply(actualTypesOnObjectArray.get(targetArrayIndex)))
          .flatMap(Streams::stream)
          .forEach(
              typeConversionSite -> {
                typeConversionSite.accept(mv);
                actualTypesOnObjectArray.set(targetArrayIndex, typeConversionSite.returnTypeName());
              });

      // Post-operation stack bottom to top:
      //     ..., value_i-1, arrayref.
      mv.visitInsn(Opcodes.AASTORE);
    }
    return ImmutableList.copyOf(actualTypesOnObjectArray);
  }

  /**
   * Emits bytecode to replace the object array reference at the operand stack bottom with its array
   * element values.
   *
   * <p>Operand Stack:
   * <li>Before instructions: [Stack Top] ..., value_n, arrayref
   * <li>After instructions: [Stack Top]..., value_n, value_n-1, ..., value_2, value_1, value_0
   *
   *     <p>where n is the array length referenced by arrayref and is expected to be equal to the
   *     size of {@code expectedTypesOnOperandStack} expanded on the operand stack.
   *
   * @param mv The current method visitor that is visiting the class.
   * @param expectedTypesOnOperandStack The expected types at the bottom of the operand stack. The
   *     end of the list corresponds to the the bottom of the operand stack.
   */
  public static void expandObjectArrayToStackValues(
      MethodVisitor mv, ImmutableList<ClassName> expectedTypesOnOperandStack) {
    int numOfValuesExpandedOnOperandStack = expectedTypesOnOperandStack.size();
    for (int i = 0; i < numOfValuesExpandedOnOperandStack; i++) {
      ClassName operandTypeName = expectedTypesOnOperandStack.get(i);
      // Pre-duplicates the array reference for next loop iteration use.
      // Post-operation stack bottom to top:
      //     ..., arrayref, arrayref
      mv.visitInsn(Opcodes.DUP);

      // Pushes the current array index on stack.
      // Post-operation stack bottom to top:
      //     ..., arrayref, arrayref, i
      visitPushInstr(mv, i);

      // Post-operation stack bottom to top:
      //     ..., arrayref, obj_value_i
      mv.visitInsn(Opcodes.AALOAD);

      // Post-operation stack bottom to top:
      //     ..., arrayref, cast_and_unboxed_value_i
      if (operandTypeName.isPrimitive()) {
        ClassName boxedTypeName = operandTypeName.toBoxedType();
        mv.visitTypeInsn(Opcodes.CHECKCAST, boxedTypeName.binaryName());
        createBoxedTypeToPrimitiveInvocationSite(boxedTypeName).accept(mv);
      } else if (!ClassName.create(Object.class).equals(operandTypeName)) {
        mv.visitTypeInsn(Opcodes.CHECKCAST, operandTypeName.binaryName());
      }

      //     ..., cast_and_unboxed_value_i, arrayref
      if (operandTypeName.isWideType()) {
        mv.visitInsn(Opcodes.DUP2_X1);
        mv.visitInsn(Opcodes.POP2);
      } else {
        mv.visitInsn(Opcodes.SWAP);
      }
    }

    // pops out the original arrayref.
    mv.visitInsn(Opcodes.POP);
  }

  public static Optional<MethodInvocationSite> anyPrimitiveToStringInvocationSite(
      ClassName className) {
    return className.isPrimitive()
        ? Optional.of(createPrimitiveToStringInvocationSite(className))
        : Optional.empty();
  }

  /** Convenient factory method for converting a primitive type to string call site. */
  private static MethodInvocationSite createPrimitiveToStringInvocationSite(
      ClassName primitiveTypeName) {
    checkArgument(
        primitiveTypeName.isPrimitive(),
        "Expected a primitive type for a type boxing call site, but got %s",
        primitiveTypeName);
    return MethodInvocationSite.builder()
        .setInvocationKind(MemberUseKind.INVOKESTATIC)
        .setMethod(
            MethodKey.create(
                primitiveTypeName.toBoxedType(),
                "toString",
                Type.getMethodDescriptor(
                    Type.getType(String.class), primitiveTypeName.toAsmObjectType())))
        .setIsInterface(false)
        .build();
  }

  /** Convenient factory method for converting a primitive type to string call site. */
  public static Optional<MethodInvocationSite> anyPrimitiveToBoxedTypeInvocationSite(
      ClassName className) {
    return className.isPrimitive()
        ? Optional.of(createPrimitiveToBoxedTypeInvocationSite(className))
        : Optional.empty();
  }

  private static MethodInvocationSite createPrimitiveToBoxedTypeInvocationSite(
      ClassName primitiveTypeName) {
    checkArgument(
        primitiveTypeName.isPrimitive(),
        "Expected a primitive type for a type boxing call site, but got %s",
        primitiveTypeName);
    return MethodInvocationSite.builder()
        .setInvocationKind(MemberUseKind.INVOKESTATIC)
        .setMethod(
            MethodKey.create(
                primitiveTypeName.toBoxedType(),
                "valueOf",
                Type.getMethodDescriptor(
                    primitiveTypeName.toBoxedType().toAsmObjectType(),
                    primitiveTypeName.toAsmObjectType())))
        .setIsInterface(false)
        .build();
  }

  private static MethodInvocationSite createBoxedTypeToPrimitiveInvocationSite(
      ClassName boxedType) {
    String boxedTypeBinaryName = boxedType.binaryName();
    final MethodKey typeUnboxingMethod;
    if ("java/lang/Boolean".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "booleanValue", "()Z");
    } else if ("java/lang/Character".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "charValue", "()C");
    } else if ("java/lang/Byte".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "byteValue", "()B");
    } else if ("java/lang/Short".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "shortValue", "()S");
    } else if ("java/lang/Integer".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "intValue", "()I");
    } else if ("java/lang/Float".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "floatValue", "()F");
    } else if ("java/lang/Long".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "longValue", "()J");
    } else if ("java/lang/Double".contentEquals(boxedTypeBinaryName)) {
      typeUnboxingMethod = MethodKey.create(boxedType, "doubleValue", "()D");
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Expected a boxed type to create a type boxing call site, but got %s", boxedType));
    }
    return MethodInvocationSite.builder()
        .setInvocationKind(MemberUseKind.INVOKEVIRTUAL)
        .setMethod(typeUnboxingMethod)
        .setIsInterface(false)
        .build();
  }

  private LangModelHelper() {}
}
