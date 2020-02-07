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

package com.google.devtools.build.android.desugar.nest;

import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;
import static org.objectweb.asm.Opcodes.GETSTATIC;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.langmodel.FieldInstrVisitor;
import com.google.devtools.build.android.desugar.langmodel.FieldKey;
import com.google.devtools.build.android.desugar.langmodel.LangModelHelper;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** A visitor class that emits bridge methods for private field accesses. */
final class FieldAccessBridgeEmitter
    implements FieldInstrVisitor<MethodVisitor, FieldKey, ClassVisitor> {

  /** Emits a bridge method for a field with a {@link Opcodes.GETSTATIC} access. */
  @Override
  public MethodVisitor visitGetStatic(FieldKey fieldKey, ClassVisitor cv) {
    MethodKey bridgeMethodKey = fieldKey.bridgeOfStaticRead();

    MethodVisitor mv =
        cv.visitMethod(
            ACC_SYNTHETIC | ACC_STATIC,
            bridgeMethodKey.name(),
            bridgeMethodKey.descriptor(),
            /* signature= */ null,
            /* exceptions= */ null);

    mv.visitFieldInsn(GETSTATIC, fieldKey.ownerName(), fieldKey.name(), fieldKey.descriptor());
    Type fieldType = fieldKey.getFieldType();
    mv.visitInsn(fieldType.getOpcode(Opcodes.IRETURN));
    int fieldTypeSize = fieldType.getSize();
    mv.visitMaxs(fieldTypeSize, fieldTypeSize);
    mv.visitEnd();
    return mv;
  }

  /** Emits a bridge method for a field with a {@link Opcodes.PUTSTATIC} access. */
  @Override
  public MethodVisitor visitPutStatic(FieldKey fieldKey, ClassVisitor cv) {
    MethodKey bridgeMethodKey = fieldKey.bridgeOfStaticWrite();
    MethodVisitor mv =
        cv.visitMethod(
            ACC_SYNTHETIC | ACC_STATIC,
            bridgeMethodKey.name(),
            bridgeMethodKey.descriptor(),
            /* signature= */ null,
            /* exceptions= */ null);
    mv.visitCode();
    Type fieldType = fieldKey.getFieldType();
    mv.visitVarInsn(fieldType.getOpcode(Opcodes.ILOAD), 0);
    mv.visitInsn(
        LangModelHelper.getTypeSizeAlignedDupOpcode(ImmutableList.of(fieldKey.getFieldType())));
    mv.visitFieldInsn(
        Opcodes.PUTSTATIC, fieldKey.ownerName(), fieldKey.name(), fieldKey.descriptor());
    mv.visitInsn(fieldType.getOpcode(Opcodes.IRETURN));
    int fieldTypeSize = fieldType.getSize();
    mv.visitMaxs(fieldTypeSize, fieldTypeSize);
    mv.visitEnd();
    return mv;
  }

  /** Emits a bridge method for a field with a {@link Opcodes.PUTFIELD} access. */
  @Override
  public MethodVisitor visitGetField(FieldKey fieldKey, ClassVisitor cv) {
    MethodKey bridgeMethodKey = fieldKey.bridgeOfInstanceRead();
    MethodVisitor mv =
        cv.visitMethod(
            ACC_SYNTHETIC | ACC_STATIC,
            bridgeMethodKey.name(),
            bridgeMethodKey.descriptor(),
            /* signature= */ null,
            /* exceptions= */ null);
    mv.visitCode();
    mv.visitVarInsn(Opcodes.ALOAD, 0);
    mv.visitFieldInsn(
        Opcodes.GETFIELD, fieldKey.ownerName(), fieldKey.name(), fieldKey.descriptor());
    Type fieldType = fieldKey.getFieldType();
    mv.visitInsn(fieldType.getOpcode(Opcodes.IRETURN));
    int fieldTypeSize = fieldType.getSize();
    mv.visitMaxs(fieldTypeSize, fieldTypeSize);
    mv.visitEnd();
    return mv;
  }

  /** Emits a bridge method for a field with a {@link Opcodes.PUTFIELD} access. */
  @Override
  public MethodVisitor visitPutField(FieldKey fieldKey, ClassVisitor cv) {
    MethodKey bridgeMethodKey = fieldKey.bridgeOfInstanceWrite();
    MethodVisitor mv =
        cv.visitMethod(
            ACC_SYNTHETIC | ACC_STATIC,
            bridgeMethodKey.name(),
            bridgeMethodKey.descriptor(),
            /* signature= */ null,
            /* exceptions= */ null);
    mv.visitCode();
    mv.visitVarInsn(Opcodes.ALOAD, 0);
    Type fieldType = fieldKey.getFieldType();
    mv.visitVarInsn(fieldType.getOpcode(Opcodes.ILOAD), 1);
    mv.visitInsn(
        LangModelHelper.getTypeSizeAlignedDupOpcode(
            ImmutableList.of(fieldKey.getFieldType()),
            ImmutableList.of(Type.getType(Object.class))));
    mv.visitFieldInsn(
        Opcodes.PUTFIELD, fieldKey.ownerName(), fieldKey.name(), fieldKey.descriptor());
    mv.visitInsn(fieldType.getOpcode(Opcodes.IRETURN));
    int fieldTypeSize = fieldType.getSize();
    mv.visitMaxs(fieldTypeSize, fieldTypeSize);
    mv.visitEnd();
    return mv;
  }
}
