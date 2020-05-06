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

import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.TypePath;

/** Desugars bytecode instructions with references to type annotations. */
public class LocalTypeAnnotationUse extends ClassVisitor {

  public LocalTypeAnnotationUse(ClassVisitor classVisitor) {
    super(Opcodes.ASM8, classVisitor);
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null ? null : new LocalTypeAnnotationUseMethodVisitor(api, mv);
  }

  private static class LocalTypeAnnotationUseMethodVisitor extends MethodVisitor {

    LocalTypeAnnotationUseMethodVisitor(int api, MethodVisitor methodVisitor) {
      super(api, methodVisitor);
    }

    @Override
    public AnnotationVisitor visitLocalVariableAnnotation(
        int typeRef,
        TypePath typePath,
        Label[] start,
        Label[] end,
        int[] index,
        String descriptor,
        boolean visible) {
      // The visited annotation has to be a type annotation. A LOCAL_VARIABLE-targeted annotation
      // isn't retained in bytecode, regardless of retention policy. See details at,
      // https://docs.oracle.com/javase/specs/jls/se11/html/jls-9.html#jls-9.6.4.2
      return null;
    }

    @Override
    public AnnotationVisitor visitInsnAnnotation(
        int typeRef, TypePath typePath, String descriptor, boolean visible) {
      // An instruction annotation has to be a type annotation.
      return null;
    }

    @Override
    public AnnotationVisitor visitTryCatchAnnotation(
        int typeRef, TypePath typePath, String descriptor, boolean visible) {
      // An exception parameter annotation present in Javac-compiled bytecode has to be a type
      // annotation. See the JLS link in visitLocalVariableAnnotation of this class.
      return null;
    }
  }
}
