// Copyright 2018 The Bazel Authors. All rights reserved.
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

import java.io.FileOutputStream;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;
import org.objectweb.asm.AnnotationVisitor;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ByteVector;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * GenKeepForCompile creates a jarfile containing a class definition that has the KeepForCOmpile
 * attribute.
 */
public class GenKeepForCompile implements Opcodes {

  public static void main(String[] args) throws Exception {
    try (JarOutputStream jos = new JarOutputStream(new FileOutputStream(args[0]))) {
      jos.putNextEntry(new ZipEntry("functions/car/CarInlineUtilsKt.class"));
      jos.write(dump());
      jos.closeEntry();
    }
  }

  public static byte[] dump() throws Exception {

    ClassWriter classWriter = new ClassWriter(0);
    MethodVisitor methodVisitor;
    AnnotationVisitor annotationVisitor0;

    classWriter.visit(
        V1_8,
        ACC_PUBLIC | ACC_FINAL | ACC_SUPER,
        "functions/car/CarInlineUtilsKt",
        null,
        "java/lang/Object",
        null);

    classWriter.visitSource("CarInlineUtils.kt", null);

    {
      annotationVisitor0 = classWriter.visitAnnotation("Lkotlin/Metadata;", true);
      annotationVisitor0.visit("mv", new int[] {1, 1, 11});
      annotationVisitor0.visit("bv", new int[] {1, 0, 2});
      annotationVisitor0.visit("k", Integer.valueOf(2));
      {
        AnnotationVisitor annotationVisitor1 = annotationVisitor0.visitArray("d1");
        annotationVisitor1.visit(
            null,
            "\u0000\u0014\n\u0000\n\u0002\u0010\u000e\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\u0010\u0008\u001a!\u0010\u0000\u001a\u00020\u0001*\u00020\u00022\u0012\u0010\u0003\u001a\u000e\u0012\u0004\u0012\u00020\u0005\u0012\u0004\u0012\u00020\u00010\u0004H\u0086\u0008");
        annotationVisitor1.visitEnd();
      }
      {
        AnnotationVisitor annotationVisitor1 = annotationVisitor0.visitArray("d2");
        annotationVisitor1.visit(null, "customName");
        annotationVisitor1.visit(null, "");
        annotationVisitor1.visit(null, "Lfunctions/car/Car;");
        annotationVisitor1.visit(null, "formatYear");
        annotationVisitor1.visit(null, "Lkotlin/Function1;");
        annotationVisitor1.visit(null, "");
        annotationVisitor1.visitEnd();
      }
      annotationVisitor0.visitEnd();
    }
    { // Private method to make sure ijar doesn't ignore those
      methodVisitor =
          classWriter.visitMethod(
              ACC_PRIVATE | ACC_FINAL | ACC_STATIC,
              "customName",
              "(Lfunctions/car/Car;Lkotlin/jvm/functions/Function1;)Ljava/lang/String;",
              "(Lfunctions/car/Car;Lkotlin/jvm/functions/Function1<-Ljava/lang/Integer;Ljava/lang/String;>;)Ljava/lang/String;",
              null);
      methodVisitor.visitParameter("$receiver", ACC_MANDATED);
      methodVisitor.visitParameter("formatYear", 0);
      {
        annotationVisitor0 =
            methodVisitor.visitAnnotation("Lorg/jetbrains/annotations/NotNull;", false);
        annotationVisitor0.visitEnd();
      }
      methodVisitor.visitAnnotableParameterCount(2, false);
      {
        annotationVisitor0 =
            methodVisitor.visitParameterAnnotation(0, "Lorg/jetbrains/annotations/NotNull;", false);
        annotationVisitor0.visitEnd();
      }
      {
        annotationVisitor0 =
            methodVisitor.visitParameterAnnotation(1, "Lorg/jetbrains/annotations/NotNull;", false);
        annotationVisitor0.visitEnd();
      }
      // ATTRIBUTE com.google.devtools.ijar.KeepForCompile
      // ASM-ifier doesn't emit attributes it does not know about :( - emitting by hand here.
      methodVisitor.visitAttribute(
          new Attribute("com.google.devtools.ijar.KeepForCompile") {
            @Override
            public ByteVector write(
                ClassWriter cw, byte[] code, int len, int maxStack, int maxLocals) {
              // nothing to write, just get the attribute's name in the file.
              return new ByteVector();
            }
          });

      methodVisitor.visitCode();
      Label label0 = new Label();
      methodVisitor.visitLabel(label0);
      methodVisitor.visitVarInsn(ALOAD, 0);
      methodVisitor.visitLdcInsn("$receiver");
      methodVisitor.visitMethodInsn(
          INVOKESTATIC,
          "kotlin/jvm/internal/Intrinsics",
          "checkParameterIsNotNull",
          "(Ljava/lang/Object;Ljava/lang/String;)V",
          false);
      methodVisitor.visitVarInsn(ALOAD, 1);
      methodVisitor.visitLdcInsn("formatYear");
      methodVisitor.visitMethodInsn(
          INVOKESTATIC,
          "kotlin/jvm/internal/Intrinsics",
          "checkParameterIsNotNull",
          "(Ljava/lang/Object;Ljava/lang/String;)V",
          false);
      Label label1 = new Label();
      methodVisitor.visitLabel(label1);
      methodVisitor.visitLineNumber(3, label1);
      methodVisitor.visitTypeInsn(NEW, "java/lang/StringBuilder");
      methodVisitor.visitInsn(DUP);
      methodVisitor.visitMethodInsn(
          INVOKESPECIAL, "java/lang/StringBuilder", "<init>", "()V", false);
      methodVisitor.visitVarInsn(ALOAD, 1);
      methodVisitor.visitVarInsn(ALOAD, 0);
      methodVisitor.visitMethodInsn(INVOKEVIRTUAL, "functions/car/Car", "getYear", "()I", false);
      methodVisitor.visitMethodInsn(
          INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", false);
      methodVisitor.visitMethodInsn(
          INVOKEINTERFACE,
          "kotlin/jvm/functions/Function1",
          "invoke",
          "(Ljava/lang/Object;)Ljava/lang/Object;",
          true);
      methodVisitor.visitTypeInsn(CHECKCAST, "java/lang/String");
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
          false);
      methodVisitor.visitIntInsn(BIPUSH, 32);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(C)Ljava/lang/StringBuilder;",
          false);
      methodVisitor.visitVarInsn(ALOAD, 0);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL, "functions/car/Car", "getMake", "()Ljava/lang/String;", false);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
          false);
      methodVisitor.visitIntInsn(BIPUSH, 32);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(C)Ljava/lang/StringBuilder;",
          false);
      methodVisitor.visitVarInsn(ALOAD, 0);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL, "functions/car/Car", "getModel", "()Ljava/lang/String;", false);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
          false);
      methodVisitor.visitMethodInsn(
          INVOKEVIRTUAL, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;", false);
      methodVisitor.visitInsn(ARETURN);
      Label label2 = new Label();
      methodVisitor.visitLabel(label2);
      methodVisitor.visitLocalVariable("$receiver", "Lfunctions/car/Car;", null, label0, label2, 0);
      methodVisitor.visitLocalVariable(
          "formatYear", "Lkotlin/jvm/functions/Function1;", null, label0, label2, 1);
      methodVisitor.visitLocalVariable("$i$f$customName", "I", null, label0, label2, 2);
      methodVisitor.visitMaxs(3, 3);
      methodVisitor.visitEnd();
    }
    classWriter.visitEnd();

    return classWriter.toByteArray();
  }
}
