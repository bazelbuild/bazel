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

import static com.google.common.base.Preconditions.checkArgument;
import static org.objectweb.asm.Opcodes.ACC_ABSTRACT;
import static org.objectweb.asm.Opcodes.ACC_FINAL;
import static org.objectweb.asm.Opcodes.ACC_INTERFACE;
import static org.objectweb.asm.Opcodes.ACC_PRIVATE;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SUPER;
import static org.objectweb.asm.Opcodes.ACONST_NULL;
import static org.objectweb.asm.Opcodes.ALOAD;
import static org.objectweb.asm.Opcodes.ARETURN;
import static org.objectweb.asm.Opcodes.ASTORE;
import static org.objectweb.asm.Opcodes.BIPUSH;
import static org.objectweb.asm.Opcodes.DCONST_0;
import static org.objectweb.asm.Opcodes.DLOAD;
import static org.objectweb.asm.Opcodes.DUP_X1;
import static org.objectweb.asm.Opcodes.FCONST_0;
import static org.objectweb.asm.Opcodes.FLOAD;
import static org.objectweb.asm.Opcodes.GOTO;
import static org.objectweb.asm.Opcodes.IADD;
import static org.objectweb.asm.Opcodes.ICONST_0;
import static org.objectweb.asm.Opcodes.ICONST_1;
import static org.objectweb.asm.Opcodes.ICONST_2;
import static org.objectweb.asm.Opcodes.ICONST_4;
import static org.objectweb.asm.Opcodes.IFNONNULL;
import static org.objectweb.asm.Opcodes.ILOAD;
import static org.objectweb.asm.Opcodes.INVOKESPECIAL;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;
import static org.objectweb.asm.Opcodes.ISTORE;
import static org.objectweb.asm.Opcodes.LCONST_0;
import static org.objectweb.asm.Opcodes.LLOAD;
import static org.objectweb.asm.Opcodes.NEW;
import static org.objectweb.asm.Opcodes.RETURN;
import static org.objectweb.asm.Opcodes.SIPUSH;
import static org.objectweb.asm.Opcodes.SWAP;
import static org.objectweb.asm.Opcodes.V1_8;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * Test data generator for b/62060793. This class creates a special labmda invocation that contains
 * *CONST_0 values on stack, which are passed as lambda arguments.
 */
public class Bug62060793TestDataGenerator {

  private static final String CLASS_NAME =
      "com/google/devtools/build/android/desugar/testdata/ConstantArgumentsInLambda";

  private static final String INTERFACE_TYPE_NAME = CLASS_NAME + "$Interface";

  public static void main(String[] args) throws IOException {
    checkArgument(
        args.length == 1, "Usage: %s <output-jar>", Bug62060793TestDataGenerator.class.getName());
    Path outputJar = Paths.get(args[0]);

    try (ZipOutputStream outZip =
        new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(outputJar)))) {
      String className = CLASS_NAME + ".class";
      writeToZipFile(outZip, className, createClass());
      String interfaceName = INTERFACE_TYPE_NAME + ".class";
      writeToZipFile(outZip, interfaceName, createInterface());
    }
  }

  private static void writeToZipFile(ZipOutputStream outZip, String entryName, byte[] content)
      throws IOException {
    ZipEntry result = new ZipEntry(entryName);
    result.setTime(0L);
    outZip.putNextEntry(result);
    outZip.write(content);
    outZip.closeEntry();
  }

  private static byte[] createClass() {
    ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_MAXS);
    MethodVisitor mv;
    cw.visit(V1_8, ACC_PUBLIC | ACC_SUPER, CLASS_NAME, null, "java/lang/Object", null);

    cw.visitInnerClass(
        INTERFACE_TYPE_NAME,
        CLASS_NAME,
        "Interface",
        ACC_PUBLIC | ACC_STATIC | ACC_ABSTRACT | ACC_INTERFACE);

    cw.visitInnerClass(
        "java/lang/invoke/MethodHandles$Lookup",
        "java/lang/invoke/MethodHandles",
        "Lookup",
        ACC_PUBLIC | ACC_FINAL | ACC_STATIC);

    {
      mv = cw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
      mv.visitCode();
      mv.visitVarInsn(ALOAD, 0);
      mv.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
      mv.visitInsn(RETURN);
      mv.visitEnd();
    }
    {
      mv =
          cw.visitMethod(
              ACC_PRIVATE | ACC_STATIC,
              "method",
              "(Ljava/lang/String;)Ljava/lang/String;",
              null,
              null);
      mv.visitParameter("str", 0);
      mv.visitCode();
      mv.visitVarInsn(ALOAD, 0);
      mv.visitInsn(ARETURN);
      mv.visitEnd();
    }
    {
      mv =
          cw.visitMethod(
              ACC_PRIVATE | ACC_STATIC,
              "method",
              "(ZCBFDJISLjava/lang/Object;[Ljava/lang/Object;Ljava/lang/String;)Ljava/lang/String;",
              null,
              null);
      mv.visitParameter("bool", 0);
      mv.visitParameter("c", 0);
      mv.visitParameter("b", 0);
      mv.visitParameter("f", 0);
      mv.visitParameter("d", 0);
      mv.visitParameter("l", 0);
      mv.visitParameter("i", 0);
      mv.visitParameter("s", 0);
      mv.visitParameter("o", 0);
      mv.visitParameter("array", 0);
      mv.visitParameter("str", 0);
      mv.visitCode();
      mv.visitVarInsn(ALOAD, 10);
      mv.visitMethodInsn(
          INVOKESTATIC,
          "java/lang/String",
          "valueOf",
          "(Ljava/lang/Object;)Ljava/lang/String;",
          false);
      mv.visitVarInsn(ASTORE, 13);
      mv.visitVarInsn(ALOAD, 11);
      Label l0 = new Label();
      mv.visitJumpInsn(IFNONNULL, l0);
      mv.visitInsn(ICONST_1);
      Label l1 = new Label();
      mv.visitJumpInsn(GOTO, l1);
      mv.visitLabel(l0);
      mv.visitFrame(Opcodes.F_APPEND, 1, new Object[] {"java/lang/String"}, 0, null);
      mv.visitInsn(ICONST_0);
      mv.visitLabel(l1);
      mv.visitFrame(Opcodes.F_SAME1, 0, null, 1, new Object[] {Opcodes.INTEGER});
      mv.visitVarInsn(ISTORE, 14);
      mv.visitIntInsn(BIPUSH, 91);
      mv.visitVarInsn(ALOAD, 12);
      mv.visitMethodInsn(
          INVOKESTATIC,
          "java/lang/String",
          "valueOf",
          "(Ljava/lang/Object;)Ljava/lang/String;",
          false);
      mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "length", "()I", false);
      mv.visitInsn(IADD);
      mv.visitVarInsn(ALOAD, 13);
      mv.visitMethodInsn(
          INVOKESTATIC,
          "java/lang/String",
          "valueOf",
          "(Ljava/lang/Object;)Ljava/lang/String;",
          false);
      mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "length", "()I", false);
      mv.visitInsn(IADD);
      mv.visitTypeInsn(NEW, "java/lang/StringBuilder");
      mv.visitInsn(DUP_X1);
      mv.visitInsn(SWAP);
      mv.visitMethodInsn(INVOKESPECIAL, "java/lang/StringBuilder", "<init>", "(I)V", false);
      mv.visitVarInsn(ALOAD, 12);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 0);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Z)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 1);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(C)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 2);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(I)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(FLOAD, 3);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(F)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(DLOAD, 4);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(D)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(LLOAD, 6);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(J)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 8);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(I)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 9);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(I)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ALOAD, 13);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
          false);
      mv.visitVarInsn(ILOAD, 14);
      mv.visitMethodInsn(
          INVOKEVIRTUAL,
          "java/lang/StringBuilder",
          "append",
          "(Z)Ljava/lang/StringBuilder;",
          false);
      mv.visitMethodInsn(
          INVOKEVIRTUAL, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;", false);
      mv.visitInsn(ARETURN);
      mv.visitEnd();
    }
    {
      mv =
          cw.visitMethod(
              ACC_PUBLIC | ACC_STATIC,
              "lambdaWithConstantArguments",
              "()L" + INTERFACE_TYPE_NAME + ";",
              null,
              null);
      mv.visitCode();
      mv.visitInsn(ICONST_0);
      mv.visitInsn(ICONST_1);
      mv.visitInsn(ICONST_2);
      mv.visitInsn(FCONST_0);
      mv.visitInsn(DCONST_0);
      mv.visitInsn(LCONST_0);
      mv.visitInsn(ICONST_4);
      mv.visitIntInsn(SIPUSH, 9);
      mv.visitInsn(ACONST_NULL);
      mv.visitInsn(ACONST_NULL);
      mv.visitInvokeDynamicInsn(
          "call",
          "(ZCBFDJISLjava/lang/Object;[Ljava/lang/Object;)L" + INTERFACE_TYPE_NAME + ";",
          new Handle(
              Opcodes.H_INVOKESTATIC,
              "java/lang/invoke/LambdaMetafactory",
              "metafactory",
              "(Ljava/lang/invoke/MethodHandles$Lookup;"
                  + "Ljava/lang/String;Ljava/lang/invoke/MethodType;"
                  + "Ljava/lang/invoke/MethodType;"
                  + "Ljava/lang/invoke/MethodHandle;"
                  + "Ljava/lang/invoke/MethodType;"
                  + ")Ljava/lang/invoke/CallSite;",
              false),
          new Object[] {
            Type.getType("(Ljava/lang/String;)Ljava/lang/String;"),
            new Handle(
                Opcodes.H_INVOKESTATIC,
                CLASS_NAME,
                "method",
                "(ZCBFDJISLjava/lang/Object;[Ljava/lang/Object;Ljava/lang/String;"
                    + ")Ljava/lang/String;",
                false),
            Type.getType("(Ljava/lang/String;)Ljava/lang/String;")
          });
      mv.visitInsn(ARETURN);
      mv.visitEnd();
    }

    cw.visitEnd();

    return cw.toByteArray();
  }

  private static byte[] createInterface() {
    ClassWriter cw = new ClassWriter(0);
    MethodVisitor mv;

    cw.visit(
        V1_8,
        ACC_PUBLIC | ACC_ABSTRACT | ACC_INTERFACE,
        INTERFACE_TYPE_NAME,
        null,
        "java/lang/Object",
        null);

    cw.visitInnerClass(
        INTERFACE_TYPE_NAME,
        CLASS_NAME,
        "Interface",
        ACC_PUBLIC | ACC_STATIC | ACC_ABSTRACT | ACC_INTERFACE);

    {
      mv =
          cw.visitMethod(
              ACC_PUBLIC | ACC_ABSTRACT,
              "call",
              "(Ljava/lang/String;)Ljava/lang/String;",
              null,
              null);
      mv.visitParameter("input", 0);
      mv.visitEnd();
    }
    cw.visitEnd();

    return cw.toByteArray();
  }
}
