// Copyright 2015 The Bazel Authors. All rights reserved.
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

package test;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;

public class GenSourceDebugExtension {
  public static void main(String[] args) throws IOException {
    try (JarOutputStream jos = new JarOutputStream(new FileOutputStream(args[0]))) {
      jos.putNextEntry(new ZipEntry("sourcedebugextension/Test.class"));
      jos.write(dump());
    }
  }

  public static byte[] dump() {
    ClassWriter cw = new ClassWriter(0);
    MethodVisitor mv;

    cw.visit(
        52,
        Opcodes.ACC_PUBLIC + Opcodes.ACC_SUPER,
        "sourcedebugextension/Test",
        null,
        "java/lang/Object",
        null);

    // This is the important part for the test - it's an example SourceDebugExtension found
    // in Clojure.
    cw.visitSource(
        "dispatch.clj",
        "SMAP\ndispatch.java\nClojure\n*S Clojure\n*F\n+ 1 dispatch.clj\nclojure/pprint/dispatch.clj\n*L\n144#1,8:144\n*E");

    {
      mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null);
      mv.visitCode();
      mv.visitVarInsn(Opcodes.ALOAD, 0);
      mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
      mv.visitInsn(Opcodes.RETURN);
      mv.visitMaxs(1, 1);
      mv.visitEnd();
    }
    {
      mv = cw.visitMethod(
          Opcodes.ACC_PUBLIC + Opcodes.ACC_STATIC,
          "main",
          "([Ljava/lang/String;)V",
          null,
          null);
      mv.visitCode();
      mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System", "err", "Ljava/io/PrintStream;");
      mv.visitLdcInsn("Hello");
      mv.visitMethodInsn(
          Opcodes.INVOKEVIRTUAL, "java/io/PrintStream", "println", "(Ljava/lang/String;)V", false);
      mv.visitInsn(Opcodes.RETURN);
      mv.visitMaxs(2, 1);
      mv.visitEnd();
    }
    cw.visitEnd();

    return cw.toByteArray();
  }
}
