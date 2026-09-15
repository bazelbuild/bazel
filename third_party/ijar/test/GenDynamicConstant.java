// Copyright 2023 The Bazel Authors. All rights reserved.
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

import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.invoke.ConstantBootstraps;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.ConstantDynamic;
import org.objectweb.asm.Handle;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

public class GenDynamicConstant {
  public static void main(String[] args) throws IOException {
    try (JarOutputStream jos = new JarOutputStream(new FileOutputStream(args[0]))) {
      jos.putNextEntry(new ZipEntry("dynamicconstant/Test.class"));
      jos.write(dump());
    }
  }

  public static byte[] dump() {
    ClassWriter cw = new ClassWriter(0);
    MethodVisitor mv;

    cw.visit(
        Opcodes.V16,
        Opcodes.ACC_PUBLIC + Opcodes.ACC_SUPER,
        "dynamicconstant/Test",
        null,
        "java/lang/Object",
        null);

    cw.visitEnd();

    var desc = Type.INT_TYPE.getDescriptor();
    var get =
        cw.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, "get", "()" + desc, null, null);
    get.visitInsn(Opcodes.ICONST_5);
    get.visitInsn(Opcodes.IRETURN);
    get.visitMaxs(1, 0);
    get.visitEnd();
    var invokeHandle =
        new Handle(
            Opcodes.H_INVOKESTATIC,
            Type.getInternalName(ConstantBootstraps.class),
            "invoke",
            "(Ljava/lang/invoke/MethodHandles$Lookup;Ljava/lang/String;Ljava/lang/Class;Ljava/lang/invoke/MethodHandle;[Ljava/lang/Object;)Ljava/lang/Object;",
            false);
    var handle =
        new Handle(Opcodes.H_INVOKESTATIC, "dynamicconstant/Test", "get", "()" + desc, false);
    var run =
        cw.visitMethod(Opcodes.ACC_PUBLIC | Opcodes.ACC_STATIC, "run", "()" + desc, null, null);
    run.visitLdcInsn(new ConstantDynamic("const", desc, invokeHandle, handle));
    run.visitInsn(Opcodes.IRETURN);
    run.visitMaxs(1, 0);
    run.visitEnd();

    return cw.toByteArray();
  }
}
