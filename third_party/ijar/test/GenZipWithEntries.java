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

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;

/**
 * Generates a jar file with the specified number of .class entries.
 *
 * usage: GenZipWithEntries <number of zip entries> <output file>
 */
public class GenZipWithEntries {

  public static void main(String[] args) throws IOException {
    int entries = Integer.parseInt(args[0]);
    Path out = Paths.get(args[1]);
    try (OutputStream os = Files.newOutputStream(out);
        JarOutputStream jos = new JarOutputStream(os)) {
      for (int i = 1; i <= entries; i++) {
        String name = String.format("Test%d", i);
        jos.putNextEntry(new ZipEntry(String.format("%s.class", name)));
        jos.write(dump(name));
      }
    }
  }

  public static byte[] dump(String name) {
    ClassWriter cw = new ClassWriter(0);
    cw.visit(52, Opcodes.ACC_PUBLIC + Opcodes.ACC_SUPER, name, null, "java/lang/Object", null);
    {
      MethodVisitor mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null);
      mv.visitCode();
      mv.visitVarInsn(Opcodes.ALOAD, 0);
      mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
      mv.visitInsn(Opcodes.RETURN);
      mv.visitMaxs(1, 1);
      mv.visitEnd();
    }
    cw.visitEnd();
    return cw.toByteArray();
  }
}
