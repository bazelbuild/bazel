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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.util.Textifier;

/** Print the types of the operand stack for each method. */
public class ByteCodeTypePrinter {

  public static void printClassesWithTypes(Path inputJarFile, PrintWriter printWriter)
      throws IOException {
    Preconditions.checkState(
        Files.exists(inputJarFile), "The input jar file %s does not exist.", inputJarFile);
    try (ZipFile jarFile = new ZipFile(inputJarFile.toFile())) {
      for (ZipEntry entry : getSortedClassEntriess(jarFile)) {
        try (InputStream classStream = jarFile.getInputStream(entry)) {
          printWriter.println("\nClass: " + entry.getName());
          ClassReader classReader = new ClassReader(classStream);
          ClassVisitor visitor = new ClassWithTypeDumper(printWriter);
          classReader.accept(visitor, 0);
        }
        printWriter.println("\n");
      }
    }
  }

  private static ImmutableList<ZipEntry> getSortedClassEntriess(ZipFile jar) {
    return jar.stream()
        .filter(entry -> entry.getName().endsWith(".class"))
        .sorted(Comparator.comparing(ZipEntry::getName))
        .collect(ImmutableList.toImmutableList());
  }

  private static class ClassWithTypeDumper extends ClassVisitor {

    private String internalName;
    private final PrintWriter printWriter;

    public ClassWithTypeDumper(PrintWriter printWriter) {
      super(Opcodes.ASM8);
      this.printWriter = printWriter;
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      internalName = name;
      super.visit(version, access, name, signature, superName, interfaces);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      printWriter.println("Method " + name);
      MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
      BytecodeTypeInference inference = new BytecodeTypeInference(access, internalName, name, desc);
      mv = new MethodIrTypeDumper(mv, inference, printWriter);
      inference.setDelegateMethodVisitor(mv);
      // Let the type inference runs first.
      return inference;
    }
  }

  private static final class TextifierExt extends Textifier {

    public TextifierExt() {
      super(Opcodes.ASM8);
    }

    public void print(String string) {
      text.add(tab2 + string);
    }
  }

  private static class MethodIrTypeDumper extends MethodVisitor {

    private final BytecodeTypeInference inference;
    private final TextifierExt printer = new TextifierExt();
    private final PrintWriter printWriter;

    public MethodIrTypeDumper(
        MethodVisitor visitor, BytecodeTypeInference inference, PrintWriter printWriter) {
      super(Opcodes.ASM8, visitor);
      this.inference = inference;
      this.printWriter = printWriter;
    }

    private void printTypeOfOperandStack() {
      printer.print("    |__STACK: " + inference.getOperandStackAsString() + "\n");
      printer.print("    |__LOCAL: " + inference.getLocalsAsString() + "\n");
    }

    @Override
    public void visitIntInsn(int opcode, int operand) {
      printer.visitIntInsn(opcode, operand);
      printTypeOfOperandStack();
      super.visitIntInsn(opcode, operand);
    }

    @Override
    public void visitInsn(int opcode) {
      printer.visitInsn(opcode);
      printTypeOfOperandStack();
      super.visitInsn(opcode);
    }

    @Override
    public void visitMultiANewArrayInsn(String desc, int dims) {
      printer.visitMultiANewArrayInsn(desc, dims);
      printTypeOfOperandStack();
      super.visitMultiANewArrayInsn(desc, dims);
    }

    @Override
    public void visitLookupSwitchInsn(Label dflt, int[] keys, Label[] labels) {
      printer.visitLookupSwitchInsn(dflt, keys, labels);
      printTypeOfOperandStack();
      super.visitLookupSwitchInsn(dflt, keys, labels);
    }

    @Override
    public void visitTableSwitchInsn(int min, int max, Label dflt, Label... labels) {
      printer.visitTableSwitchInsn(min, max, dflt, labels);
      printTypeOfOperandStack();
      super.visitTableSwitchInsn(min, max, dflt, labels);
    }

    @Override
    public void visitIincInsn(int var, int increment) {
      printer.visitIincInsn(var, increment);
      printTypeOfOperandStack();
      super.visitIincInsn(var, increment);
    }

    @Override
    public void visitLdcInsn(Object cst) {
      printer.visitLdcInsn(cst);
      printTypeOfOperandStack();
      super.visitLdcInsn(cst);
    }

    @Override
    public void visitJumpInsn(int opcode, Label label) {
      printer.visitJumpInsn(opcode, label);
      printTypeOfOperandStack();
      super.visitJumpInsn(opcode, label);
    }

    @Override
    public void visitInvokeDynamicInsn(String name, String desc, Handle bsm, Object... bsmArgs) {
      printer.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
      printTypeOfOperandStack();
      super.visitInvokeDynamicInsn(name, desc, bsm, bsmArgs);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      printer.visitMethodInsn(opcode, owner, name, desc, itf);
      printTypeOfOperandStack();
      super.visitMethodInsn(opcode, owner, name, desc, itf);
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc) {
      printer.visitMethodInsn(opcode, owner, name, desc);
      printTypeOfOperandStack();
      super.visitMethodInsn(opcode, owner, name, desc);
    }

    @Override
    public void visitFieldInsn(int opcode, String owner, String name, String desc) {
      printer.visitFieldInsn(opcode, owner, name, desc);
      printTypeOfOperandStack();
      super.visitFieldInsn(opcode, owner, name, desc);
    }

    @Override
    public void visitTypeInsn(int opcode, String type) {
      printer.visitTypeInsn(opcode, type);
      printTypeOfOperandStack();
      super.visitTypeInsn(opcode, type);
    }

    @Override
    public void visitVarInsn(int opcode, int var) {
      printer.visitVarInsn(opcode, var);
      printTypeOfOperandStack();
      super.visitVarInsn(opcode, var);
    }

    @Override
    public void visitFrame(int type, int nLocal, Object[] local, int nStack, Object[] stack) {
      printer.visitFrame(type, nLocal, local, nStack, stack);
      super.visitFrame(type, nLocal, local, nStack, stack);
    }

    @Override
    public void visitLabel(Label label) {
      printer.visitLabel(label);
      super.visitLabel(label);
    }

    @Override
    public void visitEnd() {
      printer.print(printWriter);
      printWriter.flush();
      super.visitEnd();
    }
  }
}
