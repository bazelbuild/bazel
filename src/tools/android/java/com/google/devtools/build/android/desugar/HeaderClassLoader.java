// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.FieldVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Class loader that can "load" classes from header Jars. This class loader stubs in missing code
 * attributes on the fly to make {@link ClassLoader#defineClass} happy. Classes loaded are unusable
 * other than to resolve method references, so this class loader should only be used to process or
 * inspect classes, not to execute their code. Also note that the resulting classes may be missing
 * private members, which header Jars may omit.
 *
 * @see java.net.URLClassLoader
 */
class HeaderClassLoader extends ClassLoader {

  private final IndexedInputs indexedInputs;
  private final CoreLibraryRewriter rewriter;

  public HeaderClassLoader(
      IndexedInputs indexedInputs, CoreLibraryRewriter rewriter, ClassLoader parent) {
    super(parent);
    this.rewriter = rewriter;
    this.indexedInputs = indexedInputs;
  }

  @Override
  protected Class<?> findClass(String name) throws ClassNotFoundException {
    String filename = rewriter.unprefix(name.replace('.', '/') + ".class");
    InputFileProvider inputFileProvider = indexedInputs.getInputFileProvider(filename);
    if (inputFileProvider == null) {
      throw new ClassNotFoundException("Class " + name + " not found");
    }
    byte[] bytecode;
    try (InputStream content = inputFileProvider.getInputStream(filename)) {
      ClassReader reader = rewriter.reader(content);
      // Have ASM compute maxs so we don't need to figure out how many formal parameters there are
      ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
      ImmutableList<FieldInfo> interfaceFieldNames = getFieldsIfReaderIsInterface(reader);
      reader.accept(new CodeStubber(writer, interfaceFieldNames), 0);
      bytecode = writer.toByteArray();
    } catch (IOException e) {
      throw new IOError(e);
    }
    return defineClass(name, bytecode, 0, bytecode.length);
  }

  /**
   * If the {@code reader} is an interface, then extract all the declared fields in it. Otherwise,
   * return an empty list.
   */
  private static ImmutableList<FieldInfo> getFieldsIfReaderIsInterface(ClassReader reader) {
    if (BitFlags.isSet(reader.getAccess(), Opcodes.ACC_INTERFACE)) {
      NonPrimitiveFieldCollector collector = new NonPrimitiveFieldCollector();
      reader.accept(collector, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG);
      return collector.declaredNonPrimitiveFields.build();
    }
    return ImmutableList.of();
  }

  /** Collect the fields defined in a class. */
  private static class NonPrimitiveFieldCollector extends ClassVisitor {

    final ImmutableList.Builder<FieldInfo> declaredNonPrimitiveFields = ImmutableList.builder();
    private String internalName;

    public NonPrimitiveFieldCollector() {
      super(Opcodes.ASM6);
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      super.visit(version, access, name, signature, superName, interfaces);
      this.internalName = name;
    }

    @Override
    public FieldVisitor visitField(
        int access, String name, String desc, String signature, Object value) {
      if (isNonPrimitiveType(desc)) {
        declaredNonPrimitiveFields.add(FieldInfo.create(internalName, name, desc));
      }
      return null;
    }

    private static boolean isNonPrimitiveType(String type) {
      char firstChar = type.charAt(0);
      return firstChar == '[' || firstChar == 'L';
    }
  }


  /**
   * Class visitor that stubs in missing code attributes, and erases the body of the static
   * initializer of functional interfaces if the interfaces have default methods. The erasion of the
   * clinit is mainly because when we are desugaring lambdas, we need to load the functional
   * interfaces via class loaders, and since the interfaces have default methods, according to the
   * JVM spec, these interfaces will be executed. This should be prevented due to security concerns.
   */
  private static class CodeStubber extends ClassVisitor {

    private String internalName;
    private boolean isInterface;
    private final ImmutableList<FieldInfo> interfaceFields;

    public CodeStubber(ClassVisitor cv, ImmutableList<FieldInfo> interfaceFields) {
      super(Opcodes.ASM6, cv);
      this.interfaceFields = interfaceFields;
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      super.visit(version, access, name, signature, superName, interfaces);
      isInterface = BitFlags.isSet(access, Opcodes.ACC_INTERFACE);
      internalName = name;
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      MethodVisitor dest = super.visitMethod(access, name, desc, signature, exceptions);
      if ((access & (Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE)) != 0) {
        // No need to stub out abstract or native methods
        return dest;
      }
      if (isInterface && "<clinit>".equals(name)) {
        // Delete class initializers, to avoid code gets executed when we desugar lambdas.
        // See b/62184142
        return new InterfaceInitializerEraser(dest, internalName, interfaceFields);
      }
      return new BodyStubber(dest);
    }
  }

  /**
   * Erase the static initializer of an interface. Given an interface with non-primitive fields,
   * this eraser discards the original body of clinit, and initializes each non-primitive field to
   * null
   */
  private static class InterfaceInitializerEraser extends MethodVisitor {

    private final MethodVisitor dest;
    private final ImmutableList<FieldInfo> interfaceFields;

    public InterfaceInitializerEraser(
        MethodVisitor mv, String internalName, ImmutableList<FieldInfo> interfaceFields) {
      super(Opcodes.ASM6);
      dest = mv;
      this.interfaceFields = interfaceFields;
    }

    @Override
    public void visitCode() {
      dest.visitCode();
    }

    @Override
    public void visitEnd() {
      for (FieldInfo fieldInfo : interfaceFields) {
        dest.visitInsn(Opcodes.ACONST_NULL);
        dest.visitFieldInsn(
            Opcodes.PUTSTATIC, fieldInfo.owner(), fieldInfo.name(), fieldInfo.desc());
      }
      dest.visitInsn(Opcodes.RETURN);
      dest.visitMaxs(0, 0);
      dest.visitEnd();
    }
  }

  /** Method visitor used by {@link CodeStubber} to put code into methods without code. */
  private static class BodyStubber extends MethodVisitor {

    private static final String EXCEPTION_INTERNAL_NAME = "java/lang/UnsupportedOperationException";

    private boolean hasCode = false;

    public BodyStubber(MethodVisitor mv) {
      super(Opcodes.ASM6, mv);
    }

    @Override
    public void visitCode() {
      hasCode = true;
      super.visitCode();
    }

    @Override
    public void visitEnd() {
      if (!hasCode) {
        super.visitTypeInsn(Opcodes.NEW, EXCEPTION_INTERNAL_NAME);
        super.visitInsn(Opcodes.DUP);
        super.visitMethodInsn(
            Opcodes.INVOKESPECIAL, EXCEPTION_INTERNAL_NAME, "<init>", "()V", /*itf*/ false);
        super.visitInsn(Opcodes.ATHROW);
        super.visitMaxs(0, 0); // triggers computation of the actual max's
      }
      super.visitEnd();
    }
  }
}
