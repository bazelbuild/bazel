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

import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.zip.ZipEntry;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Class loader that can "load" classes from header Jars. This class loader stubs in missing code
 * attributes on the fly to make {@link ClassLoader#defineClass} happy.  Classes loaded are unusable
 * other than to resolve method references, so this class loader should only be used to process
 * or inspect classes, not to execute their code.  Also note that the resulting classes may be
 * missing private members, which header Jars may omit.
 *
 * @see java.net.URLClassLoader
 */
class HeaderClassLoader extends ClassLoader {

  private final Map<String, JarFile> jarfiles;

  /** Creates a classloader from the given classpath with the given parent. */
  public static HeaderClassLoader fromClassPath(List<Path> classpath, ClassLoader parent)
      throws IOException {
    return new HeaderClassLoader(indexJars(classpath), parent);
  }

  /**
   * Opens the given list of Jar files and returns an index of all classes in them, to avoid
   * scanning all Jars over and over for each class in {@link #findClass}.
   */
  private static Map<String, JarFile> indexJars(List<Path> classpath) throws IOException {
    HashMap<String, JarFile> result = new HashMap<>();
    for (Path jarfile : classpath) {
      JarFile jar = new JarFile(jarfile.toFile());
      for (Enumeration<JarEntry> cur = jar.entries(); cur.hasMoreElements(); ) {
        JarEntry entry = cur.nextElement();
        if (entry.getName().endsWith(".class") && !result.containsKey(entry.getName())) {
          result.put(entry.getName(), jar);
        }
      }
    }
    return result;
  }

  private HeaderClassLoader(Map<String, JarFile> jarfiles, ClassLoader parent) {
    super(parent);
    this.jarfiles = jarfiles;
  }

  @Override
  protected Class<?> findClass(String name) throws ClassNotFoundException {
    String filename = name.replace('.', '/') + ".class";
    JarFile jarfile = jarfiles.get(filename);
    if (jarfile == null) {
      throw new ClassNotFoundException();
    }
    ZipEntry entry = jarfile.getEntry(filename);
    byte[] bytecode;
    try (InputStream content = jarfile.getInputStream(entry)) {
      ClassReader reader = new ClassReader(content);
      // Have ASM compute maxs so we don't need to figure out how many formal parameters there are
      ClassWriter writer = new ClassWriter(ClassWriter.COMPUTE_MAXS);
      reader.accept(new CodeStubber(writer), 0);
      bytecode = writer.toByteArray();
    } catch (IOException e) {
      throw new IOError(e);
    }
    return defineClass(name, bytecode, 0, bytecode.length);
  }

  /** Class visitor that stubs in missing code attributes. */
  private static class CodeStubber extends ClassVisitor {

    public CodeStubber(ClassVisitor cv) {
      super(Opcodes.ASM5, cv);
    }

    @Override
    public MethodVisitor visitMethod(
        int access, String name, String desc, String signature, String[] exceptions) {
      MethodVisitor dest = super.visitMethod(access, name, desc, signature, exceptions);
      if ((access & (Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE)) != 0) {
        // No need to stub out abstract or native methods
        return dest;
      }
      return new BodyStubber(dest);
    }

  }

  /** Method visitor used by {@link CodeStubber} to put code into methods without code. */
  private static class BodyStubber extends MethodVisitor {

    private static final String EXCEPTION_INTERNAL_NAME = "java/lang/UnsupportedOperationException";

    private boolean hasCode = false;

    public BodyStubber(MethodVisitor mv) {
      super(Opcodes.ASM5, mv);
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
