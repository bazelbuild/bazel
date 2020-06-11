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
package com.google.devtools.build.android.desugar.io;

import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.TypeMapper;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.Remapper;

/** Utility class to prefix or unprefix class names of core library classes */
public class CoreLibraryRewriter {

  private final String prefix;
  private final TypeMapper prefixer;

  public CoreLibraryRewriter(String prefix) {
    this.prefix = prefix;
    this.prefixer = new TypeMapper(this::prefix);
  }

  /**
   * Factory method that returns either a normal ClassReader if prefix is empty, or a ClassReader
   * with a ClassRemapper that prefixes class names of core library classes if prefix is not empty.
   */
  public ClassReader reader(InputStream content) throws IOException {
    if (prefix.isEmpty()) {
      return new ClassReader(content);
    } else {
      return new PrefixingClassReader(content, prefixer);
    }
  }

  /**
   * Factory method that returns a ClassVisitor that delegates to a ClassWriter, removing prefix
   * from core library class names if it is not empty.
   */
  public UnprefixingClassWriter writer(int flags) {
    return new UnprefixingClassWriter(flags);
  }

  static boolean shouldPrefix(String typeName) {
    return (typeName.startsWith("java/")
            || typeName.startsWith("sun/")
            || typeName.startsWith("javadesugar/") // Testing-only fake package prefix.
        )
        && !except(typeName);
  }

  private static boolean except(String typeName) {
    if (typeName.startsWith("java/lang/invoke/")) {
      return true;
    }

    switch (typeName) {
        // Autoboxed types
      case "java/lang/Boolean":
      case "java/lang/Byte":
      case "java/lang/Character":
      case "java/lang/Double":
      case "java/lang/Float":
      case "java/lang/Integer":
      case "java/lang/Long":
      case "java/lang/Number":
      case "java/lang/Short":

        // Special types
      case "java/lang/Class":
      case "java/lang/Object":
      case "java/lang/String":
      case "java/lang/Throwable":
        return true;

      default: // fall out
    }

    return false;
  }

  public String getPrefix() {
    return prefix;
  }

  public TypeMapper getPrefixer() {
    return prefixer;
  }

  private ClassName prefix(ClassName className) {
    if (shouldPrefix(className.binaryName())) {
      return className.withPackagePrefix(prefix);
    }
    return className;
  }

  /** Removes prefix from class names */
  public String unprefix(String typeName) {
    if (prefix.isEmpty() || !typeName.startsWith(prefix)) {
      return typeName;
    }
    return typeName.substring(prefix.length());
  }

  /** ClassReader that prefixes core library class names as they are read */
  private static class PrefixingClassReader extends ClassReader {

    private final TypeMapper prefixer;

    PrefixingClassReader(InputStream content, TypeMapper prefixer) throws IOException {
      super(content);
      this.prefixer = prefixer;
    }

    @Override
    public void accept(ClassVisitor cv, Attribute[] attrs, int flags) {
      cv = new ClassRemapper(cv, prefixer);
      super.accept(cv, attrs, flags);
    }

    @Override
    public String getClassName() {
      return prefixer.map(super.getClassName());
    }

    @Override
    public String getSuperName() {
      String result = super.getSuperName();
      return result != null ? prefixer.map(result) : null;
    }

    @Override
    public String[] getInterfaces() {
      String[] result = super.getInterfaces();
      for (int i = 0, len = result.length; i < len; ++i) {
        result[i] = prefixer.map(result[i]);
      }
      return result;
    }
  }

  /**
   * ClassVisitor that delegates to a ClassWriter, but removes a prefix as each class is written.
   * The unprefixing is optimized out if prefix is empty.
   */
  public class UnprefixingClassWriter extends ClassVisitor {
    private final ClassWriter writer;

    private String finalClassName;

    UnprefixingClassWriter(int flags) {
      super(Opcodes.ASM8);
      this.writer = new ClassWriter(flags);
      this.cv = this.writer;
      if (!prefix.isEmpty()) {
        this.cv =
            new ClassRemapper(
                this.writer,
                new Remapper() {
                  @Override
                  public String map(String typeName) {
                    return unprefix(typeName);
                  }
                });
      }
    }

    /** Returns the (unprefixed) name of the class once written. */
    @Nullable
    public String getClassName() {
      return finalClassName;
    }

    public byte[] toByteArray() {
      return writer.toByteArray();
    }

    @Override
    public void visit(
        int version,
        int access,
        String name,
        String signature,
        String superName,
        String[] interfaces) {
      finalClassName = unprefix(name);
      super.visit(version, access, name, signature, superName, interfaces);
    }
  }
}
