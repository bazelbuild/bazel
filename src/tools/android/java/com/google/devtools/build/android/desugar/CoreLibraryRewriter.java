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

import java.io.IOException;
import java.io.InputStream;
import org.objectweb.asm.Attribute;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.Remapper;

/** Utility class to prefix or unprefix class names of core library classes */
class CoreLibraryRewriter {
  private final String prefix;

  public CoreLibraryRewriter(String prefix) {
    this.prefix = prefix;
  }

  /**
   * Factory method that returns either a normal ClassReader if prefix is empty, or a ClassReader
   * with a ClassRemapper that prefixes class names of core library classes if prefix is not empty.
   */
  public ClassReader reader(InputStream content) throws IOException {
    if (prefix.isEmpty()) {
      return new ClassReader(content);
    } else {
      return new PrefixingClassReader(content, prefix);
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
    return (typeName.startsWith("java/") || typeName.startsWith("sun/")) && !except(typeName);
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

  /** Removes prefix from class names */
  public String unprefix(String typeName) {
    if (prefix.isEmpty() || !typeName.startsWith(prefix)) {
      return typeName;
    }
    return typeName.substring(prefix.length());
  }

  /** ClassReader that prefixes core library class names as they are read */
  private static class PrefixingClassReader extends ClassReader {
    private final String prefix;

    PrefixingClassReader(InputStream content, String prefix) throws IOException {
      super(content);
      this.prefix = prefix;
    }

    @Override
    public void accept(ClassVisitor cv, Attribute[] attrs, int flags) {
      cv =
          new ClassRemapper(
              cv,
              new Remapper() {
                @Override
                public String map(String typeName) {
                  return prefix(typeName);
                }
              });
      super.accept(cv, attrs, flags);
    }

    /** Prefixes core library class names with prefix. */
    private String prefix(String typeName) {
      if (shouldPrefix(typeName)) {
        return prefix + typeName;
      }
      return typeName;
    }
  }

  /**
   * ClassVisitor that delegates to a ClassWriter, but removes a prefix as each class is written.
   * The unprefixing is optimized out if prefix is empty.
   */
  public class UnprefixingClassWriter extends ClassVisitor {
    private final ClassWriter writer;

    UnprefixingClassWriter(int flags) {
      super(Opcodes.ASM6);
      this.writer = new ClassWriter(flags);
      this.cv = this.writer;
      if (!prefix.isEmpty()) {
        this.cv =
            new ClassRemapper(
                this.cv,
                new Remapper() {
                  @Override
                  public String map(String typeName) {
                    return unprefix(typeName);
                  }
                });
      }
    }

    byte[] toByteArray() {
      return writer.toByteArray();
    }
  }
}
