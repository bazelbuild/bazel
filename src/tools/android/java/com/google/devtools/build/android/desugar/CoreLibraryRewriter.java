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
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.commons.ClassRemapper;
import org.objectweb.asm.commons.MethodRemapper;
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
    if (prefix.length() != 0) {
      return new PrefixingClassReader(content);
    } else {
      return new ClassReader(content);
    }
  }

  /**
   * Factory method that returns a ClassVisitor that delegates to a ClassWriter, removing prefix
   * from core library class names if it is not empty.
   */
  public UnprefixingClassWriter writer(int flags) {
    return new UnprefixingClassWriter(flags);
  }

  private static boolean shouldPrefix(String typeName) {
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

  /** Prefixes core library class names with prefix */
  public String prefix(String typeName) {
    if (prefix.length() > 0 && shouldPrefix(typeName)) {
      return prefix + typeName;
    }
    return typeName;
  }

  /** Removes prefix from class names */
  public String unprefix(String typeName) {
    if (prefix.length() == 0 || !typeName.startsWith(prefix)) {
      return typeName;
    }
    return typeName.substring(prefix.length());
  }

  /** ClassReader that prefixes core library class names as they are read */
  private class PrefixingClassReader extends ClassReader {
    PrefixingClassReader(InputStream content) throws IOException {
      super(content);
    }

    @Override
    public void accept(ClassVisitor cv, Attribute[] attrs, int flags) {
      cv =
          new ClassRemapperWithBugFix(
              cv,
              new Remapper() {
                @Override
                public String map(String typeName) {
                  return prefix(typeName);
                }
              });
      super.accept(cv, attrs, flags);
    }
  }

  /**
   * ClassVisitor that delegates to a ClassWriter, but removes a prefix as each class is written.
   * The unprefixing is optimized out if prefix is empty.
   */
  public class UnprefixingClassWriter extends ClassVisitor {
    private final ClassWriter writer;

    UnprefixingClassWriter(int flags) {
      super(Opcodes.ASM5);
      this.writer = new ClassWriter(flags);
      this.cv = this.writer;
      if (prefix.length() != 0) {
        this.cv =
            new ClassRemapperWithBugFix(
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

  /** ClassRemapper subclass to work around b/36654936 (caused by ASM bug 317785) */
  private static class ClassRemapperWithBugFix extends ClassRemapper {

    public ClassRemapperWithBugFix(ClassVisitor cv, Remapper remapper) {
      super(cv, remapper);
    }

    @Override
    protected MethodVisitor createMethodRemapper(MethodVisitor mv) {
      return new MethodRemapper(mv, this.remapper) {

        @Override
        public void visitFrame(int type, int nLocal, Object[] local, int nStack, Object[] stack) {
          if (this.mv != null) {
            mv.visitFrame(
                type,
                nLocal,
                remapEntriesWithBugfix(nLocal, local),
                nStack,
                remapEntriesWithBugfix(nStack, stack));
          }
        }

        /**
         * In {@code FrameNode.accept(MethodVisitor)}, when the frame is Opcodes.F_CHOP, it is
         * possible that nLocal is greater than 0, and local is null, which causes MethodRemapper to
         * throw a NPE. So the patch is to make sure that the {@code nLocal<=local.length} and
         * {@code nStack<=stack.length}
         */
        private Object[] remapEntriesWithBugfix(int n, Object[] entries) {
          if (entries == null || entries.length == 0) {
            return entries;
          }
          for (int i = 0; i < n; i++) {
            if (entries[i] instanceof String) {
              Object[] newEntries = new Object[n];
              if (i > 0) {
                System.arraycopy(entries, 0, newEntries, 0, i);
              }
              do {
                Object t = entries[i];
                newEntries[i++] = t instanceof String ? remapper.mapType((String) t) : t;
              } while (i < n);
              return newEntries;
            }
          }
          return entries;
        }
      };
    }
  }
}
