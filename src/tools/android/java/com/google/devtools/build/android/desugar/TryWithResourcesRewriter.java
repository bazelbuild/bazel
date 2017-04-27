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

import static org.objectweb.asm.Opcodes.ASM5;
import static org.objectweb.asm.Opcodes.INVOKESTATIC;
import static org.objectweb.asm.Opcodes.INVOKEVIRTUAL;

import com.google.common.base.Function;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import java.util.concurrent.atomic.AtomicInteger;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;

/**
 * Desugar try-with-resources. This class visitor intercepts calls to the following methods, and
 * redirect them to ThrowableExtension.
 * <li>{@code Throwable.addSuppressed(Throwable)}
 * <li>{@code Throwable.getSuppressed()}
 * <li>{@code Throwable.printStackTrace()}
 * <li>{@code Throwable.printStackTrace(PrintStream)}
 * <li>{@code Throwable.printStackTrace(PringWriter)}
 */
public class TryWithResourcesRewriter extends ClassVisitor {

  private static final String RUNTIME_PACKAGE_INTERNAL_NAME =
      "com/google/devtools/build/android/desugar/runtime";

  static final String THROWABLE_EXTENSION_INTERNAL_NAME =
      RUNTIME_PACKAGE_INTERNAL_NAME + '/' + "ThrowableExtension";

  /** The extension classes for java.lang.Throwable. */
  static final ImmutableSet<String> THROWABLE_EXT_CLASS_INTERNAL_NAMES =
      ImmutableSet.of(
          THROWABLE_EXTENSION_INTERNAL_NAME,
          THROWABLE_EXTENSION_INTERNAL_NAME + "$AbstractDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$MimicDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$NullDesugaringStrategy",
          THROWABLE_EXTENSION_INTERNAL_NAME + "$ReuseDesugaringStrategy");

  /** The extension classes for java.lang.Throwable. All the names end with ".class" */
  static final ImmutableSet<String> THROWABLE_EXT_CLASS_INTERNAL_NAMES_WITH_CLASS_EXT =
      FluentIterable.from(THROWABLE_EXT_CLASS_INTERNAL_NAMES)
          .transform(
              new Function<String, String>() {
                @Override
                public String apply(String s) {
                  return s + ".class";
                }
              })
          .toSet();

  static final ImmutableMultimap<String, String> TARGET_METHODS =
      ImmutableMultimap.<String, String>builder()
          .put("addSuppressed", "(Ljava/lang/Throwable;)V")
          .put("getSuppressed", "()[Ljava/lang/Throwable;")
          .put("printStackTrace", "()V")
          .put("printStackTrace", "(Ljava/io/PrintStream;)V")
          .put("printStackTrace", "(Ljava/io/PrintWriter;)V")
          .build();

  static final ImmutableMap<String, String> METHOD_DESC_MAP =
      ImmutableMap.<String, String>builder()
          .put("(Ljava/lang/Throwable;)V", "(Ljava/lang/Throwable;Ljava/lang/Throwable;)V")
          .put("()[Ljava/lang/Throwable;", "(Ljava/lang/Throwable;)[Ljava/lang/Throwable;")
          .put("()V", "(Ljava/lang/Throwable;)V")
          .put("(Ljava/io/PrintStream;)V", "(Ljava/lang/Throwable;Ljava/io/PrintStream;)V")
          .put("(Ljava/io/PrintWriter;)V", "(Ljava/lang/Throwable;Ljava/io/PrintWriter;)V")
          .build();

  private final ClassLoader classLoader;

  private final AtomicInteger numOfTryWithResourcesInvoked;

  public TryWithResourcesRewriter(
      ClassVisitor classVisitor,
      ClassLoader classLoader,
      AtomicInteger numOfTryWithResourcesInvoked) {
    super(ASM5, classVisitor);
    this.classLoader = classLoader;
    this.numOfTryWithResourcesInvoked = numOfTryWithResourcesInvoked;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String desc, String signature, String[] exceptions) {
    MethodVisitor visitor = super.cv.visitMethod(access, name, desc, signature, exceptions);
    return visitor == null || THROWABLE_EXT_CLASS_INTERNAL_NAMES.contains(name)
        ? visitor
        : new TryWithResourceVisitor(visitor, classLoader);
  }

  private class TryWithResourceVisitor extends MethodVisitor {

    private final ClassLoader classLoader;

    public TryWithResourceVisitor(MethodVisitor methodVisitor, ClassLoader classLoader) {
      super(ASM5, methodVisitor);
      this.classLoader = classLoader;
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
      if (!isMethodCallTargeted(opcode, owner, name, desc)) {
        super.visitMethodInsn(opcode, owner, name, desc, itf);
        return;
      }
      numOfTryWithResourcesInvoked.incrementAndGet();
      super.visitMethodInsn(
          INVOKESTATIC, THROWABLE_EXTENSION_INTERNAL_NAME, name, METHOD_DESC_MAP.get(desc), false);
    }

    private boolean isMethodCallTargeted(int opcode, String owner, String name, String desc) {
      if (opcode != INVOKEVIRTUAL) {
        return false;
      }
      if (!TARGET_METHODS.containsEntry(name, desc)) {
        return false;
      }
      if (owner.equals("java/lang/Throwable")) {
        return true; // early return, for performance.
      }
      try {
        Class<?> throwableClass = classLoader.loadClass("java.lang.Throwable");
        Class<?> klass = classLoader.loadClass(owner.replace('/', '.'));
        return throwableClass.isAssignableFrom(klass);
      } catch (ClassNotFoundException e) {
        throw new AssertionError(e);
      }
    }
  }
}
