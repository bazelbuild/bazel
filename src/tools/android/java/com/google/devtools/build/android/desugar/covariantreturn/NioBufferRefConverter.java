/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.covariantreturn;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import com.google.devtools.build.android.desugar.langmodel.TypeMapper;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * A bytecode converter that supports to use covariant return types in the NIO buffer hierarchy.
 *
 * @see https://bugs.openjdk.java.net/browse/JDK-4774077.
 */
public final class NioBufferRefConverter extends ClassVisitor {

  /** The inheritance hierarchy root of Java NIO buffer family. */
  private static final ClassName NIO_BUFFER_BASE = ClassName.create("java/nio/Buffer");

  /** All overloading methods in {@link java.nio.Buffer} with covariant return type. */
  private static final ImmutableList<MethodKey> BASE_METHODS_WITH_COVARIANT_RETURN_TYPES =
      ImmutableList.of(
          MethodKey.create(NIO_BUFFER_BASE, "position", "(I)Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "limit", "(I)Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "mark", "()Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "reset", "()Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "clear", "()Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "flip", "()Ljava/nio/Buffer;"),
          MethodKey.create(NIO_BUFFER_BASE, "rewind", "()Ljava/nio/Buffer;"));

  /** All public type-specific NIO buffer classes derived from {@link java.nio.Buffer}. */
  private static final ImmutableSet<ClassName> TYPE_SPECIFIC_NIO_BUFFERS =
      ImmutableSet.of(
          ClassName.create("java/nio/IntBuffer"),
          ClassName.create("java/nio/CharBuffer"),
          ClassName.create("java/nio/FloatBuffer"),
          ClassName.create("java/nio/DoubleBuffer"),
          ClassName.create("java/nio/ShortBuffer"),
          ClassName.create("java/nio/LongBuffer"),
          ClassName.create("java/nio/ByteBuffer"));

  /** Used to find the replacement method from the original method invocation specification. */
  private final ImmutableMap<MethodKey, MethodKey> methodInvocationMappings;

  /** The public factory API for this class. */
  public static NioBufferRefConverter create(
      ClassVisitor classVisitor, TypeMapper corePackagePrefixer) {
    return new NioBufferRefConverter(
        classVisitor, corePackagePrefixer.map(getMethodInvocationMappings()));
  }

  private NioBufferRefConverter(
      ClassVisitor classVisitor, ImmutableMap<MethodKey, MethodKey> methodInvocationMappings) {
    super(Opcodes.ASM8, classVisitor);
    this.methodInvocationMappings = methodInvocationMappings;
  }

  @Override
  public MethodVisitor visitMethod(
      int access, String name, String descriptor, String signature, String[] exceptions) {
    MethodVisitor mv = super.visitMethod(access, name, descriptor, signature, exceptions);
    return mv == null ? null : new NioBufferMethodVisitor(api, mv, methodInvocationMappings);
  }

  /** Computes methods in Java NIO buffer family that are subject to invocation conversion. */
  private static ImmutableMap<MethodKey, MethodKey> getMethodInvocationMappings() {
    ImmutableMap.Builder<MethodKey, MethodKey> methodMappings = ImmutableMap.builder();
    for (ClassName typeSpecificNioBuffer : TYPE_SPECIFIC_NIO_BUFFERS) {
      for (MethodKey baseMethod : BASE_METHODS_WITH_COVARIANT_RETURN_TYPES) {
        methodMappings.put(
            MethodKey.create(
                typeSpecificNioBuffer,
                baseMethod.name(),
                Type.getMethodDescriptor(
                    typeSpecificNioBuffer.toAsmObjectType(), baseMethod.getArgumentTypeArray())),
            MethodKey.create(typeSpecificNioBuffer, baseMethod.name(), baseMethod.descriptor()));
      }
    }
    return methodMappings.build();
  }

  private static class NioBufferMethodVisitor extends MethodVisitor {

    private final ImmutableMap<MethodKey, MethodKey> methodInvocationMappings;

    NioBufferMethodVisitor(
        int api,
        MethodVisitor methodVisitor,
        ImmutableMap<MethodKey, MethodKey> methodInvocationMappings) {
      super(api, methodVisitor);
      this.methodInvocationMappings = methodInvocationMappings;
    }

    @Override
    public void visitMethodInsn(
        int opcode, String owner, String name, String descriptor, boolean isInterface) {
      MethodKey methodKey = MethodKey.create(ClassName.create(owner), name, descriptor);
      if (methodInvocationMappings.containsKey(methodKey)) {
        MethodKey mappedMethodKey = methodInvocationMappings.get(methodKey);
        super.visitMethodInsn(
            opcode,
            mappedMethodKey.ownerName(),
            mappedMethodKey.name(),
            mappedMethodKey.descriptor(),
            isInterface);
        super.visitTypeInsn(Opcodes.CHECKCAST, owner);
        return;
      }
      super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }
  }
}
