/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest;

import static org.objectweb.asm.Opcodes.ACC_PRIVATE;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;

import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclInfo;
import com.google.devtools.build.android.desugar.langmodel.MethodDeclVisitor;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/**
 * A visitor class that generation of accessor methods, including bridge method generation for class
 * and private interface method re-writing.
 */
final class MethodAccessorEmitter
    implements MethodDeclVisitor<MethodVisitor, MethodDeclInfo, ClassVisitor> {

  private final NestDigest nestDigest;

  MethodAccessorEmitter(NestDigest nestDigest) {
    this.nestDigest = nestDigest;
  }

  /**
   * Emits a synthetic overloaded constructor which delegates the construction logic to the source
   * constructor. For example,
   *
   * <pre><code>
   *   class Foo {
   *     private Foo(A a) {...}
   *
   *     &#047;&#047; Synthetic overloaded constructor
   *     Foo(A a, NestCC var) {
   *       this(a);
   *     }
   *   }
   * </code></pre>
   */
  @Override
  public MethodVisitor visitClassConstructor(MethodDeclInfo methodDeclInfo, ClassVisitor cv) {
    ClassName nestCompanion = nestDigest.nestCompanion(methodDeclInfo.methodKey.owner());
    MethodKey constructorBridge = methodDeclInfo.methodKey.bridgeOfConstructor(nestCompanion);
    MethodVisitor mv =
        cv.visitMethod(
            (methodDeclInfo.memberAccess & ~ACC_PRIVATE) | ACC_SYNTHETIC,
            constructorBridge.name(),
            constructorBridge.descriptor(),
            /* signature= */ null,
            methodDeclInfo.exceptions);
    mv.visitCode();
    mv.visitVarInsn(Opcodes.ALOAD, 0);

    Type[] constructorBridgeArgTypes = constructorBridge.getArgumentTypes();
    // Exclude last placeholder element loading.
    for (int i = 0, slotOffset = 1; i < constructorBridgeArgTypes.length - 1; i++) {
      mv.visitVarInsn(constructorBridgeArgTypes[i].getOpcode(Opcodes.ILOAD), slotOffset);
      slotOffset += constructorBridgeArgTypes[i].getSize();
    }
    mv.visitMethodInsn(
        Opcodes.INVOKESPECIAL,
        methodDeclInfo.methodKey.ownerName(),
        methodDeclInfo.methodKey.name(),
        methodDeclInfo.methodKey.descriptor(),
        /* isInterface= */ false);
    mv.visitInsn(Opcodes.RETURN);
    int slot = 0;
    for (Type bridgeConstructorArgType : constructorBridgeArgTypes) {
      slot += bridgeConstructorArgType.getSize();
    }
    mv.visitMaxs(slot, slot);
    mv.visitEnd();
    return mv;
  }

  /**
   * Emits a bridge method for a static method in a class. For example,
   *
   * <pre><code>
   *   class Foo {
   *     private static X execute(A a) {...}
   *
   *     &#047;&#047; Synthetic bridge method for a static method.
   *     static X execute$bridge(A a) {
   *       return execute(a);
   *     }
   *   }
   * </code></pre>
   */
  @Override
  public MethodVisitor visitClassStaticMethod(MethodDeclInfo methodDeclInfo, ClassVisitor cv) {
    MethodKey bridgeMethod = methodDeclInfo.methodKey.bridgeOfClassStaticMethod();
    MethodVisitor mv =
        cv.visitMethod(
            ACC_STATIC | ACC_SYNTHETIC,
            bridgeMethod.name(),
            bridgeMethod.descriptor(),
            /* signature= */ null,
            methodDeclInfo.exceptions);
    mv.visitCode();
    int slotOffset = 0;
    for (Type argType : bridgeMethod.getArgumentTypes()) {
      mv.visitVarInsn(argType.getOpcode(Opcodes.ILOAD), slotOffset);
      slotOffset += argType.getSize();
    }

    mv.visitMethodInsn(
        Opcodes.INVOKESTATIC,
        methodDeclInfo.methodKey.ownerName(),
        methodDeclInfo.methodKey.name(),
        methodDeclInfo.methodKey.descriptor(),
        /* isInterface= */ false);
    mv.visitInsn(bridgeMethod.getReturnType().getOpcode(Opcodes.IRETURN));
    mv.visitMaxs(slotOffset, slotOffset);
    mv.visitEnd();
    return mv;
  }

  /**
   * Emits a bridge method for an instance method in a class. For example,
   *
   * <pre><code>
   *   class Foo {
   *     private X execute(A a) {...}
   *
   *     &#047;&#047; Synthetic bridge method for a static method.
   *     static X execute$bridge(Foo foo, A a) {
   *       return foo.execute(a);
   *     }
   *   }
   * </code></pre>
   */
  @Override
  public MethodVisitor visitClassInstanceMethod(MethodDeclInfo methodDeclInfo, ClassVisitor cv) {
    MethodKey bridgeMethod = methodDeclInfo.methodKey.bridgeOfClassInstanceMethod();
    MethodVisitor mv =
        cv.visitMethod(
            ACC_STATIC | ACC_SYNTHETIC,
            bridgeMethod.name(),
            bridgeMethod.descriptor(),
            /* signature= */ null,
            methodDeclInfo.exceptions);
    mv.visitCode();
    int slotOffset = 0;
    for (Type argType : bridgeMethod.getArgumentTypes()) {
      mv.visitVarInsn(argType.getOpcode(Opcodes.ILOAD), slotOffset);
      slotOffset += argType.getSize();
    }

    mv.visitMethodInsn(
        Opcodes.INVOKESPECIAL,
        methodDeclInfo.methodKey.ownerName(),
        methodDeclInfo.methodKey.name(),
        methodDeclInfo.methodKey.descriptor(),
        /* isInterface= */ false);
    mv.visitInsn(bridgeMethod.getReturnType().getOpcode(Opcodes.IRETURN));
    mv.visitMaxs(slotOffset, slotOffset);
    mv.visitEnd();
    return mv;
  }

  /**
   * Rewrites the modifiers, name and descriptor of a declared interface static method.
   *
   * <p>For example,
   *
   * <p>The visitor converts
   *
   * <pre><code>
   *   package path.a.b;
   *
   *   interface Foo {
   *     private static X execute(A a) {...}
   *   }
   * </code></pre>
   *
   * to
   *
   * <pre><code>
   *   package path.a.b;
   *
   *   interface Foo {
   *     // The package path is mangled to the new method name to avoid name clashing for
   *     // interfaces with inheritance.
   *     static X execute(A a) {...}
   *   }
   * </code></pre>
   *
   * Note: The desugared method will be subsequently moved to the interface's companion class by
   * {@link com.google.devtools.build.android.desugar.InterfaceDesugaring}.
   */
  @Override
  public MethodVisitor visitInterfaceStaticMethod(MethodDeclInfo methodDeclInfo, ClassVisitor cv) {
    MethodKey methodSubstitute = methodDeclInfo.methodKey.substituteOfInterfaceStaticMethod();

    // Unset static and access modifier bits.
    return cv.visitMethod(
        (methodDeclInfo.memberAccess & ~0xf) | ACC_PUBLIC | ACC_STATIC,
        methodSubstitute.name(),
        methodSubstitute.descriptor(),
        /* signature= */ null,
        methodDeclInfo.exceptions);
  }

  /**
   * Rewrites the modifiers and header of a declared interface instance method. For example,
   *
   * <p>The visitor converts
   *
   * <pre><code>
   *   package path.a.b;
   *
   *   interface Foo {
   *     private X execute(A a) {...}
   *   }
   * </code></pre>
   *
   * to
   *
   * <pre><code>
   *   package path.a.b;
   *
   *   interface Foo {
   *     // The package path is mangled to the new method name to avoid name clashing for
   *     // interfaces with inheritance.
   *     static X path_a_b_execute(Foo foo, A a) {...}
   *   }
   * </code></pre>
   *
   * Note: The desugared method will be subsequently moved to the interface's companion class by
   * {@link com.google.devtools.build.android.desugar.InterfaceDesugaring}.
   */
  @Override
  public MethodVisitor visitInterfaceInstanceMethod(
      MethodDeclInfo methodDeclInfo, ClassVisitor cv) {
    MethodKey methodSubstitute = methodDeclInfo.methodKey.substituteOfInterfaceInstanceMethod();

    // Unset static and access modifier bits.
    return cv.visitMethod(
        (methodDeclInfo.memberAccess & ~0xf) | ACC_PUBLIC | ACC_STATIC,
        methodSubstitute.name(),
        methodSubstitute.descriptor(),
        /* signature= */ null,
        methodDeclInfo.exceptions);
  }
}
