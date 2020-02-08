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

package com.google.devtools.build.android.desugar.langmodel;

import static org.objectweb.asm.Opcodes.ACC_PRIVATE;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** A unit data object represents a class or interface declaration. */
@AutoValue
public abstract class MethodDeclInfo implements TypeMappable<MethodDeclInfo> {

  public abstract MethodKey methodKey();

  public abstract int ownerAccess();

  public abstract int memberAccess();

  @Nullable
  public abstract String signature();

  public abstract ImmutableList<String> exceptions();

  public static MethodDeclInfo create(
      MethodKey methodKey,
      int ownerAccess,
      int memberAccess,
      @Nullable String signature,
      @Nullable String[] exceptions) {
    return create(
        methodKey,
        ownerAccess,
        memberAccess,
        signature,
        exceptions == null ? ImmutableList.of() : ImmutableList.copyOf(exceptions));
  }

  private static MethodDeclInfo create(
      MethodKey methodKey,
      int ownerAccess,
      int memberAccess,
      String signature,
      ImmutableList<String> exceptions) {
    return new AutoValue_MethodDeclInfo(
        methodKey, ownerAccess, memberAccess, signature, exceptions);
  }

  public final ClassName owner() {
    return methodKey().owner();
  }

  public final String ownerName() {
    return methodKey().ownerName();
  }

  public final String name() {
    return methodKey().name();
  }

  public final String descriptor() {
    return methodKey().descriptor();
  }

  public final Type returnType() {
    return methodKey().getReturnType();
  }

  public final ImmutableList<Type> argumentTypes() {
    return ImmutableList.copyOf(methodKey().getArgumentTypes());
  }

  public final String[] exceptionArray() {
    return exceptions().toArray(new String[0]);
  }

  /** The synthetic constructor for a private constructor. */
  public final MethodDeclInfo bridgeOfConstructor(ClassName nestCompanion) {
    return create(
        methodKey().bridgeOfConstructor(nestCompanion),
        ownerAccess(),
        (memberAccess() & ~ACC_PRIVATE) | ACC_SYNTHETIC,
        /* signature= */ null,
        exceptions());
  }

  /** The synthetic bridge method for a private static method in a class. */
  public final MethodDeclInfo bridgeOfClassStaticMethod() {
    return create(
        methodKey().bridgeOfClassStaticMethod(),
        ownerAccess(),
        ACC_STATIC | ACC_SYNTHETIC,
        /* signature= */ null,
        exceptions());
  }

  /** The synthetic bridge method for a private instance method in a class. */
  public final MethodDeclInfo bridgeOfClassInstanceMethod() {
    return create(
        methodKey().bridgeOfClassInstanceMethod(),
        ownerAccess(),
        /* memberAccess= */ ACC_STATIC | ACC_SYNTHETIC,
        /* signature= */ null,
        exceptions());
  }

  /** The substitute method for a private static method in an interface. */
  public final MethodDeclInfo substituteOfInterfaceStaticMethod() {
    return create(
        methodKey().substituteOfInterfaceStaticMethod(),
        ownerAccess(),
        (memberAccess() & ~0xf) | ACC_PUBLIC | ACC_STATIC,
        /* signature= */ null,
        exceptions());
  }

  /** The substitute method for a private instance method in an interface. */
  public final MethodDeclInfo substituteOfInterfaceInstanceMethod() {
    return create(
        methodKey().substituteOfInterfaceInstanceMethod(),
        ownerAccess(),
        (memberAccess() & ~0xf) | ACC_PUBLIC | ACC_STATIC, // Unset static and access modifier bits.
        /* signature= */ null,
        exceptions());
  }

  public final MethodVisitor accept(ClassVisitor cv) {
    return cv.visitMethod(
        memberAccess(),
        methodKey().name(),
        methodKey().descriptor(),
        signature(),
        exceptionArray());
  }

  public final <R, P> R accept(MethodDeclVisitor<R, ? super MethodDeclInfo, P> visitor, P param) {
    if (methodKey().isConstructor()) {
      return visitor.visitClassConstructor(this, param);
    }
    boolean isInterface = (ownerAccess() & Opcodes.ACC_INTERFACE) != 0;
    boolean isStatic = (memberAccess() & Opcodes.ACC_STATIC) != 0;
    if (isInterface) {
      return isStatic
          ? visitor.visitInterfaceStaticMethod(this, param)
          : visitor.visitInterfaceInstanceMethod(this, param);
    } else {
      return isStatic
          ? visitor.visitClassStaticMethod(this, param)
          : visitor.visitClassInstanceMethod(this, param);
    }
  }

  @Override
  public MethodDeclInfo acceptTypeMapper(TypeMapper typeMapper) {
    return create(
        methodKey().acceptTypeMapper(typeMapper),
        ownerAccess(),
        memberAccess(),
        signature(),
        exceptions());
  }
}
