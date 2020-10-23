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

import static org.objectweb.asm.Opcodes.ACC_INTERFACE;
import static org.objectweb.asm.Opcodes.ACC_PRIVATE;
import static org.objectweb.asm.Opcodes.ACC_PROTECTED;
import static org.objectweb.asm.Opcodes.ACC_PUBLIC;
import static org.objectweb.asm.Opcodes.ACC_STATIC;
import static org.objectweb.asm.Opcodes.ACC_SYNTHETIC;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** A unit data object represents a class or interface declaration. */
@AutoValue
public abstract class MethodDeclInfo
    implements TypeMappable<MethodDeclInfo>, Comparable<MethodDeclInfo> {

  public abstract MethodKey methodKey();

  public abstract int ownerAccess();

  public abstract int memberAccess();

  @Nullable
  public abstract String signature();

  public abstract ImmutableList<String> exceptions();

  public static MethodDeclInfoBuilder builder() {
    return new AutoValue_MethodDeclInfo.Builder()
        .setSignature(null)
        .setExceptions(ImmutableList.of());
  }

  public static MethodDeclInfo create(
      MethodKey methodKey,
      int ownerAccess,
      int memberAccess,
      @Nullable String signature,
      @Nullable String[] exceptions) {
    return builder()
        .setMethodKey(methodKey)
        .setOwnerAccess(ownerAccess)
        .setMemberAccess(memberAccess)
        .setSignature(signature)
        .setExceptionArray(exceptions)
        .build();
  }

  public abstract MethodDeclInfoBuilder toBuilder();

  public final ClassName owner() {
    return methodKey().owner();
  }

  public final String ownerName() {
    return methodKey().ownerName();
  }

  public final String packageName() {
    return owner().getPackageName();
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

  public final ClassName returnTypeName() {
    return methodKey().getReturnTypeName();
  }

  public final ImmutableList<Type> argumentTypes() {
    return ImmutableList.copyOf(methodKey().getArgumentTypes());
  }

  public final ImmutableList<ClassName> argumentTypeNames() {
    return ImmutableList.copyOf(methodKey().getArgumentTypeNames());
  }

  public final ImmutableSet<ClassName> headerTypeNameSet() {
    return methodKey().getHeaderTypeNameSet();
  }

  public final boolean isStaticMethod() {
    return (memberAccess() & ACC_STATIC) != 0;
  }

  public final boolean isPrivateAccess() {
    return (memberAccess() & ACC_PRIVATE) != 0;
  }

  public final boolean isPackageAccess() {
    return (memberAccess() & (ACC_PROTECTED | ACC_PRIVATE | ACC_PUBLIC)) == 0;
  }

  public final boolean isProtectedAccess() {
    return (memberAccess() & ACC_PROTECTED) != 0;
  }

  public final boolean isPublicAccess() {
    return (memberAccess() & ACC_PUBLIC) != 0;
  }

  public final boolean isInterfaceMethod() {
    return (ownerAccess() & ACC_INTERFACE) != 0;
  }

  public final String[] exceptionArray() {
    return exceptions().toArray(new String[0]);
  }

  /** The synthetic constructor for a private constructor. */
  public final MethodDeclInfo bridgeOfConstructor(ClassName nestCompanion) {
    int memberAccess = (memberAccess() & ~ACC_PRIVATE) | ACC_SYNTHETIC;
    return toBuilder()
        .setMethodKey(methodKey().bridgeOfConstructor(nestCompanion))
        .setMemberAccess(memberAccess)
        .build();
  }

  /** The synthetic bridge method for a private static method in a class. */
  public final MethodDeclInfo bridgeOfClassStaticMethod() {
    return toBuilder()
        .setMethodKey(methodKey().bridgeOfClassStaticMethod())
        .setMemberAccess(ACC_STATIC | ACC_SYNTHETIC)
        .build();
  }

  /** The synthetic bridge method for a private instance method in a class. */
  public final MethodDeclInfo bridgeOfClassInstanceMethod() {
    return toBuilder()
        .setMethodKey(methodKey().bridgeOfClassInstanceMethod())
        .setMemberAccess(ACC_STATIC | ACC_SYNTHETIC)
        .build();
  }

  /** The substitute method for a private static method in an interface. */
  public final MethodDeclInfo substituteOfInterfaceStaticMethod() {
    return toBuilder()
        .setMethodKey(methodKey().substituteOfInterfaceStaticMethod())
        .setMemberAccess((memberAccess() & ~0xf) | ACC_PUBLIC | ACC_STATIC)
        .build();
  }

  /** The substitute method for a private instance method in an interface. */
  public final MethodDeclInfo substituteOfInterfaceInstanceMethod() {
    // Unset static and access modifier bits.
    return toBuilder()
        .setMethodKey(methodKey().substituteOfInterfaceInstanceMethod())
        .setMemberAccess((memberAccess() & ~0xf) | ACC_PUBLIC | ACC_STATIC)
        .build();
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
    return toBuilder()
        .setMethodKey(methodKey().acceptTypeMapper(typeMapper))
        .setSignature(typeMapper.mapSignature(signature(), /* typeSignature= */ false))
        .setExceptionArray(typeMapper.mapTypes(exceptionArray()))
        .build();
  }

  @Override
  public int compareTo(MethodDeclInfo other) {
    return methodKey().compareTo(other.methodKey());
  }

  /** The builder for {@link MethodDeclInfo}. */
  @AutoValue.Builder
  public abstract static class MethodDeclInfoBuilder {

    public abstract MethodDeclInfoBuilder setMethodKey(MethodKey value);

    public abstract MethodDeclInfoBuilder setOwnerAccess(int value);

    public abstract MethodDeclInfoBuilder setMemberAccess(int value);

    public abstract MethodDeclInfoBuilder setSignature(String value);

    public abstract MethodDeclInfoBuilder setExceptions(ImmutableList<String> value);

    public final MethodDeclInfoBuilder setExceptionArray(String[] exceptions) {
      return setExceptions(
          exceptions == null ? ImmutableList.of() : ImmutableList.copyOf(exceptions));
    }

    public abstract MethodDeclInfo build();
  }
}
