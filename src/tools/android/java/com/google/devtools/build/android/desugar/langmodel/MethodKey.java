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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import org.objectweb.asm.Type;

/** The key to index a class or interface method or constructor. */
@AutoValue
public abstract class MethodKey extends ClassMemberKey<MethodKey> {

  /** The factory method for {@link MethodKey}. */
  public static MethodKey create(ClassName ownerClass, String name, String descriptor) {
    checkState(
        descriptor.isEmpty() // Allows empty descriptor for non-overloaded methods.
            || descriptor.startsWith("("),
        "Expected a method descriptor. Actual: (%s#%s:%s)",
        ownerClass,
        name,
        descriptor);
    return new AutoValue_MethodKey(ownerClass, name, descriptor);
  }

  /** The return type of a method. */
  public Type getReturnType() {
    return Type.getReturnType(descriptor());
  }

  /** The return type of a method. */
  public ClassName getReturnTypeName() {
    return ClassName.create(Type.getReturnType(descriptor()));
  }

  /** The formal parameter types of a method. */
  public Type[] getArgumentTypeArray() {
    return Type.getArgumentTypes(descriptor());
  }

  /** The formal parameter types of a method. */
  public ImmutableList<Type> getArgumentTypes() {
    return ImmutableList.copyOf(getArgumentTypeArray());
  }

  /** The formal parameter type names of a method. */
  public ImmutableList<ClassName> getArgumentTypeNames() {
    return getArgumentTypes().stream().map(ClassName::create).collect(toImmutableList());
  }

  public MethodKey toArgumentTypeAdapter(boolean fromStaticOrigin) {
    ClassName typeAdapterOwner = owner().typeAdapterOwner();
    checkState(
        !isConstructor(), "Argument type adapter for constructor is not supported: %s. ", this);

    return MethodKey.create(
            typeAdapterOwner,
            name(),
            fromStaticOrigin ? descriptor() : instanceMethodToStaticDescriptor())
        .acceptTypeMapper(ClassName.DELIVERY_TYPE_MAPPER);
  }

  /** The synthetic constructor for a private constructor. */
  public final MethodKey bridgeOfConstructor(ClassName nestCompanion) {
    checkState(isConstructor(), "Expect to use for a constructor but is %s", this);
    Type companionClassType = nestCompanion.toAsmObjectType();
    ImmutableList<Type> argumentTypes = getArgumentTypes();
    ImmutableList<Type> bridgeConstructorArgTypes =
        ImmutableList.<Type>builder().addAll(argumentTypes).add(companionClassType).build();
    return create(
        owner(),
        name(),
        Type.getMethodDescriptor(getReturnType(), bridgeConstructorArgTypes.toArray(new Type[0])));
  }

  /** The synthetic bridge method for a private static method in a class. */
  public final MethodKey bridgeOfClassStaticMethod() {
    checkState(!isConstructor(), "Expect a non-constructor method but is a constructor %s", this);
    return create(owner(), nameWithSuffix("bridge"), descriptor());
  }

  /** The synthetic bridge method for a private instance method in a class. */
  public final MethodKey bridgeOfClassInstanceMethod() {
    return create(owner(), nameWithSuffix("bridge"), this.instanceMethodToStaticDescriptor());
  }

  /** The substitute method for a private static method in an interface. */
  public final MethodKey substituteOfInterfaceStaticMethod() {
    checkState(!isConstructor(), "Expect a non-constructor: %s", this);
    return create(owner(), name(), descriptor());
  }

  /** The substitute method for a private instance method in an interface. */
  public final MethodKey substituteOfInterfaceInstanceMethod() {
    return create(owner(), name(), this.instanceMethodToStaticDescriptor());
  }

  /** The descriptor of the static version of a given instance method. */
  private String instanceMethodToStaticDescriptor() {
    checkState(!isConstructor(), "Expect a Non-constructor method: %s", this);
    ImmutableList<Type> argumentTypes = getArgumentTypes();
    ImmutableList<Type> bridgeMethodArgTypes =
        ImmutableList.<Type>builder().add(ownerAsmObjectType()).addAll(argumentTypes).build();
    return Type.getMethodDescriptor(getReturnType(), bridgeMethodArgTypes.toArray(new Type[0]));
  }

  @Override
  public MethodKey acceptTypeMapper(TypeMapper typeMapper) {
    return MethodKey.create(typeMapper.map(owner()), name(), typeMapper.mapDesc(descriptor()));
  }

  /**
   * Accepts a {@link MethodInstrVisitor} that visits all kinds of method invocation instructions.
   */
  public final <R, P> R accept(
      MemberUseKind methodInstrOpcode,
      boolean isInterface,
      MethodInstrVisitor<R, ? super MethodKey, P> visitor,
      P param) {
    if (isConstructor()) {
      checkState(methodInstrOpcode == MemberUseKind.INVOKESPECIAL);
      return visitor.visitConstructorInvokeSpecial(this, param);
    } else if (isInterface) {
      switch (methodInstrOpcode) {
        case INVOKESPECIAL:
          // Super call to an non-abstract instance interface method.
          return visitor.visitInterfaceInvokeSpecial(this, param);
        case INVOKESTATIC:
          return visitor.visitInterfaceInvokeStatic(this, param);
        case INVOKEINTERFACE:
          return visitor.visitInvokeInterface(this, param);
        default:
          throw new AssertionError(
              String.format(
                  "Unexpected Opcode: (%s) invoked on Interface Method(%s).\n"
                      + "Opcode Reference: {INVOKEVIRTUAL(182), INVOKESPECIAL(183),"
                      + " INVOKESTATIC(184), INVOKEINTERFACE(185), INVOKEDYNAMIC(186)}",
                  methodInstrOpcode, this));
      }
    } else {
      switch (methodInstrOpcode) {
        case INVOKEVIRTUAL:
          return visitor.visitInvokeVirtual(this, param);
        case INVOKESPECIAL:
          return visitor.visitInvokeSpecial(this, param);
        case INVOKESTATIC:
          return visitor.visitInvokeStatic(this, param);
        case INVOKEDYNAMIC:
          return visitor.visitInvokeDynamic(this, param);
        default:
          throw new AssertionError(
              String.format(
                  "Unexpected Opcode: (%s) invoked on class Method(%s).\n"
                      + "Opcode Reference: {INVOKEVIRTUAL(182), INVOKESPECIAL(183),"
                      + " INVOKESTATIC(184), INVOKEINTERFACE(185), INVOKEDYNAMIC(186)}",
                  methodInstrOpcode, this));
      }
    }
  }
}
