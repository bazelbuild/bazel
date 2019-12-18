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

import com.google.auto.value.AutoValue;
import java.util.Arrays;
import org.objectweb.asm.Type;

/** The key to index a class or interface method or constructor. */
@AutoValue
public abstract class MethodKey extends ClassMemberKey {

  /** The factory method for {@link MethodKey}. */
  public static MethodKey create(String ownerClass, String name, String descriptor) {
    return new AutoValue_MethodKey(ownerClass, name, descriptor);
  }

  /** The return type of a method. */
  public Type getReturnType() {
    return Type.getReturnType(descriptor());
  }

  /** The formal parameter types of a method. */
  public Type[] getArgumentTypes() {
    return Type.getArgumentTypes(descriptor());
  }

  /** The synthetic constructor for a private constructor. */
  public final MethodKey bridgeOfConstructor() {
    checkState(isConstructor(), "Expect to use for a constructor but is %s", this);
    Type companionClassType = Type.getObjectType(nestCompanion());
    Type[] argumentTypes = getArgumentTypes();
    Type[] bridgeConstructorArgTypes = Arrays.copyOf(argumentTypes, argumentTypes.length + 1);
    bridgeConstructorArgTypes[argumentTypes.length] = companionClassType;
    return MethodKey.create(
        owner(), name(), Type.getMethodDescriptor(getReturnType(), bridgeConstructorArgTypes));
  }

  /** The synthetic bridge method for a private static method in a class. */
  public final MethodKey bridgeOfClassStaticMethod() {
    checkState(!isConstructor(), "Expect a non-constructor method but is a constructor %s", this);
    return MethodKey.create(owner(), nameWithSuffix("bridge"), descriptor());
  }

  /** The synthetic bridge method for a private instance method in a class. */
  public final MethodKey bridgeOfClassInstanceMethod() {
    return MethodKey.create(
        owner(), nameWithSuffix("bridge"), instanceMethodToStaticDescriptor(this));
  }

  /** The substitute method for a private static method in an interface. */
  public final MethodKey substituteOfInterfaceStaticMethod() {
    checkState(!isConstructor(), "Expect a non-constructor: %s", this);
    return MethodKey.create(owner(), name(), descriptor());
  }

  /** The substitute method for a private instance method in an interface. */
  public final MethodKey substituteOfInterfaceInstanceMethod() {
    return MethodKey.create(owner(), name(), instanceMethodToStaticDescriptor(this));
  }

  /** The descriptor of the static version of a given instance method. */
  private static String instanceMethodToStaticDescriptor(MethodKey methodKey) {
    checkState(!methodKey.isConstructor(), "Expect a Non-constructor method: %s", methodKey);
    Type[] argumentTypes = methodKey.getArgumentTypes();
    Type[] bridgeMethodArgTypes = new Type[argumentTypes.length + 1];
    bridgeMethodArgTypes[0] = Type.getObjectType(methodKey.owner());
    System.arraycopy(argumentTypes, 0, bridgeMethodArgTypes, 1, argumentTypes.length);
    return Type.getMethodDescriptor(methodKey.getReturnType(), bridgeMethodArgTypes);
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
