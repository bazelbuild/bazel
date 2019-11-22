// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.android.desugar.nest.ClassMemberTrackReason.MemberUseKind;
import java.util.Arrays;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;

/** The key that indexes a class member, including fields, constructors and methods. */
public abstract class ClassMemberKey {

  /**
   * The class or interface that owns the class member, i.e. the immediate enclosing class of the
   * declaration site of a field, constructor or method.
   */
  public abstract String owner();

  /** The simple name of the class member. */
  public abstract String name();

  /** The descriptor of the class member. */
  public abstract String descriptor();

  /** Whether member key represents a constructor. */
  public final boolean isConstructor() {
    return "<init>".equals(name());
  }

  /** The binary name of the nest host that the member owner is affiliated with. */
  public final String nestHost() {
    return NestDesugarHelper.nestHost(owner());
  }

  /**
   * The binary name of the nest companion that the member owner is affiliated with. One nest has at
   * most one associated nest companion class.
   */
  final String nestCompanion() {
    return NestDesugarHelper.nestCompanion(owner());
  }

  /**
   * The binary simple name for a class member mangled with its owner name to avoid member name
   * duplication.
   */
  final String ownerMangledName() {
    return owner().replace('/', '_') + '$' + name();
  }

  /** The simple name with name suffix. */
  final String nameWithSuffix(String suffix) {
    return name() + '$' + suffix;
  }

  /** The key to index a class or interface field. */
  @AutoValue
  public abstract static class FieldKey extends ClassMemberKey {

    /** The factory method for {@link FieldKey}. */
    public static FieldKey create(String ownerClass, String name, String descriptor) {
      return new AutoValue_ClassMemberKey_FieldKey(ownerClass, name, descriptor);
    }

    /**
     * Accepts {@link FieldInstrVisitor} to perform distinct operations based on different
     * invocation codes.
     */
    public final <R, P> R accept(
        MemberUseKind fieldInstrOpcode,
        FieldInstrVisitor<R, ? super FieldKey, P> visitor,
        P param) {
      switch (fieldInstrOpcode) {
        case GETSTATIC:
          return visitor.visitGetStatic(this, param);
        case PUTSTATIC:
          return visitor.visitPutStatic(this, param);
        case GETFIELD:
          return visitor.visitGetField(this, param);
        case PUTFIELD:
          return visitor.visitPutField(this, param);
        default:
          throw new AssertionError(
              String.format(
                  "Unexpected opcode(%s): Expect one of {GETSTATIC, PUTSTATIC, GETFIELD, PUTFIELD}"
                      + " for field instructions.",
                  fieldInstrOpcode));
      }
    }

    /**
     * Returns the bridge method for reading a static field, identified by (getstatic) instruction.
     */
    public final MethodKey bridgeOfStaticRead() {
      return MethodKey.create(
          owner(), nameWithSuffix("bridge_getter"), Type.getMethodDescriptor(getFieldType()));
    }

    /**
     * Returns the bridge method for reading an instance field, identified by (getfield)
     * instruction.
     */
    public final MethodKey bridgeOfInstanceRead() {
      return MethodKey.create(
          owner(),
          nameWithSuffix("bridge_getter"),
          Type.getMethodDescriptor(getFieldType(), Type.getObjectType(owner())));
    }

    /**
     * Returns the bridge method for writing a static field, identified by (putstatic) instruction.
     */
    public final MethodKey bridgeOfStaticWrite() {
      return MethodKey.create(
          owner(),
          nameWithSuffix("bridge_setter"),
          Type.getMethodDescriptor(getFieldType(), getFieldType()));
    }

    /**
     * Returns the bridge method for writing an instance field, identified by (putfield)
     * instruction.
     */
    public final MethodKey bridgeOfInstanceWrite() {
      return MethodKey.create(
          owner(),
          nameWithSuffix("bridge_setter"),
          Type.getMethodDescriptor(getFieldType(), Type.getObjectType(owner()), getFieldType()));
    }

    public Type getFieldType() {
      return Type.getType(descriptor());
    }
  }

  /** The key to index a class or interface method or constructor. */
  @AutoValue
  public abstract static class MethodKey extends ClassMemberKey {

    /** The factory method for {@link MethodKey}. */
    public static MethodKey create(String ownerClass, String name, String descriptor) {
      return new AutoValue_ClassMemberKey_MethodKey(ownerClass, name, descriptor);
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
    final MethodKey bridgeOfClassStaticMethod() {
      checkState(!isConstructor(), "Expect a non-constructor method but is a constructor %s", this);
      return MethodKey.create(owner(), nameWithSuffix("bridge"), descriptor());
    }

    /** The synthetic bridge method for a private instance method in a class. */
    final MethodKey bridgeOfClassInstanceMethod() {
      return MethodKey.create(
          owner(), nameWithSuffix("bridge"), instanceMethodToStaticDescriptor(this));
    }

    /** The substitute method for a private static method in an interface. */
    final MethodKey substituteOfInterfaceStaticMethod() {
      checkState(!isConstructor(), "Expect a non-constructor: %s", this);
      return MethodKey.create(owner(), ownerMangledName(), descriptor());
    }

    /** The substitute method for a private instance method in an interface. */
    final MethodKey substituteOfInterfaceInstanceMethod() {
      return MethodKey.create(owner(), ownerMangledName(), instanceMethodToStaticDescriptor(this));
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

  /** A unit data object represents a class or interface declaration. */
  // TODO(deltazulu): Consider @AutoValue-ize this class. (String[] as attribute is not supported).
  static final class MethodDeclInfo {
    final MethodKey methodKey;
    final int ownerAccess;
    final int memberAccess;
    final String signature;
    final String[] exceptions;

    MethodDeclInfo(
        MethodKey methodKey,
        int ownerAccess,
        int memberAccess,
        String signature,
        String[] exceptions) {
      this.methodKey = methodKey;
      this.ownerAccess = ownerAccess;
      this.memberAccess = memberAccess;
      this.signature = signature;
      this.exceptions = exceptions;
    }

    public final <R, P> R accept(MethodDeclVisitor<R, ? super MethodDeclInfo, P> visitor, P param) {
      if (methodKey.isConstructor()) {
        return visitor.visitClassConstructor(this, param);
      }
      boolean isInterface = (ownerAccess & Opcodes.ACC_INTERFACE) != 0;
      boolean isStatic = (memberAccess & Opcodes.ACC_STATIC) != 0;
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
  }

  /** A visitor for all method invocation instructions in combination with method types. */
  public interface MethodInstrVisitor<R, K extends MethodKey, P> {

    R visitInvokeVirtual(K methodKey, P param);

    R visitInvokeSpecial(K methodKey, P param);

    R visitConstructorInvokeSpecial(K methodKey, P param);

    R visitInterfaceInvokeSpecial(K methodKey, P param);

    R visitInvokeStatic(K methodKey, P param);

    R visitInterfaceInvokeStatic(K methodKey, P param);

    R visitInvokeInterface(K methodKey, P param);

    R visitInvokeDynamic(K methodKey, P param);
  }

  /** Visits all field access instructions. */
  public interface FieldInstrVisitor<R, K extends FieldKey, P> {

    R visitGetStatic(K fieldKey, P param);

    R visitPutStatic(K fieldKey, P param);

    R visitGetField(K fieldKey, P param);

    R visitPutField(K fieldKey, P param);
  }

  /** A visitor that directs different operations based on the method types. */
  public interface MethodDeclVisitor<R, K extends MethodDeclInfo, P> {

    R visitClassConstructor(K methodDeclInfo, P param);

    R visitClassStaticMethod(K methodDeclInfo, P param);

    R visitClassInstanceMethod(K methodDeclInfo, P param);

    R visitInterfaceStaticMethod(K methodDeclInfo, P param);

    R visitInterfaceInstanceMethod(K methodDeclInfo, P param);
  }
}
