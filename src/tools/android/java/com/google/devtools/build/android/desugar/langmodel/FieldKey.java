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

import com.google.auto.value.AutoValue;
import org.objectweb.asm.Type;

/** The key to index a class or interface field. */
@AutoValue
public abstract class FieldKey extends ClassMemberKey {

  /** The factory method for {@link FieldKey}. */
  public static FieldKey create(String ownerClass, String name, String descriptor) {
    return new AutoValue_FieldKey(ownerClass, name, descriptor);
  }

  /**
   * Accepts {@link FieldInstrVisitor} to perform distinct operations based on different invocation
   * codes.
   */
  public final <R, P> R accept(
      MemberUseKind fieldUseKind, FieldInstrVisitor<R, ? super FieldKey, P> visitor, P param) {
    switch (fieldUseKind) {
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
                fieldUseKind));
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
   * Returns the bridge method for reading an instance field, identified by (getfield) instruction.
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
   * Returns the bridge method for writing an instance field, identified by (putfield) instruction.
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
