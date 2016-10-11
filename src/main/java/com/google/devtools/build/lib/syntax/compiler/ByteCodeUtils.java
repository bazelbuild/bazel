// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax.compiler;

import com.google.devtools.build.lib.syntax.compiler.Jump.PrimitiveComparison;
import java.util.List;
import net.bytebuddy.description.method.MethodDescription.ForLoadedMethod;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.description.type.generic.GenericTypeDescription;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender.Compound;
import net.bytebuddy.implementation.bytecode.Removal;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.FieldAccess;
import net.bytebuddy.implementation.bytecode.member.MethodInvocation;

/**
 * Various utility methods for byte code generation.
 */
public class ByteCodeUtils {

  /**
   * Helper method to wrap {@link StackManipulation}s into a {@link ByteCodeAppender} and add it
   * to a list of appenders.
   */
  public static void append(List<ByteCodeAppender> code, StackManipulation... manipulations) {
    code.add(new ByteCodeAppender.Simple(manipulations));
  }

  /**
   * As {@link #invoke(Class, String, Class...)} and additionally clears the returned value from
   * the stack, if any.
   */
  public static StackManipulation cleanInvoke(
      Class<?> clazz, String methodName, Class<?>... parameterTypes) {
    ForLoadedMethod method = ReflectionUtils.getMethod(clazz, methodName, parameterTypes);
    GenericTypeDescription returnType = method.getReturnType();
    if (returnType.equals(TypeDescription.VOID)) {
      return MethodInvocation.invoke(method);
    }
    return new StackManipulation.Compound(
        MethodInvocation.invoke(method), Removal.pop(returnType.asErasure()));
  }

  /**
   * Create a {@link ByteCodeAppender} applying a list of them.
   *
   * <p>Exists just because {@link Compound} does not have a constructor taking a list.
   */
  public static ByteCodeAppender compoundAppender(List<ByteCodeAppender> code) {
    return new Compound(code.toArray(new ByteCodeAppender[0]));
  }

  /**
   * Builds a {@link StackManipulation} that loads the field.
   */
  public static StackManipulation getField(Class<?> clazz, String field) {
    return FieldAccess.forField(ReflectionUtils.getField(clazz, field)).getter();
  }

  /**
   * Build a {@link StackManipulation} that logically negates an integer on the stack.
   *
   * <p>Java byte code does not have an instruction for this, so this produces a conditional jump
   * which puts 0/1 on the stack.
   */
  public static StackManipulation intLogicalNegation() {
    return intToPrimitiveBoolean(PrimitiveComparison.NOT_EQUAL);
  }

  /**
   * Build a {@link StackManipulation} that converts an integer to 0/1 depending on a comparison
   * with 0.
   */
  public static StackManipulation intToPrimitiveBoolean(PrimitiveComparison operator) {
    LabelAdder afterLabel = new LabelAdder();
    LabelAdder putFalseLabel = new LabelAdder();
    return new StackManipulation.Compound(
        Jump.ifIntOperandToZero(operator).to(putFalseLabel),
        // otherwise put "false" on the stack and jump to end
        IntegerConstant.ONE,
        Jump.to(afterLabel.getLabel()),
        // add label for "else" and put "true" on the stack
        putFalseLabel,
        IntegerConstant.ZERO,
        afterLabel);
  }

  /**
   * Builds a {@link StackManipulation} that invokes the method identified via reflection on the
   * given class, method and parameter types.
   */
  public static StackManipulation invoke(
      Class<?> clazz, String methodName, Class<?>... parameterTypes) {
    return MethodInvocation.invoke(ReflectionUtils.getMethod(clazz, methodName, parameterTypes));
  }
}
