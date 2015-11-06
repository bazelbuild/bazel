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

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender.Compound;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.FieldAccess;
import net.bytebuddy.implementation.bytecode.member.MethodInvocation;

import java.util.List;

/**
 * Various utility methods for byte code generation.
 */
public class ByteCodeUtils {

  /**
   * Create a {@link ByteCodeAppender} applying a list of them.
   *
   * <p>Exists just because {@link Compound} does not have a constructor taking a list.
   */
  public static ByteCodeAppender compoundAppender(List<ByteCodeAppender> code) {
    return new Compound(code.toArray(new ByteCodeAppender[code.size()]));
  }

  /**
   * Builds a {@link StackManipulation} that loads the field.
   */
  public static StackManipulation getField(Class<?> clazz, String field) {
    return FieldAccess.forField(ReflectionUtils.getField(clazz, field)).getter();
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
