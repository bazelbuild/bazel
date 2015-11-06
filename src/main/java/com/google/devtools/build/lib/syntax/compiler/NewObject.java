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

import net.bytebuddy.description.method.MethodDescription.ForLoadedConstructor;
import net.bytebuddy.implementation.Implementation.Context;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.member.MethodInvocation;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.util.List;

/**
 * Builds byte code for new object creation from constructor calls.
 *
 * <p>Java byte code for new object creation looks like the following pseudo code:
 * <pre>
 * NEW
 * DUP
 * ...load constructor parameters...
 * INVOKESPECIAL constructor
 * </pre>
 * This is because a constructor is actually called on the reference to the object itself, which
 * is left on the stack by NEW.
 *
 * This class helps with wrapping the parameter loading with this structure.
 */
public class NewObject implements StackManipulation {

  private final ForLoadedConstructor constructor;
  private final StackManipulation arguments;

  /**
   * Intermediate state builder for new object construction with missing constructor argument
   * loading.
   */
  public static final class NewObjectBuilder {
    private final ForLoadedConstructor constructor;

    private NewObjectBuilder(ForLoadedConstructor constructor) {
      this.constructor = constructor;
    }

    /**
     * Adds the argument loading in the correct for new object construction.
     */
    public NewObject arguments(StackManipulation... arguments) {
      return new NewObject(constructor, new StackManipulation.Compound(arguments));
    }

    /**
     * Adds the argument loading in the correct for new object construction.
     */
    public NewObject arguments(List<StackManipulation> arguments) {
      return new NewObject(constructor, new StackManipulation.Compound(arguments));
    }
  }

  private NewObject(ForLoadedConstructor constructor, StackManipulation arguments) {
    this.constructor = constructor;
    this.arguments = arguments;
  }

  /**
   * Looks for a constructor in the class with the given parameter types and returns an
   * intermediate builder.
   */
  public static NewObjectBuilder fromConstructor(Class<?> clazz, Class<?>... parameterTypes) {
    return new NewObjectBuilder(ReflectionUtils.getConstructor(clazz, parameterTypes));
  }

  @Override
  public boolean isValid() {
    return true;
  }

  @Override
  public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
    methodVisitor.visitTypeInsn(Opcodes.NEW, constructor.getDeclaringType().getInternalName());
    return new StackManipulation.Compound(
            Duplication.SINGLE, arguments, MethodInvocation.invoke(constructor))
        .apply(methodVisitor, implementationContext);
  }

  @Override
  public String toString() {
    return "NewObject(" + constructor + ", " + arguments + ")";
  }
}
