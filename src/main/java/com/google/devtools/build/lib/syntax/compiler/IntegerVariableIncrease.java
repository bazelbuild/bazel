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

import net.bytebuddy.implementation.Implementation.Context;
import net.bytebuddy.implementation.bytecode.StackManipulation;

import org.objectweb.asm.MethodVisitor;

/**
 * A {@link StackManipulation} that increases a given integer variable of the method by a given
 * amount.
 */
public final class IntegerVariableIncrease implements StackManipulation {

  private final Variable variable;
  private final int increment;

  public IntegerVariableIncrease(Variable variable, int increment) {
    this.variable = variable;
    this.increment = increment;
  }

  @Override
  public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
    methodVisitor.visitIincInsn(variable.index, increment);
    return new Size(0, 0);
  }

  @Override
  public boolean isValid() {
    return true;
  }

  @Override
  public String toString() {
    return "IntInc(" + variable.index + ", " + increment + ")";
  }
}
