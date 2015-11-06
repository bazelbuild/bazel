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

import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;

/**
 * Adds a fresh label to the byte code.
 */
public class LabelAdder implements StackManipulation {

  private final Label label;

  public LabelAdder() {
    this.label = new Label();
  }

  public Label getLabel() {
    return label;
  }

  @Override
  public boolean isValid() {
    return true;
  }

  @Override
  public Size apply(MethodVisitor methodVisitor, Context implementationContext) {
    methodVisitor.visitLabel(label);
    return new Size(0, 0);
  }

  @Override
  public String toString() {
    return "Label(" + label + ")";
  }
}
