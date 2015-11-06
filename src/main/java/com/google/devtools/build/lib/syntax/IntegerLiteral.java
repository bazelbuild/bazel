// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;

/**
 * Syntax node for an integer literal.
 */
public final class IntegerLiteral extends Literal<Integer> {

  public IntegerLiteral(Integer value) {
    super(value);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
  }

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) {
    return new ByteCodeAppender.Simple(
        IntegerConstant.forValue(value),
        ByteCodeMethodCalls.BCInteger.valueOf);
  }
}
