// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.Jump;
import com.google.devtools.build.lib.syntax.compiler.LoopLabels;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.Preconditions;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

/**
 * A class for flow statements (e.g. break and continue)
 */
public final class FlowStatement extends Statement {
  public enum Kind {
    BREAK("break"),
    CONTINUE("continue");

    private final String name;

    private Kind(String name) {
      this.name = name;
    }
  }

  private final Kind kind;
  private final FlowException ex;

  /**
   *
   * @param kind The label of the statement (either break or continue)
   */
  public FlowStatement(Kind kind) {
    this.kind = kind;
    this.ex = new FlowException(kind);
  }

  Kind getKind() {
    return kind;
  }

  @Override
  void doExec(Environment env) throws EvalException {
    throw ex;
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    if (!env.isInsideLoop()) {
      throw new EvalException(getLocation(), kind.name + " statement must be inside a for loop");
    }
  }

  @Override
  public String toString() {
    return kind.name;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  ByteCodeAppender compile(
      VariableScope scope, Optional<LoopLabels> loopLabels, DebugInfo debugInfo) {
    Preconditions.checkArgument(loopLabels.isPresent(), "break/continue not within loop");
    return new ByteCodeAppender.Simple(Jump.to(loopLabels.get().labelFor(kind)));
  }

  /**
   * An exception that signals changes in the control flow (e.g. break or continue)
   */
  class FlowException extends EvalException {
    private final Kind kind;

    public FlowException(Kind kind) {
      super(FlowStatement.this.getLocation(), "FlowException with kind = " + kind.name);
      this.kind = kind;
    }

    /**
     * Returns whether the enclosing loop should be terminated completely (break)
     *
     * @return {@code True} for 'break', {@code false} for 'continue'
     */
    public boolean mustTerminateLoop() {
      return kind == Kind.BREAK;
    }

    @Override
    public boolean canBeAddedToStackTrace() {
      return false;
    }
  }
}
