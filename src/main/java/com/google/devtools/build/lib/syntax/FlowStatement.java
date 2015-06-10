// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * A class for flow statements (e.g. break and continue)
 */
public final class FlowStatement extends Statement {

  public static final FlowStatement BREAK = new FlowStatement("break", true);
  public static final FlowStatement CONTINUE = new FlowStatement("continue", false);

  private final String name;
  private final FlowException ex;

  /**
   *
   * @param name The label of the statement (either break or continue)
   * @param terminateLoop Determines whether the enclosing loop should be terminated completely
   *        (break)
   */
  protected FlowStatement(String name, boolean terminateLoop) {
    this.name = name;
    this.ex = new FlowException(terminateLoop);
  }

  @Override
  void exec(Environment env) throws EvalException {
    throw ex;
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    if (!env.isInsideLoop()) {
      throw new EvalException(getLocation(), name + " statement must be inside a for loop");
    }
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * An exception that signals changes in the control flow (e.g. break or continue)
   */
  class FlowException extends EvalException {
    private final boolean terminateLoop;

    /**
     *
     * @param terminateLoop Determines whether the enclosing loop should be terminated completely
     *        (break)
     */
    public FlowException(boolean terminateLoop) {
      super(FlowStatement.this.getLocation(), "FlowException with terminateLoop = "
          + terminateLoop);
      this.terminateLoop = terminateLoop;
    }

    /**
     * Returns whether the enclosing loop should be terminated completely (break)
     *
     * @return {@code True} for 'break', {@code false} for 'continue'
     */
    public boolean mustTerminateLoop() {
      return terminateLoop;
    }
  }
}
