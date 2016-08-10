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

import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.Jump.ReferenceComparison;

import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.Removal;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.TextConstant;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;

/**
 * Superclass for various variable representations during compile time.
 */
public abstract class Variable {

  /**
   * The index in the byte code local variable array of the method.
   *
   * <p>package-protected for access by VariableScope and variable-modifying byte code generators
   */
  final int index;

  private Variable(int index) {
    this.index = index;
  }

  /**
   * Store operands on the stack to this variables array slot in byte code.
   */
  public abstract ByteCodeAppender store();

  /**
   * A variable generated for a named local variable in Skylark.
   *
   * <p>To get correct Python-style semantics in byte code, we need to allow control-flow paths
   * along which variables may be "undefined". This must result in a runtime error.
   */
  public static class SkylarkVariable extends Variable {

    public final String name;
    private final ByteCodeAppender store;

    SkylarkVariable(String name, int index) {
      super(index);
      this.name = name;
      this.store = VariableStore.into(this);
    }

    /**
     * Builds a ByteCodeAppender for loading this variable which makes sure it was defined before
     * this use.
     */
    public ByteCodeAppender load(VariableScope scope, AstAccessors debugAccessors) {
      // the variable may not be defined
      // along the control-flow path taken at runtime, so we need to check for that then
      LabelAdder end = new LabelAdder();
      return new ByteCodeAppender.Simple(
          // we don't generate primitive local variables from Skylark variables currently
          // we'd also need a more elaborate way to check whether they were undefined
          MethodVariableAccess.REFERENCE.loadOffset(index),
          Duplication.SINGLE,
          Jump.ifReferenceOperandToNull(ReferenceComparison.NOT_EQUAL).to(end),
          Removal.SINGLE,
          // not a method parameter or variable defined previously in the method body, must look it
          // up in the environment passed to the call
          scope.loadEnvironment(),
          new TextConstant(name),
          debugAccessors.loadAstNode,
          ByteCodeUtils.invoke(
              SkylarkVariable.class,
              "lookupUnboundVariable",
              Environment.class,
              String.class,
              ASTNode.class),
          end);
    }

    @Override
    public ByteCodeAppender store() {
      return store;
    }

    /**
     * Looks for the variable in the method calls outside environment and fail with debug info
     * if not found.
     */
    public static Object lookupUnboundVariable(Environment global, String variable, ASTNode node)
        throws EvalExceptionWithStackTrace {
      Object value = global.lookup(variable);
      if (value == null) {
        throw new EvalExceptionWithStackTrace(
            new EvalException(
                node.getLocation(),
                "local variable '" + variable + "' referenced before assignment"),
            node);
      }
      return value;
    }
  }

  /**
   * Parameter of a Skylark function.
   *
   * <p>These will always have a value along each intra-procedural control-flow path and thus we
   * can generate simpler code.
   */
  public static final class SkylarkParameter extends SkylarkVariable {

    SkylarkParameter(String name, int index) {
      super(name, index);
    }

    @Override
    public ByteCodeAppender load(VariableScope scope, AstAccessors debugAccessors) {
      return new ByteCodeAppender.Simple(MethodVariableAccess.REFERENCE.loadOffset(index));
    }
  }

  /**
   * A variable generated for internal use and thus the property holds that all uses are dominated
   * by their definition and we can actually type it.
   */
  public static final class InternalVariable extends Variable {

    public final TypeDescription type;
    private final ByteCodeAppender store;

    InternalVariable(TypeDescription type, int index) {
      super(index);
      this.type = type;
      this.store = VariableStore.into(this);
    }

    /**
     * Builds a simple StackManipulation which loads the variable from its index while taking into
     * account the correct type.
     */
    public StackManipulation load() {
      return MethodVariableAccess.forType(type).loadOffset(index);
    }

    @Override
    public ByteCodeAppender store() {
      return store;
    }
  }
}
