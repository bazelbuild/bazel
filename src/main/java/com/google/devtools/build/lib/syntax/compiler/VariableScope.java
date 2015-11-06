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

import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.Variable.SkylarkParameter;
import com.google.devtools.build.lib.syntax.compiler.Variable.SkylarkVariable;

import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender.Simple;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.NullConstant;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Tracks variable scopes and and their assignment to indices of the methods local variable array.
 */
public class VariableScope {

  /**
   * The scope of a whole function.
   *
   * <p>This manages the variable index allocation for the whole function.
   * We do not try to re-use variable slots in sub-scopes and once variables are not live anymore.
   * Instead we rely on the register allocation of the JVM JIT compiler to remove any unnecessary
   * ones we add.
   */
  private static final class FunctionVariableScope extends VariableScope {
    private final StackManipulation loadEnvironment;
    private final int parameterCount;
    private int next;

    private FunctionVariableScope(List<String> parameterNames) {
      super(null, parameterNames);
      this.parameterCount = parameterNames.size();
      InternalVariable environment =
          freshVariable(new TypeDescription.ForLoadedType(Environment.class));
      loadEnvironment = MethodVariableAccess.REFERENCE.loadOffset(environment.index);
    }

    @Override
    int next() {
      return next++;
    }

    @Override
    public StackManipulation loadEnvironment() {
      return loadEnvironment;
    }

    @Override
    public ByteCodeAppender createLocalVariablesUndefined() {
      List<ByteCodeAppender> code = new ArrayList<>();
      int skipped = 0;
      Simple nullConstant = new ByteCodeAppender.Simple(NullConstant.INSTANCE);
      for (Variable variable : variables.values()) {
        if (skipped >= parameterCount) {
          code.add(nullConstant);
          code.add(variable.store());
        } else {
          skipped++;
        }
      }
      return ByteCodeUtils.compoundAppender(code);
    }
  }

  /**
   * Initialize a new VariableScope for a function.
   *
   * <p>Will associate the names in order with the local variable indices beginning at 0.
   *  Additionally adds an unnamed local variable parameter at the end for the
   *  {@link com.google.devtools.build.lib.syntax.Environment}. This is needed for
   *  compiling {@link com.google.devtools.build.lib.syntax.UserDefinedFunction}s.
   */
  public static VariableScope function(List<String> parameterNames) {
    return new FunctionVariableScope(parameterNames);
  }

  /** only null for the topmost FunctionVariableScope */
  @Nullable private final VariableScope parent;

  /** default for access by subclass */
  final Map<String, SkylarkVariable> variables;

  private VariableScope(VariableScope parent) {
    this.parent = parent;
    variables = new LinkedHashMap<>();
  }

  private VariableScope(VariableScope parent, List<String> parameterNames) {
    this(parent);
    for (String variable : parameterNames) {
      variables.put(variable, new SkylarkParameter(variable, next()));
    }
  }

  /** delegate next variable index allocation to topmost scope */
  int next() {
    return parent.next();
  }

  // TODO(klaasb) javadoc
  public SkylarkVariable getVariable(Identifier identifier) {
    String name = identifier.getName();
    SkylarkVariable variable = variables.get(name);
    if (variable == null) {
      variable = new SkylarkVariable(name, next());
      variables.put(name, variable);
    }
    return variable;
  }

  /**
   * @return a {@link StackManipulation} that loads the {@link Environment} parameter of the
   * function.
   */
  public StackManipulation loadEnvironment() {
    return parent.loadEnvironment();
  }

  /**
   * @return a fresh anonymous variable which will never collide with user-defined ones
   */
  public InternalVariable freshVariable(TypeDescription type) {
    return new InternalVariable(type, next());
  }

  /**
   * @return a fresh anonymous variable which will never collide with user-defined ones
   */
  public InternalVariable freshVariable(Class<?> type) {
    return freshVariable(new TypeDescription.ForLoadedType(type));
  }

  /**
   * Create a sub scope in which variables can shadow variables from super scopes like this one.
   *
   * <p>Sub scopes don't ensure that variables are initialized with null for "undefined".
   */
  public VariableScope createSubScope() {
    return new VariableScope(this);
  }

  /**
   * Create code that initializes all variables corresponding to Skylark variables to null.
   *
   * <p>This is needed to make sure a byte code variable exists along all code paths and that we
   * can check at runtime whether it wasn't defined along the path actually taken.
   */
  public ByteCodeAppender createLocalVariablesUndefined() {
    return parent.createLocalVariablesUndefined();
  }
}
