// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;

/**
 * A helper class for calling Skylark functions from Java.
 */
public class SkylarkCallbackFunction {

  private final Function callback;
  private final FuncallExpression ast;
  private final SkylarkEnvironment funcallEnv;

  public SkylarkCallbackFunction(Function callback, FuncallExpression ast,
      SkylarkEnvironment funcallEnv) {
    this.callback = callback;
    this.ast = ast;
    this.funcallEnv = funcallEnv;
  }

  public Object call(ClassObject ctx, Object... arguments) throws EvalException {
    try {
      return callback.call(
          ImmutableList.<Object>builder().add(ctx).add(arguments).build(), null, ast, funcallEnv);
    } catch (InterruptedException | ClassCastException
        | IllegalArgumentException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }
}
