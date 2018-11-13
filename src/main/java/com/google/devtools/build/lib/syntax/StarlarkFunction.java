// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A base interface for function-like objects that are callable in Starlark, whether builtin or
 * user-defined.
 */
public interface StarlarkFunction extends SkylarkValue {

  /**
   * Call this function with the given arguments.
   *
   * @param args a list of all positional arguments (as in *starArg)
   * @param kwargs a map for key arguments (as in **kwArgs)
   * @param ast the expression for this function's definition
   * @param env the Environment in which the function is called
   * @return the value resulting from evaluating the function with the given arguments
   * @throws EvalException if there was an error invoking this function
   */
  public Object call(
      List<Object> args,
      @Nullable Map<String, Object> kwargs,
      @Nullable FuncallExpression ast,
      Environment env)
      throws EvalException, InterruptedException;
}
