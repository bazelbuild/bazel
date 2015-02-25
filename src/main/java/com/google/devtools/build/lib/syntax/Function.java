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

import java.util.List;
import java.util.Map;

/**
 * Function values in the BUILD language.
 *
 * <p>Each implementation of this interface defines a function in the BUILD language.
 *
 */
public interface Function {

  /**
   * Implements the behavior of a call to a function with positional arguments
   * "args" and keyword arguments "kwargs". The "ast" argument is provided to
   * allow construction of EvalExceptions containing source information.
   */
  Object call(List<Object> args,
              Map<String, Object> kwargs,
              FuncallExpression ast,
              Environment env)
      throws EvalException, InterruptedException;

  /**
   * Returns the name of the function.
   */
  String getName();

  // TODO(bazel-team): implement this for MethodLibrary functions as well.
  /**
   * Returns the type of the object on which this function is defined or null
   * if this is a global function.
   */
  Class<?> getObjectType();
}
