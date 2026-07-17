// Copyright 2022 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import javax.annotation.Nullable;

/**
 * Wrapper interface on a value in the predeclared lexical block, that controls its accessibility to
 * Starlark based on the value of a semantic flag and/or the Module's client data.
 *
 * <p>For example, this could control whether symbol "Foo" exists in the Starlark global frame: such
 * a symbol might only be accessible if --experimental_foo is set to true. In order to create this
 * control, an instance of this class should be added to the global frame under name "Foo". This
 * guard will throw a descriptive {@link EvalException} when "Foo" would be accessed without the
 * proper flag.
 */
public interface GuardedValue {

  /**
   * Returns an error describing an attempt to access this guard's protected object when it should
   * be inaccessible under the (contextually implied) semantics and client data.
   */
  String getErrorFromAttemptingAccess(String name);

  /**
   * Returns this guard's underlying object. This should be called when appropriate validation has
   * occurred to ensure that the object is accessible with the (implied) semantics.
   */
  Object getObject();

  /**
   * Returns true if this guard's underlying object is accessible under the given semantics and
   * client data.
   */
  boolean isObjectAccessibleUsingSemantics(
      StarlarkSemantics semantics, @Nullable Object clientData);
}
