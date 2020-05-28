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

package com.google.devtools.build.lib.syntax;

/** A Starlark value that supports membership tests, {@code key in object}. */
// TODO(adonovan): merge with StarlarkIndexable: no type supports 'x in y' without y[x],
// and 'x in y' can be defined in terms of y[x], at least as a default implementation.
// (Implementations of 'x in y' may choose to interpret failure of y[x] as false or a failure.)
public interface StarlarkQueryable extends StarlarkValue {

  /** Returns whether the key is in the object. */
  boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException;
}
