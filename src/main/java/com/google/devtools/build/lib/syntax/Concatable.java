// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import com.google.devtools.build.lib.events.Location;
import javax.annotation.Nullable;

/**
 * Skylark values that support '+' operator should implement this interface.
 */
public interface Concatable {

  /**
   * Implements 'plus' operator on ClassObjects.
   */
  interface Concatter {
    Concatable concat(Concatable lval, Concatable rval, Location loc) throws EvalException;
  }

  /* Returns a concatter for this {@link Concatable}.
   * Two {@link Concatable}s can be added together if their concatters are equal.
   */
  @Nullable
  Concatter getConcatter();
}
