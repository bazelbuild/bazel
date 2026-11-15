// Copyright 2026 The Bazel Authors. All rights reserved.
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

import net.starlark.java.syntax.Location;

/**
 * A PrintHandler determines how a Starlark thread deals with print statements. It is invoked by the
 * built-in {@code print} function. Its default behavior is to write the message to standard error,
 * preceded by the location of the print statement, {@code thread.getCallerLocation()}.
 */
@FunctionalInterface
public interface PrintHandler {
  void print(StarlarkThreadInfo thread, String msg);

  /** Read-only view of a {@link StarlarkThread} exposed to a {@link PrintHandler}. */
  interface StarlarkThreadInfo {
    /**
     * Returns the location of the program counter in the enclosing call frame. If called from
     * within a built-in function, this is the location of the call expression that called the
     * built-in. It returns BUILTIN if called with fewer than two frames (such as within a test).
     */
    Location getCallerLocation();
  }
}
