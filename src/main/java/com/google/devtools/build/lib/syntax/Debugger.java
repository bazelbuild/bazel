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


/**
 * A simple interface for the Starlark interpreter to notify a debugger of events during execution.
 */
public interface Debugger {

  /** Notify the debugger that execution is at the point immediately before {@code loc}. */
  void before(StarlarkThread thread, Location loc);

  /** Notify the debugger that it will no longer receive events from the interpreter. */
  void close();
}
