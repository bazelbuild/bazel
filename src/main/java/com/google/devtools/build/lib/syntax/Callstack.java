// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import java.util.ArrayList;
import java.util.List;

/**
 * Holds the Skylark callstack in thread-local storage. Contains all Expressions and BaseFunctions
 * currently being evaluated.
 *
 * <p>This is needed for memory tracking, since the evaluator is not available in the context of
 * instrumentation. It should not be used by normal Skylark interpreter logic.
 */
public class Callstack {
  // This field should be read, but not written, directly in order to avoid the overhead of a
  // method call.
  static boolean enabled;
  private static final ThreadLocal<List<Object>> callstack =
      ThreadLocal.withInitial(ArrayList::new);

  public static void setEnabled(boolean enabled) {
    Callstack.enabled = enabled;
  }

  public static void push(ASTNode node) {
    callstack.get().add(node);
  }

  public static void push(BaseFunction function) {
    callstack.get().add(function);
  }

  public static void pop() {
    List<Object> threadStack = callstack.get();
    threadStack.remove(threadStack.size() - 1);
  }

  public static List<Object> get() {
    Preconditions.checkState(enabled, "Must call Callstack#setEnabled before getting");
    return callstack.get();
  }

  @VisibleForTesting
  public static void resetStateForTest() {
    enabled = false;
    callstack.get().clear();
  }
}
