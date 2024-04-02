// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

/** A mutable action graph. Implementations of this interface must be thread-safe. */
public interface MutableActionGraph extends ActionGraph {

  /**
   * Attempts to register the action. If any of the action's outputs already has a generating
   * action, and the two actions are not compatible, then an {@link ActionConflictException} is
   * thrown. The internal data structure may be partially modified when that happens; it is not
   * guaranteed that all potential conflicts are detected, but at least one of them is.
   *
   * <p>For example, take three actions A, B, and C, where A creates outputs a and b, B creates just
   * b, and C creates c and b. There are two potential conflicts in this case, between A and B, and
   * between B and C. Depending on the ordering of calls to this method and the ordering of outputs
   * in the action output lists, either one or two conflicts are detected: if B is registered first,
   * then both conflicts are detected; if either A or C is registered first, then only one conflict
   * is detected.
   */
  void registerAction(ActionAnalysisMetadata action)
      throws ActionConflictException, InterruptedException;

  /** Returns the size of the action graph. */
  int getSize();
}
