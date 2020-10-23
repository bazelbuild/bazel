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

/** The action type. */
public enum MiddlemanType {

  /** A normal action. */
  NORMAL,

  /** A normal middleman, which just encapsulates a list of artifacts. */
  AGGREGATING_MIDDLEMAN,

  /**
   * A middleman that denotes a scheduling dependency.
   *
   * <p>If an action has dependencies through scheduling dependency middleman, those dependencies
   * will get built before the action is run and the build will error out if they cannot be built,
   * but the dependencies will not be considered inputs of the action.
   *
   * <p>This is useful in cases when an action <em>might</em> need some inputs, but that is only
   * found out right before it gets executed. The most salient case is C++ compilation where all
   * files that can possibly be included need to be built before the action is executed, but if
   * include scanning is used, only a subset of them will end up as inputs.
   */
  SCHEDULING_DEPENDENCY_MIDDLEMAN,

  /**
   * A runfiles middleman, which is validated by the dependency checker, but is not expanded in
   * blaze. Instead, the runfiles manifest is sent to remote execution client, which performs the
   * expansion.
   */
  RUNFILES_MIDDLEMAN;

  public boolean isMiddleman() {
    return this != NORMAL;
  }
}
