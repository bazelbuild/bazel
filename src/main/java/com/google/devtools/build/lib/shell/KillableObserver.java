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

package com.google.devtools.build.lib.shell;

/**
 * Implementations of this interface observe, and potentially kill, a {@link Killable} object.
 */
interface KillableObserver {
  /**
   * Begin observing the given {@link Killable}. This method must return promptly; until it returns,
   * {@link Command#execute()} cannot complete. Implementations may wish to start a new
   * {@link Thread} here to handle kill logic, and to interrupt or otherwise ask the thread to stop
   * in the {@link #stopObserving(Killable)} method. See
   * <a href="http://builder.com.com/5100-6370-5144546.html">Interrupting Java threads</a> for notes
   * on how to implement this correctly.
   *
   * <p>Implementations may or may not be able to observe more than one {@link Killable} at a time;
   * see javadoc for details.
   *
   * @param killable killable to observer
   */
  void startObserving(Killable killable);

  /** Stop observing the given {@link Killable}, since it is no longer active. */
  void stopObserving(Killable killable);
}
