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
 * Implementations encapsulate a running process that can be killed. In particular, here, it is used
 * to wrap up a {@link Process} object and expose it to a {@link KillableObserver}. It is wrapped in
 * this way so that the actual {@link Process} object can't be altered by a
 * {@link KillableObserver}.
 */
interface Killable {

  /**
   * Kill this killable instance.
   */
  void kill();
}
