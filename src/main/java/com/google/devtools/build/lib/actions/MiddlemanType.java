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

/** Deprecated, slated to be removed. */
public enum MiddlemanType {

  /** A normal action. */
  NORMAL,

  /**
   * A runfiles middleman, which is validated by the dependency checker, but is not expanded in
   * blaze. Instead, the runfiles manifest is sent to remote execution client, which performs the
   * expansion.
   */
  RUNFILES_MIDDLEMAN;

  public boolean isMiddleman() {
    // This value is always false, which means that in theory, the MiddlemanType enum is not useful
    // anymore. It's kept here to facilitate an easy rollback for the change that made the enum
    // unnecessary should trouble arise.
    return false;
  }
}
