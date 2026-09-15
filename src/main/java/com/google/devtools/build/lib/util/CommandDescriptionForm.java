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

package com.google.devtools.build.lib.util;

/**
 * Forms in which a command can be described by {@link CommandFailureUtils#describeCommand}.
 */
public enum CommandDescriptionForm {
  /**
   * A form that is usually suitable for identifying the command but not for
   * re-executing it.  The working directory and environment are not shown, and
   * the arguments are truncated to a maximum of a few hundred bytes.
   */
  ABBREVIATED,

  /**
   * A form that is complete and suitable for a user to copy and paste into a shell. On Linux, the
   * command is placed in a subshell so it has no side effects on the user's shell. On Windows, this
   * is not implemented, but the side effects in question are less severe (no "exec").
   */
  COMPLETE,
}
