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

package com.google.devtools.build.lib.actions;

/**
 * Thrown by command line classes during expansion.
 *
 * <p>This exception should be thrown deterministically, i.e. the command line should fail to expand
 * in exactly the same way every time at attempt is made. An example would be an illegal format
 * string, or a failure in Skylark evaluation of a compact command line.
 */
public final class CommandLineExpansionException extends Exception {
  /** @param userVisibleErrorMessage An error string that will be displayed to the user. */
  public CommandLineExpansionException(String userVisibleErrorMessage) {
    super(userVisibleErrorMessage);
  }
}
