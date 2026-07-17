// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.DetailedExitCode;

/**
 * Thrown when an output provided by a top-level target could not be staged.
 *
 * <p>May occur when:
 *
 * <ul>
 *   <li>An output is lost and action rewinding was ineffective.
 *   <li>There was an error placing the output at its final location.
 * </ul>
 *
 * <p>It is the responsibility of the caller to report this error.
 */
public final class TopLevelOutputException extends BuildFailedException {
  public TopLevelOutputException(String message, DetailedExitCode detailedExitCode) {
    super(message, /* catastrophic= */ false, /* errorAlreadyShown= */ true, detailedExitCode);
  }
}
