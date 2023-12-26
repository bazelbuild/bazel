// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/** An exception that indicates that a glob pattern is syntactically invalid. */
@ThreadSafe
public final class InvalidGlobPatternException extends Exception {
  private final String pattern;

  InvalidGlobPatternException(String pattern, String error) {
    super(error);
    this.pattern = pattern;
  }

  @Override
  public String getMessage() {
    return String.format("invalid glob pattern '%s': %s", pattern, super.getMessage());
  }
}
