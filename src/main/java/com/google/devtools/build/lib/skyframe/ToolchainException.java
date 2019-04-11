// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import javax.annotation.Nullable;

/** Base class for exceptions that happen during toolchain resolution. */
public class ToolchainException extends Exception {

  public ToolchainException(String message) {
    super(message);
  }

  public ToolchainException(Throwable cause) {
    super(cause);
  }

  public ToolchainException(String message, Throwable cause) {
    super(message, cause);
  }

  @Nullable
  public ConfiguredValueCreationException asConfiguredValueCreationException() {
    for (Throwable cause = getCause();
        cause != null && cause != cause.getCause();
        cause = cause.getCause()) {
      if (cause instanceof ConfiguredValueCreationException) {
        return (ConfiguredValueCreationException) cause;
      }
    }
    return null;
  }
}
