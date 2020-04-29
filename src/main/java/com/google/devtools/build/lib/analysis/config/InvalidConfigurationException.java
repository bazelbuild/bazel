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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import javax.annotation.Nullable;

/**
 * Thrown if the configuration options lead to an invalid configuration, or if any of the
 * configuration labels cannot be loaded.
 */
public class InvalidConfigurationException extends Exception {

  @Nullable private final Code detailedCode;

  public InvalidConfigurationException(String message) {
    super(message);
    this.detailedCode = null;
  }

  public InvalidConfigurationException(String message, Throwable cause) {
    super(message, cause);
    this.detailedCode = null;
  }

  public InvalidConfigurationException(Throwable cause) {
    super(cause.getMessage(), cause);
    this.detailedCode = null;
  }

  public InvalidConfigurationException(Code detailedCode, Exception cause) {
    super(cause.getMessage(), cause);
    this.detailedCode = detailedCode;
  }

  @Nullable
  public Code getDetailedCode() {
    return detailedCode;
  }
}
