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
package com.google.devtools.build.lib.skyframe.config;

import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Inner wrapper exception to work around the fact that {@link SkyFunctionException} cannot carry a
 * message of its own.
 */
final class PlatformMappingParsingException extends Exception {

  PlatformMappingParsingException(String message, Throwable cause) {
    super(message, cause);
  }

  public PlatformMappingParsingException(OptionsParsingException cause) {
    super(cause);
  }
}
