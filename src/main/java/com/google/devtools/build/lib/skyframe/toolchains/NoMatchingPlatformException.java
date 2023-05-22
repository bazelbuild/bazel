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
package com.google.devtools.build.lib.skyframe.toolchains;

import com.google.devtools.build.lib.server.FailureDetails;

/** Indicates a missing execution platform. */
public final class NoMatchingPlatformException extends ToolchainException {
  public NoMatchingPlatformException(NoMatchingPlatformData error) {
    super(error.formatError());
  }

  @Override
  protected FailureDetails.Toolchain.Code getDetailedCode() {
    return FailureDetails.Toolchain.Code.NO_MATCHING_EXECUTION_PLATFORM;
  }
}
