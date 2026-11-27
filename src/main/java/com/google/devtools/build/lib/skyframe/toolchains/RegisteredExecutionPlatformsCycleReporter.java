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
package com.google.devtools.build.lib.skyframe.toolchains;

import com.google.devtools.build.lib.skyframe.SkyFunctions;

/**
 * {@link AbstractToolchainCycleReporter} implementation that can handle cycles involving registered
 * execution platforms.
 */
public class RegisteredExecutionPlatformsCycleReporter extends AbstractToolchainCycleReporter {

  public RegisteredExecutionPlatformsCycleReporter() {
    super(SkyFunctions.REGISTERED_EXECUTION_PLATFORMS);
  }

  @Override
  protected String formatCycleMessage(String rootTarget) {
    // TODO: jcater - Clean this up: ideally we only need to list the actual platform and the direct
    // dependency of it in the cycle.
    return "Misconfigured execution platforms: "
        + rootTarget
        + " is declared as a platform but has inappropriate dependencies."
        + " Execution platforms should not have dependencies that themselves require"
        + " toolchain resolution.";
  }
}
