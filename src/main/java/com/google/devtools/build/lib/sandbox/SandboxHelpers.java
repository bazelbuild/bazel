// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildtool.BuildRequest;

/** Helper methods that are shared by the different sandboxing strategies in this package. */
final class SandboxHelpers {

  static boolean shouldAllowNetwork(BuildRequest buildRequest, Spawn spawn) {
    // If we don't run tests, allow network access.
    if (!buildRequest.shouldRunTests()) {
      return true;
    }

    // If the Spawn specifically requests network access, allow it.
    if (spawn.getExecutionInfo().containsKey("requires-network")) {
      return true;
    }

    // Allow network access, when --java_debug is specified, otherwise we can't connect to the
    // remote debug server of the test.
    if (buildRequest
        .getOptions(BuildConfiguration.Options.class)
        .testArguments
        .contains("--wrapper_script_flag=--debug")) {
      return true;
    }

    return false;
  }
}
