// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import java.util.Map;

/** Info item for the effective current client environment. */
public final class ClientEnv extends InfoItem {
  public ClientEnv() {
    super(
        "client-env",
        "The specifications that need to be added to the project-specific rc file to freeze the"
            + " current client environment",
        true);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    String result = "";
    for (Map.Entry<String, String> entry : env.getWhitelistedActionEnv().entrySet()) {
      // TODO(bazel-team): as the syntax of our rc-files does not support to express new-lines in
      // values, we produce syntax errors if the value of the entry contains a newline character.
      result += "build --action_env=" + entry.getKey() + "=" + entry.getValue() + "\n";
    }
    for (Map.Entry<String, String> entry : env.getWhitelistedTestEnv().entrySet()) {
      // TODO(bazel-team): as the syntax of our rc-files does not support to express new-lines in
      // values, we produce syntax errors if the value of the entry contains a newline character.
      result += "build --test_env=" + entry.getKey() + "=" + entry.getValue() + "\n";
    }
    return print(result);
  }
}
