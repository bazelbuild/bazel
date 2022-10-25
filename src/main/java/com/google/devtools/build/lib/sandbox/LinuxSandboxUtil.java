// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.Path;

/** Utility functions for the {@code linux-sandbox} embedded tool. */
public final class LinuxSandboxUtil {
  private static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();

  /** Returns whether using the {@code linux-sandbox} is supported in the command environment. */
  public static boolean isSupported(CommandEnvironment cmdEnv) {
    // We can only use the linux-sandbox if the linux-sandbox exists in the embedded tools.
    // This might not always be the case, e.g. while bootstrapping.
    return getLinuxSandbox(cmdEnv) != null;
  }

  /** Returns the path of the {@code linux-sandbox} binary, or null if it doesn't exist. */
  public static Path getLinuxSandbox(CommandEnvironment cmdEnv) {
    return cmdEnv.getBlazeWorkspace().getBinTools().getEmbeddedPath(LINUX_SANDBOX);
  }
}
