// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import javax.annotation.Nullable;

/** Implements detour-based sandboxed spawn. */
public class WindowsSandboxedSpawn implements SandboxedSpawn {

  private final Path execRoot;
  private final ImmutableMap<String, String> environment;
  private final ImmutableList<String> arguments;
  private final String mnemonic;

  public WindowsSandboxedSpawn(
      Path execRoot,
      ImmutableMap<String, String> environment,
      ImmutableList<String> arguments,
      String mnemonic) {
    this.execRoot = execRoot;
    this.environment = environment;
    this.arguments = arguments;
    this.mnemonic = mnemonic;
  }

  @Override
  public Path getSandboxExecRoot() {
    return execRoot;
  }

  @Override
  public ImmutableList<String> getArguments() {
    return arguments;
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    return environment;
  }

  @Override
  @Nullable
  public Path getSandboxDebugPath() {
    // On Windows, sandbox debugging output is written to stderr rather than a separate file.
    return null;
  }

  @Override
  @Nullable
  public Path getStatisticsPath() {
    // On Windows, Bazel currently does not support per-process statistics.
    return null;
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public void createFileSystem() throws IOException {}

  @Override
  public void copyOutputs(Path execRoot) throws IOException {}

  @Override
  public void delete() {}
}
