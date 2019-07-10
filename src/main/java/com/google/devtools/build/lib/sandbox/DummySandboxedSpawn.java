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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils.MoveResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Implements detour-based sandboxed spawn.
 */
public class DummySandboxedSpawn implements SandboxedSpawn {

  private final Path execRoot;
  private final Map<String, String> environment;
  private final List<String> arguments;

  public DummySandboxedSpawn(
      Path execRoot,
      Map<String, String> environment,
      List<String> arguments) {
    this.execRoot = execRoot;
    this.environment = environment;
    this.arguments = arguments;
  }

  @Override
  public Path getSandboxExecRoot() {
    return execRoot;
  }

  @Override
  public List<String> getArguments() {
    return arguments;
  }

  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  @Override
  public void createFileSystem() throws IOException {
  }

  @Override
  public void copyOutputs(Path execRoot) throws IOException {
  }

  @Override
  public void delete() {
  }
}
