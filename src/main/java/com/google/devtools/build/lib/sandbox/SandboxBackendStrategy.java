// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;

/** Strategy backed by an external sandbox-backend binary that speaks Bazel's sandbox protocol. */
public final class SandboxBackendStrategy extends AbstractSpawnStrategy {
  private final String name;

  SandboxBackendStrategy(String name, SpawnRunner spawnRunner, ExecutionOptions executionOptions) {
    super(spawnRunner, executionOptions);
    this.name = name;
  }

  @Override
  public String toString() {
    return name;
  }
}
