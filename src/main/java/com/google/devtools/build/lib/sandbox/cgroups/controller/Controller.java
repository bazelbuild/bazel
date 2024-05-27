// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox.cgroups.controller;

import java.io.IOException;
import java.nio.file.Path;

/** Interface of a cgroup controller. */
public interface Controller {
  boolean isLegacy();

  default boolean exists() {
    return getPath().toFile().isDirectory();
  }

  Path getPath();

  /** Specialized interface for a cgroup memory controller. */
  interface Memory extends Controller {
    Memory child(String name) throws IOException;

    void setMaxBytes(long bytes) throws IOException;

    long getMaxBytes() throws IOException;

    long getUsageInBytes() throws IOException;
  }

  /** Specialized interface for a cgroup cpu controller. */
  interface Cpu extends Controller {
    Cpu child(String name) throws IOException;

    void setCpus(double cpus) throws IOException;

    long getCpus() throws IOException;
  }
}
