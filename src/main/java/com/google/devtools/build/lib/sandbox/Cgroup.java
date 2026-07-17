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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableSet;
import java.io.IOException;
import java.nio.file.Path;

/** Interface for a cgroup (v1 or v2) that encapsulates all controllers. */
public interface Cgroup {
  int getMemoryUsageInKb();

  boolean exists();

  void addProcess(long pid) throws IOException;

  ImmutableSet<Path> paths();

  void destroy();
}
