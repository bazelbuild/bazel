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

package com.google.devtools.build.lib.sandbox.cgroups.controller.v1;

import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** V1 memory controller. */
public class LegacyMemory extends LegacyController implements Controller.Memory {
  private final Path path;

  @Override
  public Path getPath() {
    return path;
  }

  public LegacyMemory(Path path) {
    this.path = path;
  }

  @Override
  public Memory child(String name) throws IOException {
    return new LegacyMemory(getChild(name));
  }

  @Override
  public void setMaxBytes(long bytes) throws IOException {
    Files.writeString(path.resolve("memory.limit_in_bytes"), Long.toString(bytes));
  }

  @Override
  public long getMaxBytes() throws IOException {
    return Long.parseLong(Files.readString(path.resolve("memory.limit_in_bytes")).trim());
  }

  @Override
  public long getUsageInBytes() throws IOException {
    return Long.parseLong(Files.readString(path.resolve("memory.usage_in_bytes")).trim());
  }
}
