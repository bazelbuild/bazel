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

/** V1 cpu controller. */
public class LegacyCpu extends LegacyController implements Controller.Cpu {
  private final Path path;

  public LegacyCpu(Path path) {
    this.path = path;
  }

  @Override
  public Cpu child(String name) throws IOException {
    return new LegacyCpu(getChild(name));
  }

  @Override
  public Path getPath() {
    return path;
  }

  @Override
  public void setCpus(double cpus) throws IOException {
    long period = Long.parseLong(Files.readString(path.resolve("cpu.cfs_period_us")).trim());
    long quota = Math.round(cpus * period);
    Files.writeString(path.resolve("cpu.cfs_quota_us"), Long.toString(quota));
  }

  @Override
  public long getCpus() throws IOException {
    long quota = Long.parseLong(Files.readString(path.resolve("cpu.cfs_quota_us")).trim());
    long period = Long.parseLong(Files.readString(path.resolve("cpu.cfs_period_us")).trim());
    return quota / period;
  }
}
