// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.vfs.Path;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.concurrent.ThreadSafe;

/** Generates temporary paths under a given directory. */
@ThreadSafe
public class TempPathGenerator {
  private final Path tempDir;
  private final AtomicInteger index = new AtomicInteger();

  public TempPathGenerator(Path tempDir) {
    this.tempDir = tempDir;
  }

  /** Generates a temporary path. */
  public Path generateTempPath() {
    return tempDir.getChild(index.getAndIncrement() + ".tmp");
  }

  public Path getTempDir() {
    return tempDir;
  }
}
