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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;

/** Used when instrumentation output is written to a local file. */
final class LocalInstrumentationOutput implements InstrumentationOutput {
  private final Path path;
  private final String name;

  LocalInstrumentationOutput(String name, Path path) {
    this.name = name;
    this.path = path;
  }

  @Override
  public void publish(BuildToolLogCollection buildToolLogCollection) {
    buildToolLogCollection.addLocalFile(name, path);
  }

  @Override
  public OutputStream createOutputStream() throws IOException {
    return path.getOutputStream();
  }

  public OutputStream createOutputStream(boolean append, boolean internal) throws IOException {
    return path.getOutputStream(append, internal);
  }
}
