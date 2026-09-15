// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.blackbox.framework;

import java.io.IOException;

/** Base interface for all integration test tools setup. */
public interface ToolsSetup {

  /**
   * Copy and create all necessary files under the working directory of the test.
   *
   * @param context {@link BlackBoxTestContext} to use for files manipulation, access the test
   *     environment, and running Bazel/Blaze commands
   * @throws IOException if any I/O happened during initialization
   */
  void setup(BlackBoxTestContext context) throws IOException;
}
