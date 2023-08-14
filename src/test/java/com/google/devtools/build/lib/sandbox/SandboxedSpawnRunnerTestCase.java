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
package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.shell.JavaSubprocessFactory;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import java.io.IOException;
import org.junit.Before;

/**
 * A base class for tests of SandboxedSpawnRunners.
 *
 * <p>Extending {@link BuildIntegrationTestCase} is necessary because SandboxedSpawnRunners expect
 * {@link CommandEnvironment}s in their constructors, and BuildIntegrationTestCase is currently the
 * only provider of CommandEnvironments we have for tests.
 */
public abstract class SandboxedSpawnRunnerTestCase extends BuildIntegrationTestCase {
  @Override
  protected void setupMockTools() throws IOException {
    // Do nothing.
  }

  /** Enables real execution by default. */
  @Before
  public final void setupEnvironmentForRealExecution() {
    SubprocessBuilder.setDefaultSubprocessFactory(JavaSubprocessFactory.INSTANCE);
  }
}
