// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.shell.BadExitStatusException;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Map;

/**
 * Tests for {@code LinuxSandboxedStrategy} that must run locally, because they need to actually
 * run the namespace-sandbox binary.
 */
@TestSpec(localOnly = true, supportedOs = OS.LINUX)
@RunWith(JUnit4.class)
public class LocalLinuxSandboxedStrategyTest extends LinuxSandboxedStrategyTestCase {
  protected Spawn createSpawn(String... arguments) {
    Map<String, String> environment = ImmutableMap.<String, String>of();
    Map<String, String> executionInfo = ImmutableMap.<String, String>of();
    ActionMetadata action = new ActionsTestUtil.NullAction();
    ResourceSet localResources = ResourceSet.ZERO;
    return new BaseSpawn(
        Arrays.asList(arguments), environment, executionInfo, action, localResources);
  }

  protected ActionExecutionContext createContext() {
    Path execRoot = executor.getExecRoot();
    return new ActionExecutionContext(
        executor,
        new SingleBuildFileCache(execRoot.getPathString(), execRoot.getFileSystem()),
        null,
        outErr,
        null);
  }

  @Test
  public void testExecutionSuccess() throws Exception {
    Spawn spawn = createSpawn("/bin/sh", "-c", "echo Hello, world.; touch dummy");
    getLinuxSandboxedStrategy().exec(spawn, createContext());
    assertThat(out()).isEqualTo("Hello, world.\n");
    assertThat(err()).isEmpty();
  }

  @Test
  public void testExecutionFailurePrintsCorrectMessage() throws Exception {
    Spawn spawn = createSpawn("/bin/sh", "-c", "echo ERROR >&2; exit 1");
    try {
      getLinuxSandboxedStrategy().exec(spawn, createContext());
      fail();
    } catch (UserExecException e) {
      assertThat(err()).isEqualTo("ERROR\n");
      assertThat(e.getCause()).isInstanceOf(BadExitStatusException.class);
    }
  }
}
