// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ExecutorLifecycleListener;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.Crash;
import com.google.devtools.build.lib.server.FailureDetails.Crash.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.EphemeralCheckIfOutputConsumed;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test to make sure that context provider initialization failure is handled correctly.
 */
@RunWith(JUnit4.class)
public class ContextProviderInitializationTest extends BuildIntegrationTestCase {

  private static class BadContextProviderModule extends BlazeModule {
    @Override
    public void executorInit(
        CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
      builder.addExecutorLifecycleListener(
          new ExecutorLifecycleListener() {

            @Override
            public void executorCreated() {}

            @Override
            public void executionPhaseStarting(
                ActionGraph actionGraph,
                Supplier<ImmutableSet<Artifact>> topLevelArtifacts,
                @Nullable EphemeralCheckIfOutputConsumed unused)
                throws AbruptExitException {
              throw new AbruptExitException(
                  DetailedExitCode.of(
                      FailureDetail.newBuilder()
                          .setMessage("eek")
                          .setCrash(Crash.newBuilder().setCode(Code.CRASH_UNKNOWN))
                          .build()));
            }

            @Override
            public void executionPhaseEnding() {}
          });
    }
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new BadContextProviderModule());
  }

  @Test
  public void testContextProviderInitializationFailure() {
    assertThrows(AbruptExitException.class, this::buildTarget);
  }
}
