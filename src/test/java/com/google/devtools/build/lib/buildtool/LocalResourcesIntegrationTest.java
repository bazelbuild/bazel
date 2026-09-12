// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for configuring local resources through the host platform. */
@RunWith(JUnit4.class)
public final class LocalResourcesIntegrationTest extends BuildIntegrationTestCase {

  private void writeBuildFiles() throws Exception {
    write(
        "platform/BUILD",
        """
        platform(
            name = "host",
            local_resources = {
                "cpu": "2",
                "gpu-2": "1",
                "gpu-memory": "16",
            },
        )
        """);
    write(
        "BUILD",
        """
        genrule(
            name = "out",
            outs = ["out.txt"],
            cmd = "echo out > $@",
        )
        """);
    addOptions("--host_platform=//platform:host");
  }

  @Test
  public void hostPlatformSetsLocalResources() throws Exception {
    writeBuildFiles();

    buildTarget("//:out");

    ResourceSet resources =
        getCommandEnvironment().getLocalResourceManager().getAvailableResources();
    assertThat(resources).isNotNull();
    assertThat(resources.get("cpu")).isEqualTo(2.0);
    assertThat(resources.get("gpu-2")).isEqualTo(1.0);
    assertThat(resources.get("gpu-memory")).isEqualTo(16.0);
  }

  @Test
  public void hostPlatformOverridesCommandLineResources_withoutMergedAnalysisExecution()
      throws Exception {
    writeBuildFiles();
    addOptions(
        "--noexperimental_merged_skyframe_analysis_execution",
        "--local_resources=cpu=3",
        "--local_resources=gpu-2=2",
        "--local_resources=extra=7");

    buildTarget("//:out");

    ResourceSet resources =
        getCommandEnvironment().getLocalResourceManager().getAvailableResources();
    assertThat(resources).isNotNull();
    assertThat(resources.get("cpu")).isEqualTo(2.0);
    assertThat(resources.get("gpu-2")).isEqualTo(1.0);
    assertThat(resources.get("gpu-memory")).isEqualTo(16.0);
    assertThat(resources.get("extra")).isEqualTo(7.0);
  }
}
