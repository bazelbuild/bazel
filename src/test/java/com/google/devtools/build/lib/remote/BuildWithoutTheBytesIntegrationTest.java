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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.IntegrationTestUtils.startWorker;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for Build without the Bytes. */
@RunWith(JUnit4.class)
public class BuildWithoutTheBytesIntegrationTest extends BuildIntegrationTestCase {
  private WorkerInstance worker;

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new RemoteModule())
        .build();
  }

  @Before
  public void setUp() throws IOException, InterruptedException {
    worker = startWorker();
  }

  @After
  public void tearDown() throws IOException {
    worker.stop();
  }

  @Test
  public void downloadMinimal_outputsAreNotDownloaded() throws Exception {
    write(
        "a/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = [],",
        "  outs = ['foo.txt'],",
        "  cmd = 'echo foo > $@',",
        ")",
        "",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo'],",
        "  outs = ['foobar.txt'],",
        "  cmd = 'cat $(location :foo) > $@ && echo bar > $@',",
        ")");

    addOptions(
        "--remote_executor=grpc://localhost:" + worker.getPort(), "--remote_download_minimal");
    buildTarget("//a:foobar");

    ImmutableList<Artifact> outputs =
        ImmutableList.<Artifact>builder()
            .addAll(getArtifacts("//a:foo"))
            .addAll(getArtifacts("//a:foobar"))
            .build();
    for (Artifact output : outputs) {
      assertThat(output.getPath().exists()).isFalse();
    }
  }
}
