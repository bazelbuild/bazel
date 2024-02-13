// Copyright 2023 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.exec.ExpandedSpawnLogContext.Encoding;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Options;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import org.junit.runner.RunWith;

/** Tests for {@link ExpandedSpawnLogContext}. */
@RunWith(TestParameterInjector.class)
public final class ExpandedSpawnLogContextTest extends SpawnLogContextTestBase {
  private final Path logPath = fs.getPath("/log");
  private final Path tempPath = fs.getPath("/temp");

  @Override
  protected SpawnLogContext createSpawnLogContext(ImmutableMap<String, String> platformProperties)
      throws IOException, InterruptedException {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultExecProperties = platformProperties.entrySet().asList();

    return new ExpandedSpawnLogContext(
        logPath,
        tempPath,
        Encoding.BINARY,
        /* sorted= */ false,
        execRoot.asFragment(),
        remoteOptions,
        DigestHashFunction.SHA256,
        SyscallCache.NO_CACHE);
  }

  @Override
  protected void closeAndAssertLog(SpawnLogContext context, SpawnExec... expected)
      throws IOException, InterruptedException {
    context.close();

    ArrayList<SpawnExec> actual = new ArrayList<>();
    try (InputStream in = logPath.getInputStream()) {
      SpawnExec ex;
      while ((ex = SpawnExec.parseDelimitedFrom(in)) != null) {
        actual.add(ex);
      }
    }

    assertThat(actual).containsExactlyElementsIn(expected).inOrder();
  }
}
