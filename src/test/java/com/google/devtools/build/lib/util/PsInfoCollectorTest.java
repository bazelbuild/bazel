// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for the PsInfoCollector. */
@RunWith(JUnit4.class)
public final class PsInfoCollectorTest {

  @Test
  public void testCollectStats_ignoreSpaces() throws Exception {
    String psOutput = "    PID  \t  PPID \t  RSS\n   2 1 3216 \t\n  \t 3 1 \t 4096 \t";
    InputStream psStream = new ByteArrayInputStream(psOutput.getBytes(UTF_8));
    Process process = mock(Process.class);

    when(process.getInputStream()).thenReturn(psStream);

    ImmutableMap<Long, PsInfoCollector.PsInfo> pidToPsInfo =
        PsInfoCollector.collectDataFromPsProcess(process);

    ImmutableMap<Long, PsInfoCollector.PsInfo> expectedPidToPsInfo =
        ImmutableMap.of(
            2L,
            PsInfoCollector.PsInfo.create(2, 1, 3216),
            3L,
            PsInfoCollector.PsInfo.create(3, 1, 4096));
    assertThat(pidToPsInfo).isEqualTo(expectedPidToPsInfo);
  }
}
