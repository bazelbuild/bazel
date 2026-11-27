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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildOrTestLogs}. */
@RunWith(JUnit4.class)
public class BuildOrTestLogsTest {

  @Test
  public void isBuildOrTestLog_testLogsAndActionOutErr() {
    assertThat(isBuildOrTestLog("bazel-out/k8-fastbuild/testlogs/pkg/a_test/test.log")).isTrue();
    assertThat(isBuildOrTestLog("bazel-out/k8-fastbuild/testlogs/pkg/a_test/test.xml")).isTrue();
    assertThat(isBuildOrTestLog("bazel-out/_tmp/actions/stdout-12")).isTrue();
    assertThat(isBuildOrTestLog("bazel-out/_tmp/actions/stderr-12")).isTrue();
  }

  @Test
  public void isBuildOrTestLog_otherFiles() {
    // A source file, and an output that is neither kind of log.
    assertThat(isBuildOrTestLog("pkg/a.cc")).isFalse();
    assertThat(isBuildOrTestLog("bazel-out/k8-fastbuild/bin/pkg/liba.a")).isFalse();
    // The parts of a log's path are there, but not where a log has them.
    assertThat(isBuildOrTestLog("testlogs/a.log")).isFalse();
    assertThat(isBuildOrTestLog("bazel-out/testlogs/a.log")).isFalse();
    assertThat(isBuildOrTestLog("bazel-out/k8-fastbuild/bin/testlogs/a.log")).isFalse();
    assertThat(isBuildOrTestLog("pkg/bazel-out/k8-fastbuild/testlogs/a.log")).isFalse();
    assertThat(isBuildOrTestLog("pkg/bazel-out/_tmp/actions/stdout-12")).isFalse();
    assertThat(isBuildOrTestLog("bazel-out/_tmp/actions/stdin-12")).isFalse();
    assertThat(isBuildOrTestLog("bazel-out/_tmp/actions/std")).isFalse();
  }

  private static boolean isBuildOrTestLog(String execPath) {
    return BuildOrTestLogs.isBuildOrTestLog(PathFragment.create(execPath));
  }
}
