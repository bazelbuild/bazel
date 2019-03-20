// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteUtils. */
@RunWith(JUnit4.class)
public class RemoteUtilsTest {

  @Test
  public void getInlineOutputFile() {
    ActionInput outputFile = ActionInputHelper.fromPath("/foo/bar");
    ImmutableMap<String, String> executionInfo =
        ImmutableMap.of(
            ExecutionRequirements.REMOTE_EXECUTION_INLINE_OUTPUTS, outputFile.getExecPathString());
    Spawn s = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        executionInfo,
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.of(outputFile),
        ResourceSet.ZERO);

    assertThat(RemoteUtils.getInlineOutputFile(s)).isEqualTo(outputFile);
  }
}
