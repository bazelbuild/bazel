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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.packages.WorkspaceFileValue.WorkspaceFileKey;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link WorkspaceFileKey_AutoCodec}. */
@RunWith(JUnit4.class)
public class WorkspaceFileKeyCodecTest {
  @Test
  public void testCodec() throws Exception {
    SerializationTester serializationTester =
        new SerializationTester(WorkspaceFileKey.create(FsUtils.TEST_ROOTED_PATH, 42));
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }
}
