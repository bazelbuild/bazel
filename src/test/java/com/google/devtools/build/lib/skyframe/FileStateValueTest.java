// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileStateValue}. */
@RunWith(JUnit4.class)
public class FileStateValueTest {
  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            new FileStateValue.RegularFileStateValue(
                /*size=*/ 1, /*digest=*/ new byte[] {1, 2, 3}, /*contentsProxy=*/ null),
            new FileStateValue.RegularFileStateValue(
                /*size=*/ 1, /*digest=*/ new byte[0], /*contentsProxy=*/ null),
            new FileStateValue.RegularFileStateValue(
                /*size=*/ 1,
                /*digest=*/ null,
                new FileContentsProxy(/* ctime= */ 2, /* nodeId= */ 42)),
            new FileStateValue.SpecialFileStateValue(
                new FileContentsProxy(/* ctime= */ 4, /* nodeId= */ 84)),
            FileStateValue.DIRECTORY_FILE_STATE_NODE,
            new FileStateValue.SymlinkFileStateValue(PathFragment.create("somewhere/elses")),
            FileStateValue.NONEXISTENT_FILE_STATE_NODE)
        .runTests();
  }

}
