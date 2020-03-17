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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileValue}. */
@RunWith(JUnit4.class)
public class FileValueTest {
  @Test
  public void testCodec() throws Exception {
    SerializationTester serializationTester =
        new SerializationTester(
            // Assume we have adequate coverage for FileStateValue serialization.
            new FileValue.RegularFileValue(
                FsUtils.TEST_ROOTED_PATH, FileStateValue.NONEXISTENT_FILE_STATE_NODE),
            new FileValue.DifferentRealPathFileValueWithStoredChain(
                FsUtils.TEST_ROOTED_PATH,
                FileStateValue.DIRECTORY_FILE_STATE_NODE,
                ImmutableList.of(FsUtils.TEST_ROOTED_PATH)),
            new FileValue.DifferentRealPathFileValueWithoutStoredChain(
                FsUtils.TEST_ROOTED_PATH, FileStateValue.DIRECTORY_FILE_STATE_NODE),
            new FileValue.SymlinkFileValueWithStoredChain(
                FsUtils.TEST_ROOTED_PATH,
                new FileStateValue.RegularFileStateValue(
                    /*size=*/ 100, /*digest=*/ new byte[] {1, 2, 3, 4, 5}, /*contentsProxy=*/ null),
                ImmutableList.of(FsUtils.TEST_ROOTED_PATH),
                PathFragment.create("somewhere/else")),
            new FileValue.SymlinkFileValueWithoutStoredChain(
                FsUtils.TEST_ROOTED_PATH,
                new FileStateValue.RegularFileStateValue(
                    /*size=*/ 100, /*digest=*/ new byte[] {1, 2, 3, 4, 5}, /*contentsProxy=*/ null),
                PathFragment.create("somewhere/else")));
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }
}
