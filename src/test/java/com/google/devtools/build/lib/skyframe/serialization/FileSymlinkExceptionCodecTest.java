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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.io.FileSymlinkCycleException;
import com.google.devtools.build.lib.io.FileSymlinkException;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileSymlinkException} serialization. */
@RunWith(JUnit4.class)
public class FileSymlinkExceptionCodecTest {

  @Test
  public void smoke() throws Exception {
    Root root = Root.absoluteRoot(FsUtils.TEST_FILESYSTEM);
    SerializationTester serializationTester =
        new SerializationTester(
                new FileSymlinkInfiniteExpansionException(
                    ImmutableList.of(RootedPath.toRootedPath(root, PathFragment.create("/dir"))),
                    ImmutableList.of(
                        RootedPath.toRootedPath(root, PathFragment.create("/dir/chain")))),
                new FileSymlinkCycleException(
                    ImmutableList.of(RootedPath.toRootedPath(root, PathFragment.create("/dir"))),
                    ImmutableList.of(
                        RootedPath.toRootedPath(root, PathFragment.create("/dir/cycle")))))
            .makeMemoizing()
            .setVerificationFunction(verifyDeserialization);
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }

  private static final SerializationTester.VerificationFunction<FileSymlinkException>
      verifyDeserialization =
          (deserialized, subject) -> {
            assertThat(deserialized).hasMessageThat().isEqualTo(subject.getMessage());
            if (deserialized instanceof FileSymlinkInfiniteExpansionException fsDeserialized) {
              FileSymlinkInfiniteExpansionException fsSubject =
                  (FileSymlinkInfiniteExpansionException) subject;
              assertThat(fsDeserialized.getPathToChain()).isEqualTo(fsSubject.getPathToChain());
              assertThat(fsDeserialized.getChain()).isEqualTo(fsSubject.getChain());
            } else if (deserialized instanceof FileSymlinkCycleException fsDeserialized) {
              FileSymlinkCycleException fsSubject = (FileSymlinkCycleException) subject;
              assertThat(fsDeserialized.getPathToCycle()).isEqualTo(fsSubject.getPathToCycle());
              assertThat(fsDeserialized.getCycle()).isEqualTo(fsSubject.getCycle());
            } else {
              throw new AssertionError("unexpected subclass of FileSymlinkException");
            }
          };
}
