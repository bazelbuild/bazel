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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.InvalidPackageNameException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NoSuchPackageException} serialization. */
@RunWith(JUnit4.class)
public class NoSuchPackageExceptionCodecTest {
  @Test
  public void smoke() throws Exception {
    new SerializationTester(
            new BuildFileNotFoundException(
                PackageIdentifier.create("repo", PathFragment.create("foo")), "msg"),
            new BuildFileNotFoundException(
                PackageIdentifier.create("repo", PathFragment.create("foo")),
                "msg",
                new IOException("bar")),
            new BuildFileContainsErrorsException(
                PackageIdentifier.create("repo", PathFragment.create("foo")), "msg"),
            new BuildFileContainsErrorsException(
                PackageIdentifier.create("repo", PathFragment.create("foo")),
                "msg",
                new IOException("bar")),
            new InvalidPackageNameException(
                PackageIdentifier.create("repo", PathFragment.create("foo")), "msg"),
            new NoSuchPackageException(
                PackageIdentifier.create("repo", PathFragment.create("foo")), "msg"),
            new NoSuchPackageException(
                PackageIdentifier.create("repo", PathFragment.create("foo")),
                "msg",
                new IOException("bar")))
        .setVerificationFunction(verifyDeserialization)
        .makeMemoizing()
        .runTests();
  }

  private static final SerializationTester.VerificationFunction<NoSuchPackageException>
      verifyDeserialization =
          (deserialized, subject) -> {
            assertThat(deserialized).hasMessageThat().isEqualTo(subject.getMessage());
            assertThat(deserialized.getPackageId()).isEqualTo(subject.getPackageId());

            if (subject.getCause() == null) {
              assertThat(deserialized).hasCauseThat().isNull();
            } else {
              assertThat(deserialized)
                  .hasCauseThat()
                  .hasMessageThat()
                  .isEqualTo(subject.getCause().getMessage());
            }
          };
}
