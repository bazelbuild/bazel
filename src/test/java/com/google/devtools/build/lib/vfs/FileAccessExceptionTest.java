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

package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link FileAccessException}. */
@RunWith(JUnit4.class)
public class FileAccessExceptionTest {
  @Test
  public void testCodec() throws Exception {
    new SerializationTester(new FileAccessException("message"))
        .<FileAccessException>setVerificationFunction(
            (original, deserialized) ->
                assertThat(original).hasMessageThat().isEqualTo(deserialized.getMessage()))
        .runTests();
  }
}
