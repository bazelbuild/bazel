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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.remote.util.Utils;
import io.grpc.Status;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for remote utility methods */
@RunWith(JUnit4.class)
public class UtilsTest {

  @Test
  public void testGrpcAwareErrorMessages() {
    IOException ioError = new IOException("io error");
    IOException wrappedGrpcError =
        new IOException(
            "wrapped error", Status.ABORTED.withDescription("grpc error").asRuntimeException());

    assertThat(Utils.grpcAwareErrorMessage(ioError)).isEqualTo("io error");
    assertThat(Utils.grpcAwareErrorMessage(wrappedGrpcError)).isEqualTo("ABORTED: grpc error");
  }
}
