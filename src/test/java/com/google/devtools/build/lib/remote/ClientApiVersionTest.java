// Copyright 2023 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.semver.SemVer;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

// Tests for {@link ApiVersion}.
@RunWith(Parameterized.class)
public class ClientApiVersionTest {
  @Parameters(name = "{0}")
  public static List<Object[]> testCases() {
    return Arrays.asList(
        new Object[][] {
          {
            "noSupportedVersion",
            new ClientApiVersion(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build())),
            ServerCapabilities.newBuilder()
                .setLowApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build())
                .setHighApiVersion(SemVer.newBuilder().setMajor(2).setMinor(2).build())
                .build(),
            ClientApiVersion.ServerSupportedStatus.unsupported(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(2).build())),
            Arrays.asList("not supported", "2.0 to 2.0", "2.1 to 2.2")
          },
          {
            "deprecated",
            new ClientApiVersion(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build())),
            ServerCapabilities.newBuilder()
                .setDeprecatedApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build())
                .setLowApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build())
                .setHighApiVersion(SemVer.newBuilder().setMajor(2).setMinor(2).build())
                .build(),
            ClientApiVersion.ServerSupportedStatus.deprecated(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(2).build())),
            Arrays.asList("2.0 is deprecated", "2.1 to 2.2")
          },
          {
            "clientHigh",
            new ClientApiVersion(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(3).build())),
            ServerCapabilities.newBuilder()
                .setLowApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build())
                .setHighApiVersion(SemVer.newBuilder().setMajor(2).setMinor(4).build())
                .build(),
            ClientApiVersion.ServerSupportedStatus.supported(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(3).build())),
            Arrays.asList()
          },
          {
            "serverHigh",
            new ClientApiVersion(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build()),
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(3).build())),
            ServerCapabilities.newBuilder()
                .setLowApiVersion(SemVer.newBuilder().setMajor(2).setMinor(0).build())
                .setHighApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build())
                .build(),
            ClientApiVersion.ServerSupportedStatus.supported(
                new ApiVersion(SemVer.newBuilder().setMajor(2).setMinor(1).build())),
            Arrays.asList()
          },
        });
  }

  private final ClientApiVersion clientApiVersion;
  private final ServerCapabilities serverCapabilities;
  private final ClientApiVersion.ServerSupportedStatus expectedHighestSupportedVersion;
  private final List<String> expectedMessages;

  public ClientApiVersionTest(
      String name,
      ClientApiVersion clientApiVersion,
      ServerCapabilities serverCapabilities,
      ClientApiVersion.ServerSupportedStatus expectedHighestSupportedVersion,
      List<String> expectedMessages) {
    this.clientApiVersion = clientApiVersion;
    this.serverCapabilities = serverCapabilities;
    this.expectedHighestSupportedVersion = expectedHighestSupportedVersion;
    this.expectedMessages = expectedMessages;
  }

  @Test
  public void testClientApiVersion() {
    var serverSupportedStatus = clientApiVersion.checkServerSupportedVersions(serverCapabilities);
    assertThat(serverSupportedStatus.isSupported())
        .isEqualTo(expectedHighestSupportedVersion.isSupported());
    assertThat(serverSupportedStatus.isUnsupported())
        .isEqualTo(expectedHighestSupportedVersion.isUnsupported());
    assertThat(serverSupportedStatus.isDeprecated())
        .isEqualTo(expectedHighestSupportedVersion.isDeprecated());
    assertThat(serverSupportedStatus.getMessage())
        .isEqualTo(expectedHighestSupportedVersion.getMessage());

    for (var expectedMessage : expectedMessages) {
      assertThat(serverSupportedStatus.getMessage()).contains(expectedMessage);
    }
    if (expectedMessages.isEmpty()) {
      assertThat(serverSupportedStatus.getMessage()).isEmpty();
    }

    var expectedHigh = expectedHighestSupportedVersion.getHighestSupportedVersion();
    if (expectedHigh != null) {
      var high = serverSupportedStatus.getHighestSupportedVersion();

      assertThat(high).isNotNull();
      assertThat(high.compareTo(expectedHigh)).isEqualTo(0);
    }
  }
}
