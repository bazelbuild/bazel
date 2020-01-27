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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.semver.SemVer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ApiVersion}.
 */
@RunWith(JUnit4.class)
public class ApiVersionTest {

  @Test
  public void testToString() throws Exception {
    assertThat(new ApiVersion(0, 0, 0, "v1test").toString()).isEqualTo("v1test");
    assertThat(new ApiVersion(1, 2, 3, "v1test").toString()).isEqualTo("v1test");
    assertThat(new ApiVersion(2, 0, 0, "").toString()).isEqualTo("2.0");
    assertThat(new ApiVersion(2, 1, 0, "").toString()).isEqualTo("2.1");
    assertThat(new ApiVersion(10, 0, 3, "").toString()).isEqualTo("10.0.3");
  }

  @Test
  public void testCompareTo() throws Exception {
    assertThat(new ApiVersion(0, 0, 0, "v1test").compareTo(new ApiVersion(0, 0, 0, "v1test")))
        .isEqualTo(0);
    assertThat(new ApiVersion(0, 0, 0, "v1test").compareTo(new ApiVersion(0, 1, 0, "")))
        .isLessThan(0);
    assertThat(new ApiVersion(0, 0, 1, "").compareTo(new ApiVersion(1, 0, 0, "v1test")))
        .isGreaterThan(0);
    assertThat(new ApiVersion(1, 0, 0, "").compareTo(new ApiVersion(2, 0, 0, "")))
        .isLessThan(0);
    assertThat(new ApiVersion(2, 0, 0, "").compareTo(new ApiVersion(1, 0, 0, "")))
        .isGreaterThan(0);
    assertThat(new ApiVersion(2, 1, 0, "").compareTo(new ApiVersion(2, 2, 0, "")))
        .isLessThan(0);
    assertThat(new ApiVersion(2, 2, 0, "").compareTo(new ApiVersion(2, 1, 0, "")))
        .isGreaterThan(0);
    assertThat(new ApiVersion(2, 2, 1, "").compareTo(new ApiVersion(2, 2, 2, "")))
        .isLessThan(0);
    assertThat(new ApiVersion(2, 2, 2, "").compareTo(new ApiVersion(2, 1, 1, "")))
        .isGreaterThan(0);
    assertThat(new ApiVersion(2, 2, 2, "").compareTo(new ApiVersion(2, 2, 2, "")))
        .isEqualTo(0);
  }

  @Test
  public void testFromToSemver() throws Exception {
    SemVer[] semvers = new SemVer[] {
      SemVer.newBuilder().setMajor(2).build(),
      SemVer.newBuilder().setMajor(2).setMinor(1).setPatch(3).build(),
      SemVer.newBuilder().setPrerelease("v1test").build(),
    };
    for (SemVer sm : semvers) {
      assertThat(new ApiVersion(sm).toSemVer()).isEqualTo(sm);
    }
  }

  @Test
  public void testCheckServerSupportedVersions_isSupported() throws Exception {
    assertThat(
            new ApiVersion(2, 1, 1, "")
                .checkServerSupportedVersions(
                    ServerCapabilities.newBuilder()
                        .setLowApiVersion(SemVer.newBuilder().setMajor(2).build())
                        .setHighApiVersion(SemVer.newBuilder().setMajor(3).build())
                        .build())
                .isSupported())
        .isTrue();
  }

  @Test
  public void testCheckServerSupportedVersions_isDeprecated() throws Exception {
    for (ApiVersion v :
        new ApiVersion[] {
          new ApiVersion(0, 0, 0, "v1test"),
          new ApiVersion(0, 0, 0, "v2test"),
          new ApiVersion(1, 0, 0, "")
        }) {
      ApiVersion.ServerSupportedStatus st =
          v.checkServerSupportedVersions(
              ServerCapabilities.newBuilder()
                  .setDeprecatedApiVersion(SemVer.newBuilder().setPrerelease("v1test").build())
                  .setLowApiVersion(SemVer.newBuilder().setMajor(2).build())
                  .setHighApiVersion(SemVer.newBuilder().setMajor(3).build())
                  .build());
      assertThat(st.isDeprecated()).isTrue();
      assertThat(st.getMessage()).contains("deprecated");
      assertThat(st.getMessage()).contains("2.0 to 3.0");
    }
  }

  @Test
  public void testCheckServerSupportedVersions_isUnsupported() throws Exception {
    for (ApiVersion v :
        new ApiVersion[] {new ApiVersion(0, 0, 0, "v1test"), new ApiVersion(3, 1, 0, "")}) {
      ApiVersion.ServerSupportedStatus st =
          v.checkServerSupportedVersions(
              ServerCapabilities.newBuilder()
                  .setLowApiVersion(SemVer.newBuilder().setMajor(2).build())
                  .setHighApiVersion(SemVer.newBuilder().setMajor(3).build())
                  .build());
      assertThat(st.isUnsupported()).isTrue();
      assertThat(st.getMessage()).contains("not supported");
      assertThat(st.getMessage()).contains("2.0 to 3.0");
    }
  }
}
