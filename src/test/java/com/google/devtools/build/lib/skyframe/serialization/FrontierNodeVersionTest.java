// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.skyframe.IntVersion;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FrontierNodeVersionTest {

  @Test
  public void constructor_nullClientId_throwsNullPointerException() {
    assertThrows(
        NullPointerException.class,
        () ->
            new FrontierNodeVersion(
                "checksum",
                HashCode.fromInt(1),
                new byte[] {1},
                IntVersion.of(1),
                "distinguisher",
                /* useFakeStampData= */ false,
                /* clientId= */ null));
  }

  @Test
  public void constructor_nullStarlarkSemanticsFingerprint_throwsNullPointerException() {
    assertThrows(
        NullPointerException.class,
        () ->
            new FrontierNodeVersion(
                "checksum",
                HashCode.fromInt(1),
                /* starlarkSemanticsFingerprint= */ null,
                IntVersion.of(1),
                "distinguisher",
                /* useFakeStampData= */ false,
                Optional.empty()));
  }

  @Test
  public void constructor_validArgs_success() {
    var unused =
        new FrontierNodeVersion(
            "checksum",
            HashCode.fromInt(1),
            new byte[] {1},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ false,
            Optional.empty());
  }

  @Test
  public void getConfigAndStarlarkSemanticsHash_includesTopLevelConfigAndStarlarkSemantics() {
    var base =
        new FrontierNodeVersion(
            "checksum1",
            HashCode.fromInt(1),
            new byte[] {1, 2},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ false,
            Optional.empty());
    var differentTopLevelConfig =
        new FrontierNodeVersion(
            "checksum2",
            HashCode.fromInt(1),
            new byte[] {1, 2},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ false,
            Optional.empty());
    var differentStarlarkSemantics =
        new FrontierNodeVersion(
            "checksum1",
            HashCode.fromInt(1),
            new byte[] {3, 4},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ false,
            Optional.empty());

    assertThat(base.getConfigAndStarlarkSemanticsHash())
        .isNotEqualTo(differentTopLevelConfig.getConfigAndStarlarkSemanticsHash());
    assertThat(base.getConfigAndStarlarkSemanticsHash())
        .isNotEqualTo(differentStarlarkSemantics.getConfigAndStarlarkSemanticsHash());
  }

  @Test
  public void
      getReleaseVersionAndInstallationHash_releasedBuild_includesReleaseNameAndInstallMd5() {
    var version =
        new FrontierNodeVersion(
            "checksum",
            HashCode.fromString("0123456789abcdef0123456789abcdef"),
            new byte[] {1},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ true,
            Optional.empty());
    var blazeVersionInfo =
        new BlazeVersionInfo(ImmutableMap.of(BlazeVersionInfo.BUILD_LABEL, "8.0.0"));

    assertThat(version.getReleaseVersionAndInstallationHash(blazeVersionInfo))
        .isEqualTo("release 8.0.0-0123456789abcdef0123456789abcdef");
  }

  @Test
  public void getReleaseVersionAndInstallationHash_developmentBuild_usesDevelopmentVersionPrefix() {
    var version =
        new FrontierNodeVersion(
            "checksum",
            HashCode.fromString("0123456789abcdef0123456789abcdef"),
            new byte[] {1},
            IntVersion.of(1),
            "distinguisher",
            /* useFakeStampData= */ true,
            Optional.empty());
    var blazeVersionInfo = new BlazeVersionInfo(ImmutableMap.of());

    assertThat(version.getReleaseVersionAndInstallationHash(blazeVersionInfo))
        .isEqualTo("development version-0123456789abcdef0123456789abcdef");
  }
}
