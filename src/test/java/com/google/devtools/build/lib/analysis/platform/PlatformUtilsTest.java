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

package com.google.devtools.build.lib.analysis.platform;

import static com.google.common.truth.Truth.assertThat;

import build.bazel.remote.execution.v2.Platform;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PlatformUtils } */
@RunWith(JUnit4.class)
public final class PlatformUtilsTest {
  private static RemoteOptions remoteOptions() {
    RemoteOptions remoteOptions = new RemoteOptions();
    remoteOptions.remoteDefaultPlatformProperties =
        String.join(
            "\n",
            "properties: {",
            " name: \"b\"",
            " value: \"2\"",
            "}",
            "properties: {",
            " name: \"a\"",
            " value: \"1\"",
            "}");

    return remoteOptions;
  }

  @Test
  public void testParsePlatformSortsProperties() throws Exception {
    Platform expected =
        Platform.newBuilder()
            .addProperties(Platform.Property.newBuilder().setName("a").setValue("1"))
            .addProperties(Platform.Property.newBuilder().setName("b").setValue("2"))
            .build();
    assertThat(PlatformUtils.getPlatformProto(null, remoteOptions())).isEqualTo(expected);
  }

  @Test
  public void testParsePlatformHandlesNull() throws Exception {
    assertThat(PlatformUtils.getPlatformProto(null, null)).isEqualTo(null);
  }

  @Test
  public void testParsePlatformSortsProperties_ExecProperties() throws Exception {
    // execProperties are chosen even if there are remoteOptions
    ImmutableMap<String, String> map = ImmutableMap.of("aa", "99", "zz", "66", "dd", "11");
    PlatformInfo platformInfo = PlatformInfo.builder().setExecProperties(map).build();

    Platform expected =
        Platform.newBuilder()
            .addProperties(Platform.Property.newBuilder().setName("aa").setValue("99"))
            .addProperties(Platform.Property.newBuilder().setName("dd").setValue("11"))
            .addProperties(Platform.Property.newBuilder().setName("zz").setValue("66"))
            .build();
    // execProperties are sorted by key
    assertThat(PlatformUtils.getPlatformProto(platformInfo, remoteOptions())).isEqualTo(expected);
  }
}
