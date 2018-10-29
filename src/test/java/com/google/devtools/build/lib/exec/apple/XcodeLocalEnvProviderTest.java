// Copyright 2017 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec.apple;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link XcodeLocalEnvProvider}. */
@RunWith(JUnit4.class)
public class XcodeLocalEnvProviderTest {
  private final FileSystem fs = new JavaIoFileSystem(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);

  @Test
  public void testIOSEnvironmentOnNonDarwin() {
    assumeTrue(OS.getCurrent() == OS.DARWIN);
    try {
      new XcodeLocalEnvProvider(ImmutableMap.of())
          .rewriteLocalEnv(
              ImmutableMap.<String, String>of(
                  AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, "8.4",
                  AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, "iPhoneSimulator"),
              BinTools.forUnitTesting(fs.getPath("/tmp"), ImmutableSet.of("xcode-locator")),
              "bazel");
      fail("action should fail due to being unable to resolve SDKROOT");
    } catch (IOException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Cannot locate iOS SDK on non-darwin operating system");
    }
  }
}
