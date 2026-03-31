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

import static org.junit.Assert.assertThrows;

import com.google.common.hash.HashCode;
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
}
