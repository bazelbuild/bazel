// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.bytesCountToDisplayString;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Utils}. */
@RunWith(JUnit4.class)
public class UtilsTest {
  @Test
  public void bytesCountToDisplayString_works() {
    assertThat(bytesCountToDisplayString(1000)).isEqualTo("1000 B");
    assertThat(bytesCountToDisplayString(1 << 10)).isEqualTo("1.0 KiB");
    assertThat(bytesCountToDisplayString((1 << 10) + (1 << 10) / 10)).isEqualTo("1.1 KiB");
    assertThat(bytesCountToDisplayString(1 << 20)).isEqualTo("1.0 MiB");
    assertThat(bytesCountToDisplayString((1 << 20) + (1 << 20) / 10)).isEqualTo("1.1 MiB");
    assertThat(bytesCountToDisplayString(1 << 30)).isEqualTo("1.0 GiB");
    assertThat(bytesCountToDisplayString((1 << 30) + (1 << 30) / 10)).isEqualTo("1.1 GiB");
    assertThat(bytesCountToDisplayString(1L << 40)).isEqualTo("1.0 TiB");
    assertThat(bytesCountToDisplayString((1L << 40) + (1L << 40) / 10)).isEqualTo("1.1 TiB");
    assertThat(bytesCountToDisplayString(1L << 50)).isEqualTo("1024.0 TiB");
  }
}
