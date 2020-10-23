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
package com.google.devtools.build.lib.remote.options;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import java.util.Arrays;
import java.util.SortedMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for RemoteOptions. */
@RunWith(JUnit4.class)
public class RemoteOptionsTest {

  @Test
  public void testDefaultValueOfExecProperties() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    assertThat(options.getRemoteDefaultExecProperties()).isEmpty();
  }

  @Test
  public void testRemoteDefaultExecProperties() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteDefaultExecProperties =
        Arrays.asList(
            Maps.immutableEntry("ISA", "x86-64"), Maps.immutableEntry("OSFamily", "linux"));

    SortedMap<String, String> properties = options.getRemoteDefaultExecProperties();
    assertThat(properties).isEqualTo(ImmutableSortedMap.of("OSFamily", "linux", "ISA", "x86-64"));
  }

  @Test
  public void testRemoteDefaultPlatformProperties() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteDefaultPlatformProperties =
        "properties:{name:\"ISA\" value:\"x86-64\"} properties:{name:\"OSFamily\" value:\"linux\"}";

    SortedMap<String, String> properties = options.getRemoteDefaultExecProperties();
    assertThat(properties).isEqualTo(ImmutableSortedMap.of("OSFamily", "linux", "ISA", "x86-64"));
  }

  @Test
  public void testConflictingRemotePlatformAndExecProperties() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteDefaultExecProperties = Arrays.asList(Maps.immutableEntry("ISA", "x86-64"));
    options.remoteDefaultPlatformProperties = "properties:{name:\"OSFamily\" value:\"linux\"}";

    // TODO(buchgr): Use assertThrows once Bazel starts using junit > 4.13
    try {
      options.getRemoteDefaultExecProperties();
      fail();
    } catch (UserExecException e) {
      // Intentionally left empty.
    }
  }

  @Test
  public void testRemoteTimeoutOptionsConverterWithoutUnit() {
    try {
      int seconds = 60;
      Duration convert =
          new RemoteOptions.RemoteTimeoutConverter().convert(String.valueOf(seconds));
      assertThat(Duration.ofSeconds(seconds)).isEqualTo(convert);
    } catch (OptionsParsingException e) {
      fail(e.getMessage());
    }
  }

  @Test
  public void testRemoteTimeoutOptionsConverterWithUnit() {
    try {
      int milliseconds = 60;
      Duration convert = new RemoteOptions.RemoteTimeoutConverter().convert(milliseconds + "ms");
      assertThat(Duration.ofMillis(milliseconds)).isEqualTo(convert);
    } catch (OptionsParsingException e) {
      fail(e.getMessage());
    }
  }
}
