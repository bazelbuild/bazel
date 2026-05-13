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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.time.Duration;
import java.util.Arrays;
import java.util.SortedMap;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for RemoteOptions. */
@RunWith(TestParameterInjector.class)
public class RemoteOptionsTest {

  @Test
  public void testDefaultValueOfExecProperties() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    assertThat(options.getRemoteDefaultExecProperties()).isEmpty();
  }

  @Test
  public void testRemoteDefaultExecProperties() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.setRemoteDefaultExecPropertiesField(
        Arrays.asList(
            Maps.immutableEntry("ISA", "x86-64"), Maps.immutableEntry("OSFamily", "linux")));

    SortedMap<String, String> properties = options.getRemoteDefaultExecProperties();
    assertThat(properties).isEqualTo(ImmutableSortedMap.of("OSFamily", "linux", "ISA", "x86-64"));
  }

  @Test
  public void testRemoteDefaultExecPropertiesWithDuplicates() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.setRemoteDefaultExecPropertiesField(
        Arrays.asList(
            Maps.immutableEntry("foo", "bar"),
            Maps.immutableEntry("qux", "quux"),
            Maps.immutableEntry("foo", "baz")));
    SortedMap<String, String> properties = options.getRemoteDefaultExecProperties();
    assertThat(properties).isEqualTo(ImmutableSortedMap.of("foo", "baz", "qux", "quux"));
  }

  @Test
  public void testRemoteTimeoutOptionsConverterWithoutUnit() {
    try {
      int seconds = 60;
      Duration convert =
          new CommonRemoteOptions.RemoteDurationConverter().convert(String.valueOf(seconds));
      assertThat(Duration.ofSeconds(seconds)).isEqualTo(convert);
    } catch (OptionsParsingException e) {
      fail(e.getMessage());
    }
  }

  @Test
  public void testRemoteTimeoutOptionsConverterWithUnit() {
    try {
      int milliseconds = 60;
      Duration convert =
          new CommonRemoteOptions.RemoteDurationConverter().convert(milliseconds + "ms");
      assertThat(Duration.ofMillis(milliseconds)).isEqualTo(convert);
    } catch (OptionsParsingException e) {
      fail(e.getMessage());
    }
  }

  @Test
  public void testRemoteMaximumOpenFilesDefault() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    int defaultMax = options.getMaximumOpenFiles();
    assertThat(defaultMax).isEqualTo(-1);
  }

  @Test
  public void testRemoteGrpcLogWithEmptyString() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--remote_grpc_log=test.log", "--remote_grpc_log=");
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getRemoteGrpcLog()).isNull();
  }

  @Test
  public void diskCache_defaultValue_disables() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse();
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isNull();
  }

  @Test
  public void diskCache_noValue_usesDefaultLocation() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache");
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void diskCache_emptyValue_disables() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache=");
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isNull();
  }

  @Test
  public void diskCache_trueValue_usesDefaultLocation(
      @TestParameter({"true", "1", "yes", "t", "y"}) String arg) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache=%s".formatted(arg));
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isEqualTo(PathFragment.EMPTY_FRAGMENT);
  }

  @Test
  public void diskCache_falseValue_disables(
      @TestParameter({"false", "0", "no", "f", "n"}) String arg) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache=%s".formatted(arg));
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isNull();
  }

  @Test
  public void diskCache_negatedForm_disables() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache", "--nodisk_cache");
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isNull();
  }

  @Test
  public void diskCache_explicitPath_usesExplicitPath() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse("--disk_cache=custom/cache/dir");
    RemoteOptions options = parser.getOptions(RemoteOptions.class);
    assertThat(options.getDiskCache()).isEqualTo(PathFragment.create("custom/cache/dir"));
  }
}
