// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.remote.RemoteModule.CasPathConverter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CasPathConverter}. */
@RunWith(JUnit4.class)
public class CasPathConverterTest {
  private final FileSystem fs = new InMemoryFileSystem();
  private final CasPathConverter converter = new CasPathConverter();

  @Test
  public void noOptionsShouldntCrash() {
    assertThat(converter.apply(fs.getPath("/foo"))).isNull();
  }

  @Test
  public void disabledRemote() {
    converter.options = Options.getDefaults(RemoteOptions.class);
    assertThat(converter.apply(fs.getPath("/foo"))).isNull();
  }

  @Test
  public void enabledRemoteExecutorNoRemoteInstance() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(RemoteOptions.class);
    parser.parse("--remote_cache=machine");
    converter.options = parser.getOptions(RemoteOptions.class);
    Path path = fs.getPath("/foo");
    FileSystemUtils.writeContentAsLatin1(path, "foobar");
    assertThat(converter.apply(fs.getPath("/foo")))
        .isEqualTo("//machine/blobs/3858f62230ac3c915f300c664312c63f/6");
  }

  @Test
  public void enabledRemoteExecutorWithRemoteInstance() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(RemoteOptions.class);
    parser.parse("--remote_cache=machine", "--remote_instance_name=projects/bazel");
    converter.options = parser.getOptions(RemoteOptions.class);
    Path path = fs.getPath("/foo");
    FileSystemUtils.writeContentAsLatin1(path, "foobar");
    assertThat(converter.apply(fs.getPath("/foo")))
        .isEqualTo("//machine/projects/bazel/blobs/3858f62230ac3c915f300c664312c63f/6");
  }
}
