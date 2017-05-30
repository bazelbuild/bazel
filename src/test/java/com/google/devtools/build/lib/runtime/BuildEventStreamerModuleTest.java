// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.FluentIterable;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.runtime.BlazeModule.ModuleEnvironment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link BuildEventStreamerModule}. **/
@RunWith(JUnit4.class)
public class BuildEventStreamerModuleTest {

  private static final Function<Object, Class<?>> GET_CLASS =
      new Function<Object, Class<?>>() {
        @Override
        public Class<?> apply(Object o) {
          return o.getClass();
        }
      };

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEventStreamOptions options;

  @Mock public OptionsProvider optionsProvider;

  @Mock public ModuleEnvironment moduleEnvironment;

  @Mock public Command command;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testReturnsBuildEventStreamerOptions() throws Exception {
    BuildEventStreamerModule module = new BuildEventStreamerModule();
    Iterable<Class<? extends OptionsBase>> commandOptions = module.getCommandOptions(command);
    assertThat(commandOptions).isNotEmpty();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(commandOptions);
    optionsParser.parse(
        "--experimental_build_event_text_file", "/tmp/foo.txt",
        "--experimental_build_event_binary_file", "/tmp/foo.bin");
    BuildEventStreamOptions options = optionsParser.getOptions(BuildEventStreamOptions.class);
    assertThat(options.getBuildEventTextFile()).isEqualTo("/tmp/foo.txt");
    assertThat(options.getBuildEventBinaryFile()).isEqualTo("/tmp/foo.bin");
  }

  @Test
  public void testCreatesStreamerForTextFormatFileTransport() throws Exception {
    when(optionsProvider.getOptions(BuildEventStreamOptions.class)).thenReturn(options);
    when(options.getBuildEventTextFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BuildEventStreamerModule module = new BuildEventStreamerModule();
    Optional<BuildEventStreamer> buildEventStreamer =
        module.tryCreateStreamer(optionsProvider, moduleEnvironment);
    assertThat(buildEventStreamer.get()).isInstanceOf(BuildEventStreamer.class);
    assertThat(FluentIterable.from(buildEventStreamer.get().getTransports()).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBinaryFormatFileTransport() throws Exception {
    when(optionsProvider.getOptions(BuildEventStreamOptions.class)).thenReturn(options);
    when(options.getBuildEventBinaryFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BuildEventStreamerModule module = new BuildEventStreamerModule();
    Optional<BuildEventStreamer> buildEventStreamer =
        module.tryCreateStreamer(optionsProvider, moduleEnvironment);
    assertThat(buildEventStreamer.get()).isInstanceOf(BuildEventStreamer.class);
    assertThat(FluentIterable.from(buildEventStreamer.get().getTransports()).transform(GET_CLASS))
        .containsExactly(BinaryFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForAllTransports() throws Exception {
    when(optionsProvider.getOptions(BuildEventStreamOptions.class)).thenReturn(options);
    when(options.getBuildEventTextFile()).thenReturn(tmp.newFile().getAbsolutePath());
    when(options.getBuildEventBinaryFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BuildEventStreamerModule module = new BuildEventStreamerModule();
    Optional<BuildEventStreamer> buildEventStreamer =
        module.tryCreateStreamer(optionsProvider, moduleEnvironment);
    assertThat(buildEventStreamer.get()).isInstanceOf(BuildEventStreamer.class);
    assertThat(FluentIterable.from(buildEventStreamer.get().getTransports()).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class, BinaryFormatFileTransport.class);
  }

  @Test
  public void testDoesNotCreatesStreamerWithoutTransports() throws Exception {
    when(optionsProvider.getOptions(BuildEventStreamOptions.class)).thenReturn(options);

    BuildEventStreamerModule module = new BuildEventStreamerModule();
    Optional<BuildEventStreamer> buildEventStreamer =
        module.tryCreateStreamer(optionsProvider, moduleEnvironment);
    assertThat(buildEventStreamer).isAbsent();
  }
}
