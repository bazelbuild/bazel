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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule.ModuleEnvironment;
import com.google.devtools.build.lib.runtime.BuildEventStreamer;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
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

/** Tests {@link BuildEventServiceModule}. **/
@RunWith(JUnit4.class)
public class BazelBuildEventServiceModuleTest {

  private static final Function<Object, Class<?>> GET_CLASS =
      new Function<Object, Class<?>>() {
        @Override
        public Class<?> apply(Object o) {
          return o.getClass();
        }
      };

  private static final PathConverter PATH_CONVERTER =
      new PathConverter() {
        @Override
        public String apply(Path path) {
          return path.getPathString();
        }
      };

  private Reporter reporter;

  private BuildEventServiceOptions besOptions;

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEventStreamOptions bepOptions;

  @Mock public OptionsProvider optionsProvider;

  @Mock public ModuleEnvironment moduleEnvironment;

  @Mock public EventHandler commandLineReporter;

  @Mock public EventBus eventBus;

  @Mock public Clock clock;

  @Mock public Command command;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
    // Reporter is final and thus can't be mocked.
    reporter = new Reporter(eventBus);
    besOptions = Options.getDefaults(BuildEventServiceOptions.class);

    when(optionsProvider.getOptions(BuildEventStreamOptions.class)).thenReturn(bepOptions);
    when(optionsProvider.getOptions(BuildEventServiceOptions.class)).thenReturn(besOptions);
    when(optionsProvider.getOptions(AuthAndTLSOptions.class))
        .thenReturn(Options.getDefaults(AuthAndTLSOptions.class));
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testReturnsBuildEventStreamerOptions() throws Exception {
    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    Iterable<Class<? extends OptionsBase>> commandOptions = module.getCommandOptions(command);
    assertThat(commandOptions).isNotEmpty();
    OptionsParser optionsParser = OptionsParser.newOptionsParser(commandOptions);
    optionsParser.parse(
        "--experimental_build_event_text_file", "/tmp/foo.txt",
        "--experimental_build_event_binary_file", "/tmp/foo.bin",
        "--experimental_build_event_json_file", "/tmp/foo.json");
    BuildEventStreamOptions options = optionsParser.getOptions(BuildEventStreamOptions.class);
    assertThat(options.getBuildEventTextFile()).isEqualTo("/tmp/foo.txt");
    assertThat(options.getBuildEventBinaryFile()).isEqualTo("/tmp/foo.bin");
    assertThat(options.getBuildEventJsonFile()).isEqualTo("/tmp/foo.json");
  }

  @Test
  public void testCreatesStreamerForTextFormatFileTransport() throws Exception {
    when(bepOptions.getBuildEventTextFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "build");
    assertThat(buildEventStreamer).isNotNull();
    verifyNoMoreInteractions(moduleEnvironment);
    assertThat(FluentIterable.from(buildEventStreamer.getTransports()).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBinaryFormatFileTransport() throws Exception {
    when(bepOptions.getBuildEventBinaryFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "test");
    assertThat(buildEventStreamer).isNotNull();
    verifyNoMoreInteractions(moduleEnvironment);
    assertThat(FluentIterable.from(buildEventStreamer.getTransports()).transform(GET_CLASS))
        .containsExactly(BinaryFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForJsonFormatFileTransport() throws Exception {
    when(bepOptions.getBuildEventJsonFile()).thenReturn(tmp.newFile().getAbsolutePath());

    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "fetch");
    assertThat(buildEventStreamer).isNotNull();
    verifyNoMoreInteractions(moduleEnvironment);
    assertThat(FluentIterable.from(buildEventStreamer.getTransports()).transform(GET_CLASS))
        .containsExactly(JsonFormatFileTransport.class);
  }

  @Test
  public void testCreatesStreamerForBesTransport() throws Exception {
    besOptions.besBackend = "does.not.exist:1234";

    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "build");
    assertThat(buildEventStreamer).isNotNull();
  }

  @Test
  public void testCreatesStreamerForAllTransports() throws Exception {
    when(bepOptions.getBuildEventTextFile()).thenReturn(tmp.newFile().getAbsolutePath());
    when(bepOptions.getBuildEventBinaryFile()).thenReturn(tmp.newFile().getAbsolutePath());
    when(bepOptions.getBuildEventJsonFile()).thenReturn(tmp.newFile().getAbsolutePath());
    besOptions.besBackend = "does.not.exist:1234";

    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "test");
    assertThat(buildEventStreamer).isNotNull();
    verifyNoMoreInteractions(moduleEnvironment);
    assertThat(FluentIterable.from(buildEventStreamer.getTransports()).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class, BinaryFormatFileTransport.class,
            JsonFormatFileTransport.class, BuildEventServiceTransport.class);
  }

  @Test
  public void testDoesNotCreatesStreamerWithoutTransports() throws Exception {
    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    BuildEventStreamer buildEventStreamer =
        module.tryCreateStreamer(
            optionsProvider,
            commandLineReporter,
            moduleEnvironment,
            clock,
            PATH_CONVERTER,
            reporter,
            "foo",
            "bar",
            "fetch");
    assertThat(buildEventStreamer).isNull();
  }

  @Test
  public void testKeywords() throws Exception {
    besOptions.besKeywords = ImmutableList.of("keyword0", "keyword1", "keyword0");
    BazelBuildEventServiceModule module = new BazelBuildEventServiceModule();
    assertThat(module.keywords(besOptions))
        .containsExactly("user_keyword=keyword0", "user_keyword=keyword1");
  }
}
