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

package com.google.devtools.build.lib.buildeventstream.transports;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.base.Function;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import java.io.File;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Matchers;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link BuildEventTransportFactory}. **/
@RunWith(JUnit4.class)
public class BuildEventTransportFactoryTest {

  private static final Function<Object, Class<?>> GET_CLASS =
      new Function<Object, Class<?>>() {
        @Override
        public Class<?> apply(Object o) {
          return o.getClass();
        }
      };

  private static final BuildEventStreamProtos.BuildEvent BUILD_EVENT_AS_PROTO =
      BuildEventStreamProtos.BuildEvent.newBuilder()
          .setStarted(BuildStarted.newBuilder().setCommand("build"))
          .build();

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEventStreamOptions options;

  @Mock public BuildEvent buildEvent;

  @Mock public PathConverter pathConverter;
  @Mock public ArtifactGroupNamer artifactGroupNamer;

  @Before
  public void before() {
    MockitoAnnotations.initMocks(this);
    when(buildEvent.asStreamProto(Matchers.<BuildEventConverters>any()))
        .thenReturn(BUILD_EVENT_AS_PROTO);
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testCreatesTextFormatFileTransport() throws IOException {
    File textFile = tmp.newFile();
    when(options.getBuildEventTextFile()).thenReturn(textFile.getAbsolutePath());
    when(options.getBuildEventTextFilePathConversion()).thenReturn(true);
    when(options.getBuildEventBinaryFile()).thenReturn("");
    ImmutableSet<BuildEventTransport> transports =
        BuildEventTransportFactory.createFromOptions(options, pathConverter);
    assertThat(FluentIterable.from(transports).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class);
    sendEventsAndClose(buildEvent, transports);
    assertThat(textFile.exists()).isTrue();
  }

  @Test
  public void testCreatesBinaryFormatFileTransport() throws IOException {
    File binaryFile = tmp.newFile();
    when(options.getBuildEventTextFile()).thenReturn("");
    when(options.getBuildEventBinaryFile()).thenReturn(binaryFile.getAbsolutePath());
    when(options.getBuildEventBinaryFilePathConversion()).thenReturn(true);
    ImmutableSet<BuildEventTransport> transports =
        BuildEventTransportFactory.createFromOptions(options, pathConverter);
    assertThat(FluentIterable.from(transports).transform(GET_CLASS))
        .containsExactly(BinaryFormatFileTransport.class);
    sendEventsAndClose(buildEvent, transports);
    assertThat(binaryFile.exists()).isTrue();
  }

  @Test
  public void testCreatesAllTransports() throws IOException {
    File textFile = tmp.newFile();
    File binaryFile = tmp.newFile();
    when(options.getBuildEventTextFile()).thenReturn(textFile.getAbsolutePath());
    when(options.getBuildEventBinaryFile()).thenReturn(binaryFile.getAbsolutePath());
    when(options.getBuildEventBinaryFilePathConversion()).thenReturn(true);
    when(options.getBuildEventTextFilePathConversion()).thenReturn(true);
    ImmutableSet<BuildEventTransport> transports =
        BuildEventTransportFactory.createFromOptions(options, pathConverter);
    assertThat(FluentIterable.from(transports).transform(GET_CLASS))
        .containsExactly(TextFormatFileTransport.class, BinaryFormatFileTransport.class);
    sendEventsAndClose(buildEvent, transports);
    assertThat(textFile.exists()).isTrue();
    assertThat(binaryFile.exists()).isTrue();
  }

  @Test
  public void testCreatesNoTransports() throws IOException {
    when(options.getBuildEventTextFile()).thenReturn("");
    ImmutableSet<BuildEventTransport> transports =
        BuildEventTransportFactory.createFromOptions(options, pathConverter);
    assertThat(transports).isEmpty();
  }

  private void sendEventsAndClose(BuildEvent event, Iterable<BuildEventTransport> transports)
      throws IOException{
    for (BuildEventTransport transport : transports) {
      transport.sendBuildEvent(event, artifactGroupNamer);
      transport.close();
    }
  }
}
