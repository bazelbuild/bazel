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

import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Progress;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetComplete;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
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

/** Tests {@link BinaryFormatFileTransport}. **/
@RunWith(JUnit4.class)
public class BinaryFormatFileTransportTest {

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEvent buildEvent;

  @Mock public PathConverter pathConverter;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testCreatesFileAndWritesProtoTextFormat() throws IOException {
    File output = tmp.newFile();

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(Matchers.<BuildEventConverters>any())).thenReturn(started);
    BinaryFormatFileTransport transport =
        new BinaryFormatFileTransport(output.getAbsolutePath(), pathConverter);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent progress =
        BuildEventStreamProtos.BuildEvent.newBuilder().setProgress(Progress.newBuilder()).build();
    when(buildEvent.asStreamProto(Matchers.<BuildEventConverters>any())).thenReturn(progress);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent completed =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setCompleted(TargetComplete.newBuilder().setSuccess(true))
            .build();
    when(buildEvent.asStreamProto(Matchers.<BuildEventConverters>any())).thenReturn(completed);
    transport.sendBuildEvent(buildEvent);

    transport.close();
    try (InputStream in = new FileInputStream(output)) {
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(started);
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(progress);
      assertThat(BuildEventStreamProtos.BuildEvent.parseDelimitedFrom(in)).isEqualTo(completed);
      assertThat(in.available()).isEqualTo(0);
    }
  }
}
