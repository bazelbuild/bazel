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

import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.protobuf.util.JsonFormat;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
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

/** Tests {@link TextFormatFileTransport}. * */
@RunWith(JUnit4.class)
public class JsonFormatFileTransportTest {

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEvent buildEvent;

  @Mock public PathConverter pathConverter;
  @Mock public ArtifactGroupNamer artifactGroupNamer;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void validateMocks() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testCreatesFileAndWritesProtoJsonFormat() throws Exception {
    File output = tmp.newFile();

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(Matchers.<BuildEventConverters>any())).thenReturn(started);
    JsonFormatFileTransport transport =
        new JsonFormatFileTransport(output.getAbsolutePath(), pathConverter);
    transport.sendBuildEvent(buildEvent, artifactGroupNamer);

    transport.close().get();
    try (InputStream in = new FileInputStream(output)) {
      Reader reader = new InputStreamReader(in);
      JsonFormat.Parser parser = JsonFormat.parser();
      BuildEventStreamProtos.BuildEvent.Builder builder =
          BuildEventStreamProtos.BuildEvent.newBuilder();
      parser.merge(reader, builder);
      assertThat(builder.build()).isEqualTo(started);
    }
  }
}
