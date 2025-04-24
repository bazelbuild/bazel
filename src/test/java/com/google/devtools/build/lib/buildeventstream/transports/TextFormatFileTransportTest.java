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

import com.google.common.base.Joiner;
import com.google.common.io.Files;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildStarted;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Progress;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetComplete;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.common.options.Options;
import com.google.protobuf.TextFormat;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests {@link TextFormatFileTransport}. **/
@RunWith(JUnit4.class)
public class TextFormatFileTransportTest {
  private final BuildEventProtocolOptions defaultOpts =
      Options.getDefaults(BuildEventProtocolOptions.class);

  @Rule public TemporaryFolder tmp = new TemporaryFolder();

  @Mock public BuildEvent buildEvent;

  @Mock public PathConverter pathConverter;
  @Mock public ArtifactGroupNamer artifactGroupNamer;

  @Before
  public void setUp() throws IOException {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void tearDown() {
    Mockito.validateMockitoUsage();
  }

  @Test
  public void testCreatesFileAndWritesProtoTextFormat() throws Exception {
    File output = tmp.newFile();
    BufferedOutputStream outputStream =
        new BufferedOutputStream(
            java.nio.file.Files.newOutputStream(Paths.get(output.getAbsolutePath())));

    BuildEventStreamProtos.BuildEvent started =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setStarted(BuildStarted.newBuilder().setCommand("build"))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(started);
    TextFormatFileTransport transport =
        new TextFormatFileTransport(
            outputStream,
            defaultOpts,
            new LocalFilesArtifactUploader(),
            artifactGroupNamer,
            BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent progress =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setProgress(Progress.getDefaultInstance())
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(progress);
    transport.sendBuildEvent(buildEvent);

    BuildEventStreamProtos.BuildEvent completed =
        BuildEventStreamProtos.BuildEvent.newBuilder()
            .setCompleted(TargetComplete.newBuilder().setSuccess(true))
            .build();
    when(buildEvent.asStreamProto(ArgumentMatchers.<BuildEventContext>any())).thenReturn(completed);
    transport.sendBuildEvent(buildEvent);

    transport.close().get();
    String contents =
        trimLines(Joiner.on("\n").join(Files.readLines(output, StandardCharsets.UTF_8)));

    assertThat(contents).contains(trimLines(TextFormat.printer().printToString(started)));
    assertThat(contents).contains(trimLines(TextFormat.printer().printToString(progress)));
    assertThat(contents).contains(trimLines(TextFormat.printer().printToString(completed)));
  }

  private static String trimLines(String text) {
    // Replace CRLF with LF and trim leading and trailing spaces.
    return text.replaceAll("\\r", "").replaceAll(" *\\n *", "\n");
  }
}
