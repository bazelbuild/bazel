// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.ExecutionException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BuildEventArtifactInstrumentationOutputTest {
  private BuildEventArtifactInstrumentationOutput.Builder bepInstrumentationOutputBuilder;

  @Before
  public void setup() {
    bepInstrumentationOutputBuilder = new BuildEventArtifactInstrumentationOutput.Builder();
  }

  @Test
  public void testBepInstrumentationBuilder_failToBuildWhenMissingName() {
    Throwable throwable =
        assertThrows(
            NullPointerException.class,
            bepInstrumentationOutputBuilder.setUploader(mock(BuildEventArtifactUploader.class))
                ::build);
    assertThat(throwable)
        .hasMessageThat()
        .isEqualTo("Cannot create BuildEventArtifactInstrumentationOutput without name");
  }

  @Test
  public void testBepInstrumentationBuilder_failToBuildWhenMissingBepUploader() {
    Throwable throwable =
        assertThrows(
            NullPointerException.class, bepInstrumentationOutputBuilder.setName("bep")::build);
    assertThat(throwable)
        .hasMessageThat()
        .isEqualTo("Cannot create BuildEventArtifactInstrumentationOutput without bepUploader");
  }

  @Test
  public void testBepInstrumentation_cannotPublishIfUploadNeverStarts() {
    BuildEventArtifactUploader fakeBuildEventArtifactUploader =
        mock(BuildEventArtifactUploader.class);
    InstrumentationOutput bepInstrumentationOutput =
        bepInstrumentationOutputBuilder
            .setName("bep")
            .setUploader(fakeBuildEventArtifactUploader)
            .build();

    BuildToolLogCollection buildToolLogCollection = new BuildToolLogCollection();
    assertThrows(
        NullPointerException.class, () -> bepInstrumentationOutput.publish(buildToolLogCollection));
  }

  @Test
  public void testBepInstrumentation_publishNameAndUriFuture()
      throws ExecutionException, InterruptedException, IOException {
    UploadContext fakeUploadLoadContext =
        new UploadContext() {
          @Override
          public OutputStream getOutputStream() {
            return new ByteArrayOutputStream();
          }

          @Override
          public ListenableFuture<String> uriFuture() {
            return immediateFuture("uri/abc12345");
          }
        };
    BuildEventArtifactUploader fakeBuildEventArtifactUploader =
        mock(BuildEventArtifactUploader.class);
    when(fakeBuildEventArtifactUploader.startUpload(LocalFileType.LOG, null))
        .thenReturn(fakeUploadLoadContext);

    InstrumentationOutput bepInstrumentationOutput =
        bepInstrumentationOutputBuilder
            .setName("bep")
            .setUploader(fakeBuildEventArtifactUploader)
            .build();
    // Create the OutputStream will enforce fakeBuildEventArtifactUploader to create the
    // uploadContext.
    var unused = bepInstrumentationOutput.createOutputStream();
    assertThat(bepInstrumentationOutput)
        .isInstanceOf(BuildEventArtifactInstrumentationOutput.class);

    BuildToolLogCollection buildToolLogCollection = new BuildToolLogCollection();
    bepInstrumentationOutput.publish(buildToolLogCollection);
    buildToolLogCollection.freeze();

    assertThat(buildToolLogCollection.toEvent().remoteUploads()).hasSize(1);
    ListenableFuture<String> soleRemoteUploadUri =
        buildToolLogCollection.toEvent().remoteUploads().get(0);
    assertThat(soleRemoteUploadUri.get()).isEqualTo("uri/abc12345");
  }
}
