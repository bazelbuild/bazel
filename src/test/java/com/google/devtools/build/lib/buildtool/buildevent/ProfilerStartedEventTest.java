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
package com.google.devtools.build.lib.buildtool.buildevent;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.profiler.Profiler.Format;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ProfilerStartedEvent} */
@RunWith(JUnit4.class)
public class ProfilerStartedEventTest {
  @Test
  public void testLocalJsonProfiler() {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path profilePath = fs.getPath("/tmp/profile");
    ProfilerStartedEvent jsonProfilerStartedEvent =
        new ProfilerStartedEvent(
            profilePath, /* streamingContext= */ null, Format.JSON_TRACE_FILE_FORMAT);

    assertThat(jsonProfilerStartedEvent.getProfilePath()).isSameInstanceAs(profilePath);
    assertThat(jsonProfilerStartedEvent.getStreamingContext()).isNull();
    assertThat(jsonProfilerStartedEvent.getName()).isEqualTo("command.profile.json");
  }

  @Test
  public void testCompressedProfilerWithBepUploadContext() {
    UploadContext streamingContext = mock(UploadContext.class);
    ProfilerStartedEvent compressedProfilerStartedEvent =
        new ProfilerStartedEvent(
            /* profilePath= */ null, streamingContext, Format.JSON_TRACE_FILE_COMPRESSED_FORMAT);

    assertThat(compressedProfilerStartedEvent.getProfilePath()).isNull();
    assertThat(compressedProfilerStartedEvent.getStreamingContext())
        .isSameInstanceAs(streamingContext);
    assertThat(compressedProfilerStartedEvent.getName()).isEqualTo("command.profile.gz");
  }
}
