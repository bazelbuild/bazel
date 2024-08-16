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

import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildToolLogs.LogFileEntry;
import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LocalInstrumentationOutputTest {
  @Test
  public void testLocalInstrumentation_publishNameAndPath() {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path path = fs.getPath("/file");
    InstrumentationOutput localInstrumentationOutput =
        new LocalInstrumentationOutput("local", path);

    assertThat(localInstrumentationOutput).isInstanceOf(LocalInstrumentationOutput.class);
    BuildToolLogCollection buildToolLogCollection = new BuildToolLogCollection();
    localInstrumentationOutput.publish(buildToolLogCollection);
    buildToolLogCollection.freeze();

    assertThat(buildToolLogCollection.getLocalFiles())
        .containsExactly(
            new LogFileEntry(
                "local",
                new LocalFile(
                    path, LocalFileType.LOG, /* artifact= */ null, /* artifactMetadata= */ null)));
  }
}