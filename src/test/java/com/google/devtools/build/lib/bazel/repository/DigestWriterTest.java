// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithDigest;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

@RunWith(JUnit4.class)
public class DigestWriterTest extends BuildViewTestCase {

  private static void assertMarkerFileEscaping(String testCase) {
    String escaped = DigestWriter.escape(testCase);
    assertThat(DigestWriter.unescape(escaped)).isEqualTo(testCase);
  }

  @Test
  public void testMarkerFileEscaping() throws Exception {
    assertMarkerFileEscaping(null);
    assertMarkerFileEscaping("\\0");
    assertMarkerFileEscaping("a\\0");
    assertMarkerFileEscaping("a b");
    assertMarkerFileEscaping("a b c");
    assertMarkerFileEscaping("a \\b");
    assertMarkerFileEscaping("a \\nb");
    assertMarkerFileEscaping("a \\\\nb");
    assertMarkerFileEscaping("a \\\nb");
    assertMarkerFileEscaping("a \nb");
  }

  @Test
  public void testFileValueToMarkerValue() throws Exception {
    RootedPath path =
        RootedPath.toRootedPath(Root.fromPath(rootDirectory), scratch.file("foo", "bar"));

    // Digest should be returned if the FileValue has it.
    FileValue fv = new RegularFileStateValueWithDigest(3, new byte[] {1, 2, 3, 4});
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv)).isEqualTo("01020304");

    // Digest should also be returned if the FileStateValue doesn't have it.
    FileStatus status = Mockito.mock(FileStatus.class);
    when(status.getLastChangeTime()).thenReturn(100L);
    when(status.getNodeId()).thenReturn(200L);
    fv = new RegularFileStateValueWithContentsProxy(3, FileContentsProxy.create(status));
    String expectedDigest = BaseEncoding.base16().lowerCase().encode(path.asPath().getDigest());
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv)).isEqualTo(expectedDigest);
  }
}
