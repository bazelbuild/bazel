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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithDigest;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.FileValue.RegularFileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link RepositoryFunction} */
@RunWith(JUnit4.class)
public class RepositoryFunctionTest extends BuildViewTestCase {

  private static void assertMarkerFileEscaping(String testCase) {
    String escaped = RepositoryDelegatorFunction.escape(testCase);
    assertThat(RepositoryDelegatorFunction.unescape(escaped)).isEqualTo(testCase);
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

    // Digest should be returned if the FileStateValue has it.
    FileStateValue fsv = new RegularFileStateValueWithDigest(3, new byte[] {1, 2, 3, 4});
    FileValue fv = new RegularFileValue(path, fsv);
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(fv)).isEqualTo("01020304");

    // Digest should also be returned if the FileStateValue doesn't have it.
    FileStatus status = Mockito.mock(FileStatus.class);
    when(status.getLastChangeTime()).thenReturn(100L);
    when(status.getNodeId()).thenReturn(200L);
    fsv = new RegularFileStateValueWithContentsProxy(3, FileContentsProxy.create(status));
    fv = new RegularFileValue(path, fsv);
    String expectedDigest = BaseEncoding.base16().lowerCase().encode(path.asPath().getDigest());
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(fv)).isEqualTo(expectedDigest);
  }
}
