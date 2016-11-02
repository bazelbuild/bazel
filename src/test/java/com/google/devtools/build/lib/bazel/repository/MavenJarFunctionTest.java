// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.packages.Rule;

import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import org.apache.maven.settings.Server;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import java.io.IOException;

/**
 * Tests for {@link MavenJarFunction}.
 * TODO(bazel-team): We don't have a good way to set up a mock Maven server yet.
 */
@RunWith(JUnit4.class)
public class MavenJarFunctionTest extends BuildViewTestCase {
  private static final MavenServerValue TEST_SERVER = new MavenServerValue(
      "server", "http://example.com", new Server(), new byte[]{});

  @Test
  public void testInvalidSha1() throws Exception {
    Rule rule = scratchRule("external", "foo",
        "maven_jar(",
        "    name = 'foo',",
        "    artifact = 'x:y:z:1.1',",
        "    sha1 = '12345',",
        ")");
    MavenDownloader downloader = new MavenDownloader(Mockito.mock(RepositoryCache.class));
    try {
      downloader.download(
          "foo", WorkspaceAttributeMapper.of(rule), scratch.dir("/whatever"), TEST_SERVER);
      fail("Invalid sha1 should have thrown.");
    } catch (IOException expected) {
      assertThat(expected.getMessage()).contains("Invalid SHA-1 for maven_jar foo");
    }
  }


  @Test
  public void testValidSha1() throws Exception {
    Rule rule = scratchRule("external", "foo",
        "maven_jar(",
        "    name = 'foo',",
        "    artifact = 'x:y:z:1.1',",
        "    sha1 = 'da39a3ee5e6b4b0d3255bfef95601890afd80709',",
        ")");
    MavenDownloader downloader = new MavenDownloader(Mockito.mock(RepositoryCache.class));
    try {
      downloader.download(
          "foo", WorkspaceAttributeMapper.of(rule), scratch.dir("/whatever"), TEST_SERVER);
      fail("Expected failure to fetch artifact because of nonexistent server and not due to "
          + "the existence of a valid SHA");
    } catch (IOException expected) {
      assertThat(expected.getMessage()).contains("Failed to fetch Maven dependency: "
          + "Could not transfer artifact x:y:z:1.1 from/to foo (http://example.com)");
    }
  }

  @Test
  public void testNoSha1() throws Exception {
    Rule rule = scratchRule("external", "foo",
        "maven_jar(",
        "    name = 'foo',",
        "    artifact = 'x:y:z:1.1',",
        ")");
    MavenDownloader downloader = new MavenDownloader(Mockito.mock(RepositoryCache.class));
    try {
      downloader.download(
          "foo", WorkspaceAttributeMapper.of(rule), scratch.dir("/whatever"), TEST_SERVER);
      fail("Expected failure to fetch artifact because of nonexistent server and not due to "
          + "lack of SHA.");
    } catch (IOException expected) {
      assertThat(expected.getMessage()).contains("Failed to fetch Maven dependency: "
          + "Could not transfer artifact x:y:z:1.1 from/to foo (http://example.com)");
    }
  }
}
