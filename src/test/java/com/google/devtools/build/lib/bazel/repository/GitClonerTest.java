// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyListOf;
import static org.mockito.Matchers.anyMapOf;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.net.URL;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests for {@link GitCloner}.
 */
@RunWith(JUnit4.class)
public class GitClonerTest extends BuildViewTestCase {
  private FileSystem diskFs = FileSystems.getNativeFileSystem();
  private Path diskTarball;
  private Path outputDirectory;
  private StoredEventHandler eventHandler = new StoredEventHandler();
  private Map<String, String> clientEnvironment = Maps.newHashMap();

  @org.junit.Rule
  public final ExpectedException expected = ExpectedException.none();

  @Before
  public void initialize() throws Exception {
    outputDirectory = diskFs.getPath(System.getenv("TEST_TMPDIR"))
        .getRelative("output-dir");
    diskTarball = diskFs.getPath(System.getenv("TEST_SRCDIR"))
        .getRelative(
            "io_bazel/src/test/java/com/google/devtools/build/lib/bazel/repository/empty.tar.gz");
  }

  @Test
  public void testSha256TarballOkay() throws Exception {
    Rule rule = scratchRule("external", "foo",
        "git_repository(",
        "    name = 'foo',",
        "    remote = 'https://github.com/foo/bar.git',",
        "    tag = '1.2.3',",
        "    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826',",
        ")");
    HttpDownloader downloader = Mockito.mock(HttpDownloader.class);
    when(downloader.download(
        anyListOf(URL.class), any(String.class), eq(Optional.of("tar.gz")), eq(outputDirectory),
        any(ExtendedEventHandler.class), anyMapOf(String.class, String.class)))
        .thenReturn(diskTarball);

    HttpDownloadValue value = GitCloner.clone(
        rule, outputDirectory, eventHandler, clientEnvironment, downloader);
    verify(downloader)
        .download(
            eq(ImmutableList.of(new URL("https://github.com/foo/bar/archive/1.2.3.tar.gz"))),
            any(String.class),
            eq(Optional.of("tar.gz")),
            eq(outputDirectory),
            any(ExtendedEventHandler.class),
            anyMapOf(String.class, String.class));
    assertThat(value).isNotNull();
    assertThat(value.getPath()).isEqualTo(outputDirectory);
  }

  @Test
  public void testNonGitHubSha256Throws() throws Exception {
    Rule nonGitHubRule = scratchRule("external", "foo",
        "git_repository(",
        "    name = 'foo',",
        "    remote = 'https://example.com/foo/bar.git',",
        "    tag = '1.2.3',",
        "    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826',",
        ")");
    HttpDownloader downloader = new HttpDownloader(Mockito.mock(RepositoryCache.class));
    expected.expect(RepositoryFunctionException.class);
    expected.expectMessage("Could not download tarball, but sha256 specified");
    GitCloner.clone(
        nonGitHubRule, outputDirectory, eventHandler, clientEnvironment, downloader);
  }

  @Test
  public void testSha256TarballErrorThrows() throws Exception {
    Rule rule = scratchRule("external", "foo",
        "git_repository(",
        "    name = 'foo',",
        "    remote = 'https://github.com/foo/bar.git',",
        "    tag = '1.2.3',",
        "    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826',",
        ")");
    HttpDownloader downloader = new HttpDownloader(Mockito.mock(RepositoryCache.class));
    expected.expect(RepositoryFunctionException.class);
    expected.expectMessage("Could not download tarball, but sha256 specified");
    GitCloner.clone(
        rule, outputDirectory, eventHandler, clientEnvironment, downloader);
  }
}
