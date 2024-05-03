// Copyright 2022 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import java.net.URISyntaxException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RegistryFactory}. */
@RunWith(JUnit4.class)
public class RegistryFactoryTest {

  @Test
  public void badSchemes() {
    RegistryFactory registryFactory =
        new RegistryFactoryImpl(
            new DownloadManager(new RepositoryCache(), new HttpDownloader()),
            Suppliers.ofInstance(ImmutableMap.of()));
    Throwable exception =
        assertThrows(
            URISyntaxException.class,
            () ->
                registryFactory.createRegistry(
                    "/home/www", ImmutableMap.of(), LockfileMode.UPDATE, ImmutableSet.of()));
    assertThat(exception).hasMessageThat().contains("Registry URL has no scheme");
    exception =
        assertThrows(
            URISyntaxException.class,
            () ->
                registryFactory.createRegistry(
                    "foo://bar", ImmutableMap.of(), LockfileMode.UPDATE, ImmutableSet.of()));
    assertThat(exception).hasMessageThat().contains("Unrecognized registry URL protocol");
  }

  @Test
  public void badPath() {
    RegistryFactory registryFactory =
        new RegistryFactoryImpl(
            new DownloadManager(new RepositoryCache(), new HttpDownloader()),
            Suppliers.ofInstance(ImmutableMap.of()));
    Throwable exception =
        assertThrows(
            URISyntaxException.class,
            () ->
                registryFactory.createRegistry(
                    "file:c:/path/to/workspace/registry",
                    ImmutableMap.of(),
                    LockfileMode.UPDATE,
                    ImmutableSet.of()));
    assertThat(exception).hasMessageThat().contains("Registry URL path is not valid");
  }
}
