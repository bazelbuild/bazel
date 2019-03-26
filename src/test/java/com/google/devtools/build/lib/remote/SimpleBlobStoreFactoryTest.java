// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;


@RunWith(JUnit4.class)
public class SimpleBlobStoreFactoryTest {

  @Rule
  public ExpectedException expectedEx = ExpectedException.none();

  @Mock
  private Path workingDirectory;

  @Mock
  private Path relativeDirectory;

  private RemoteOptions remoteOptions;

  @Before
  public final void setUp() {
    MockitoAnnotations.initMocks(this);
    remoteOptions = Options.getDefaults(RemoteOptions.class);
  }

  @Test
  public void createCombinedCache() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(remoteOptions.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
  }

  @Test
  public void createCombinedCacheWithNotExistingWorkingDirectory() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(remoteOptions.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(false);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
    verify(relativeDirectory, times(1)).createDirectoryAndParents();
  }

  @Test
  public void createCombinedCacheWithMissingWorkingDirectoryShouldtThrowException() throws IOException {
    // interesting case: workingDirectory = null -> NPE.
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    expectedEx.expect(NullPointerException.class);

    SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, /* workingDirectory= */ null);
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";
    remoteOptions.remoteCacheProxy = "unix://some-proxy";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheFailsWithUnsupportedProxyProtocol() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";
    remoteOptions.remoteCacheProxy = "bad-proxy";

    expectedEx.expect(Exception.class);
    expectedEx.expectMessage("Remote cache proxy unsupported: bad-proxy");

    SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);
  }

  @Test
  public void createHttpCacheWithoutProxy() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotesist.com";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createDiskCache() throws IOException {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(remoteOptions.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpCache() {
    remoteOptions.remoteHttpCache = "http://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenDiskCache() {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpAndDiskCache() {
    remoteOptions.remoteHttpCache = "http://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDiskCacheOptionEmpty() {
    remoteOptions.diskCache = PathFragment.EMPTY_FRAGMENT;
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_false_whenRemoteHttpCacheEmpty() {
    remoteOptions.remoteHttpCache = "";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDefaultOptions() {
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }
}
