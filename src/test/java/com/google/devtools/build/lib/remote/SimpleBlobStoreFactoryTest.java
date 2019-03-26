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

  @Before
  public final void setUp() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void createCombinedCache() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
  }

  @Test
  public void createCombinedCacheWithNotExistingWorkingDirectory() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(false);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
    verify(relativeDirectory, times(1)).createDirectoryAndParents();
  }

  @Test
  public void createCombinedCacheWithMissingWorkingDirectoryShouldtThrowException() throws IOException {
    // interesting case: workingDirectory = null -> NPE.
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    expectedEx.expect(NullPointerException.class);

    SimpleBlobStoreFactory.create(options, /* creds= */ null, /* workingDirectory= */ null);
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.remoteCacheProxy = "unix://some-proxy";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheFailsWithUnsupportedProxyProtocol() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";
    options.remoteCacheProxy = "bad-proxy";

    expectedEx.expect(Exception.class);
    expectedEx.expectMessage("Remote cache proxy unsupported: bad-proxy");

    SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);
  }

  @Test
  public void createHttpCacheWithoutProxy() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotesist.com";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createDiskCache() throws IOException {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    when(workingDirectory.getRelative(options.diskCache)).thenReturn(relativeDirectory);
    when(relativeDirectory.exists()).thenReturn(true);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(options, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotexist:90";

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenDiskCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_true_whenHttpAndDiskCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "http://doesnotexist:90";
    options.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDiskCacheOptionEmpty() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.diskCache = PathFragment.EMPTY_FRAGMENT;

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_false_whenRemoteHttpCacheEmpty() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteHttpCache = "";

    // TODO(ishikhman): Flip to false. See #7650
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_false_whenDefaultOptions() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(options)).isFalse();
  }
}