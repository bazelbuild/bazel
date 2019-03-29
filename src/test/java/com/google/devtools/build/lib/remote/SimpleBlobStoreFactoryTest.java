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

import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

/** Tests for {@link SimpleBlobStoreFactory}. */
@RunWith(JUnit4.class)
public class SimpleBlobStoreFactoryTest {

  private RemoteOptions remoteOptions;
  private Path workingDirectory;
  private InMemoryFileSystem fs;

  @Before
  public final void setUp() {
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    workingDirectory = fs.getPath("/etc/something");
    remoteOptions = Options.getDefaults(RemoteOptions.class);
  }

  @Test
  public void createCombinedCacheWithExistingWorkingDirectory() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    fs.getPath("/etc/something/cache/here").createDirectoryAndParents();

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
  }

  @Test
  public void createCombinedCacheWithNotExistingWorkingDirectory() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(workingDirectory.exists()).isFalse();

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
    assertThat(workingDirectory.exists()).isTrue();
  }

  @Test
  public void createCombinedCacheWithMissingWorkingDirectoryShouldThrowException() {
    // interesting case: workingDirectory = null -> NPE.
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThrows(NullPointerException.class,
        () -> SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, /* workingDirectory= */ null));
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "unix://some-proxy";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheFailsWithUnsupportedProxyProtocol() {
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "bad-proxy";

    assertThat(
        assertThrows(RuntimeException.class,
            () -> SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory)))
        .hasMessageThat().contains("Remote cache proxy unsupported: bad-proxy");
  }

  @Test
  public void createHttpCacheWithoutProxy() throws IOException {
    remoteOptions.remoteHttpCache = "http://doesnotexist.com";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createDiskCache() throws IOException {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void isRemoteCacheOptions_httpCacheEnabled() {
    remoteOptions.remoteHttpCache = "http://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_diskCacheEnabled() {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpAndDiskCacheEnabled() {
    remoteOptions.remoteHttpCache = "http://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_diskCacheOptionEmpty() {
    remoteOptions.diskCache = PathFragment.EMPTY_FRAGMENT;
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_remoteHttpCacheOptionEmpty() {
    remoteOptions.remoteHttpCache = "";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_defaultOptions() {
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }
}
