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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.blobstore.CombinedDiskHttpBlobStore;
import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    fs.getPath("/etc/something/cache/here").createDirectoryAndParents();

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
  }

  @Test
  public void createCombinedCacheWithNotExistingWorkingDirectory() throws IOException {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(workingDirectory.exists()).isFalse();

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskHttpBlobStore.class);
    assertThat(workingDirectory.exists()).isTrue();
  }

  @Test
  public void createCombinedCacheWithMissingWorkingDirectoryShouldThrowException() {
    // interesting case: workingDirectory = null -> NPE.
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThrows(
        NullPointerException.class,
        () ->
            SimpleBlobStoreFactory.create(
                remoteOptions, /* creds= */ null, /* workingDirectory= */ null));
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "unix://some-proxy";

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheFailsWithUnsupportedProxyProtocol() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "bad-proxy";

    assertThat(
            assertThrows(
                RuntimeException.class,
                () ->
                    SimpleBlobStoreFactory.create(
                        remoteOptions, /* creds= */ null, workingDirectory)))
        .hasMessageThat()
        .contains("Remote cache proxy unsupported: bad-proxy");
  }

  @Test
  public void createHttpCacheWithoutProxy() throws IOException {
    remoteOptions.remoteCache = "http://doesnotexist.com";

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createDiskCache() throws IOException {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void isRemoteCacheOptions_httpCacheEnabled() {
    remoteOptions.remoteCache = "http://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpCacheEnabledInUpperCase() {
    remoteOptions.remoteCache = "HTTP://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpsCacheEnabled() {
    remoteOptions.remoteCache = "https://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_badProtocolStartsWithHttp() {
    remoteOptions.remoteCache = "httplolol://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_diskCacheEnabled() {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpAndDiskCacheEnabled() {
    remoteOptions.remoteCache = "http://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpsAndDiskCacheEnabled() {
    remoteOptions.remoteCache = "https://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();
  }

  @Test
  public void isRemoteCacheOptions_httpCacheDisabledWhenGrpcEnabled() {
    remoteOptions.remoteCache = "grpc://doesnotexist:90";

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_httpCacheDisabledWhenNoProtocol() {
    remoteOptions.remoteCache = "doesnotexist:90";

    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_diskCacheOptionEmpty() {
    remoteOptions.diskCache = PathFragment.EMPTY_FRAGMENT;
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_remoteHttpCacheOptionEmpty() {
    remoteOptions.remoteCache = "";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void isRemoteCacheOptions_defaultOptions() {
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }

  @Test
  public void create_httpCacheWhenHttpAndDiskCacheEnabled() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void create_httpCacheWithProxy() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "unix://some-proxy";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void create_httpCacheFailsWithUnsupportedProxyProtocol() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteCacheProxy = "bad-proxy";

    assertThat(
            assertThrows(
                Exception.class,
                () -> SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null)))
        .hasMessageThat()
        .contains("Remote cache proxy unsupported: bad-proxy");
  }

  @Test
  public void create_httpCacheWithoutProxy() {
    remoteOptions.remoteCache = "http://doesnotexist.com";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void create_diskCacheWithCasPath() {
    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, fs.getPath("/cas/path/is/here"));

    assertThat(blobStore).isInstanceOf(OnDiskBlobStore.class);
  }

  @Test
  public void create_defaultCacheWhenDiskCacheEnabled() {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null);

    assertThat(blobStore).isInstanceOf(ConcurrentMapBlobStore.class);
  }

  @Test
  public void create_defaultCache() {
    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, null);

    assertThat(blobStore).isInstanceOf(ConcurrentMapBlobStore.class);
  }
}
