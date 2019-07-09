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

import com.google.api.client.json.GenericJson;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.blobstore.CombinedDiskRemoteBlobStore;
import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.OnDiskBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.blobstore.grpc.GrpcBlobStore;
import com.google.devtools.build.lib.remote.blobstore.http.HttpBlobStore;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.shared.ByteStreamUploader;
import com.google.devtools.build.lib.remote.shared.ReferenceCountedChannel;
import com.google.devtools.build.lib.remote.shared.RemoteRetrier;
import com.google.devtools.build.lib.remote.shared.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.MethodDescriptor;
import io.grpc.inprocess.InProcessChannelBuilder;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SimpleBlobStoreFactory}. */
@RunWith(JUnit4.class)
public class SimpleBlobStoreFactoryTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);
  private RemoteOptions remoteOptions;
  private Path workingDirectory;
  private InMemoryFileSystem fs;
  private final String fakeServerName = "fake server for " + getClass();
  private static ListeningScheduledExecutorService retryService;

  private static class CallCredentialsInterceptor implements ClientInterceptor {
    private final CallCredentials credentials;

    public CallCredentialsInterceptor(CallCredentials credentials) {
      this.credentials = credentials;
    }

    @Override
    public <RequestT, ResponseT> ClientCall<RequestT, ResponseT> interceptCall(
        MethodDescriptor<RequestT, ResponseT> method, CallOptions callOptions, Channel next) {
      assertThat(callOptions.getCredentials()).isEqualTo(credentials);
      // Remove the call credentials to allow testing with dummy ones.
      return next.newCall(method, callOptions.withCallCredentials(null));
    }
  }

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

    assertThat(blobStore).isInstanceOf(CombinedDiskRemoteBlobStore.class);
  }

  @Test
  public void createCombinedCacheWithNotExistingWorkingDirectory() throws IOException {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(workingDirectory.exists()).isFalse();

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(CombinedDiskRemoteBlobStore.class);
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
  public void createGrpcCache() throws IOException {
    remoteOptions.remoteCache = "grpc://doesnotexist:90";

    AuthAndTLSOptions authTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
    authTlsOptions.useGoogleDefaultCredentials = true;
    authTlsOptions.googleCredentials = "/exec/root/creds.json";
    authTlsOptions.googleAuthScopes = ImmutableList.of("dummy.scope");

    GenericJson json = new GenericJson();
    json.put("type", "authorized_user");
    json.put("client_id", "some_client");
    json.put("client_secret", "foo");
    json.put("refresh_token", "bar");
    Scratch scratch = new Scratch();
    scratch.file(authTlsOptions.googleCredentials, new JacksonFactory().toString(json));

    CallCredentials creds;
    try (InputStream in = scratch.resolve(authTlsOptions.googleCredentials).getInputStream()) {
      creds = GoogleAuthUtils.newCallCredentials(in, authTlsOptions.googleAuthScopes);
    }
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions), RemoteRetrier.RETRIABLE_GRPC_ERRORS, retryService);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(InProcessChannelBuilder.forName(fakeServerName).directExecutor()
            .intercept(new CallCredentialsInterceptor(creds)).build());
    ByteStreamUploader uploader =
        new ByteStreamUploader(remoteOptions.remoteInstanceName, channel.retain(), creds,
            remoteOptions.remoteTimeout, retrier);

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(
        remoteOptions,
        channel.retain(),
        creds,
        retrier,
        uploader,
        DIGEST_UTIL);

    assertThat(blobStore).isInstanceOf(GrpcBlobStore.class);
  }

  @Test
  public void createHttpCacheWithProxy() throws IOException {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteProxy = "unix://some-proxy";

    SimpleBlobStore blobStore =
        SimpleBlobStoreFactory.create(remoteOptions, /* creds= */ null, workingDirectory);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void createHttpCacheFailsWithUnsupportedProxyProtocol() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteProxy = "bad-proxy";

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
  public void isHttpCache_httpCacheEnabled() {
    remoteOptions.remoteCache = "http://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isTrue();
  }

  @Test
  public void isHttpCache_httpCacheEnabledInUpperCase() {
    remoteOptions.remoteCache = "HTTP://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isTrue();
  }

  @Test
  public void isHttpCache_httpsCacheEnabled() {
    remoteOptions.remoteCache = "https://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isTrue();
  }

  @Test
  public void isHttpCache_badProtocolStartsWithHttp() {
    remoteOptions.remoteCache = "httplolol://doesnotexist:90";
    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isFalse();
  }

  @Test
  public void isDiskCache_diskCacheEnabled() {
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    assertThat(SimpleBlobStoreFactory.isDiskCache(remoteOptions)).isTrue();
  }

  @Test
  public void isDiskCache_isHttpCache_httpAndDiskCacheEnabled() {
    remoteOptions.remoteCache = "http://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isDiskCache(remoteOptions) && SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isTrue();
  }

  @Test
  public void isDiskCache_isHttpCache_httpsAndDiskCacheEnabled() {
    remoteOptions.remoteCache = "https://doesnotexist:90";
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");

    assertThat(SimpleBlobStoreFactory.isDiskCache(remoteOptions) && SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isTrue();
  }

  @Test
  public void isHttpCache_httpCacheDisabledWhenGrpcEnabled() {
    remoteOptions.remoteCache = "grpc://doesnotexist:90";

    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isFalse();
  }

  @Test
  public void isHttpCache_httpCacheDisabledWhenNoProtocol() {
    remoteOptions.remoteCache = "doesnotexist:90";

    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isFalse();
  }

  @Test
  public void isDiskCache_diskCacheOptionEmpty() {
    remoteOptions.diskCache = PathFragment.EMPTY_FRAGMENT;
    assertThat(SimpleBlobStoreFactory.isDiskCache(remoteOptions)).isFalse();
  }

  @Test
  public void isHttpCache_remoteHttpCacheOptionEmpty() {
    remoteOptions.remoteCache = "";
    assertThat(SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isFalse();
  }

  @Test
  public void isDiskCache_isHttpCache_defaultOptions() {
    assertThat(SimpleBlobStoreFactory.isDiskCache(remoteOptions) || SimpleBlobStoreFactory.isHttpCache(remoteOptions)).isFalse();
  }

  @Test
  public void isGrpcCache_WhenGrpcEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grpc://some-host.com";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenGrpcEnabledUpperCase() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "GRPC://some-host.com";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenDefaultRemoteCacheEnabledForLocalhost() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "localhost:1234";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenDefaultRemoteCacheEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "some-host.com:1234";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenHttpEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "http://some-host.com";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isFalse();
  }

  @Test
  public void isGrpcCache_WhenHttpEnabledWithUpperCase() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "HTTP://some-host.com";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isFalse();
  }

  @Test
  public void isGrpcCache_WhenHttpsEnabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "https://some-host.com";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isFalse();
  }

  @Test
  public void isGrpcCache_WhenUnknownScheme() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grp://some-host.com";

    // TODO(ishikhman): add proper vaildation and flip to false
    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenUnknownSchemeStartsAsGrpc() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "grpcsss://some-host.com";

    // TODO(ishikhman): add proper vaildation and flip to false
    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isTrue();
  }

  @Test
  public void isGrpcCache_WhenEmptyCacheProvided() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteCache = "";

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isFalse();
  }

  @Test
  public void isGrpcCache_WhenRemoteCacheDisabled() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    assertThat(SimpleBlobStoreFactory.isGrpcCache(options)).isFalse();
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
    remoteOptions.remoteProxy = "unix://some-proxy";

    SimpleBlobStore blobStore = SimpleBlobStoreFactory.create(remoteOptions, /* casPath= */ null);

    assertThat(blobStore).isInstanceOf(HttpBlobStore.class);
  }

  @Test
  public void create_httpCacheFailsWithUnsupportedProxyProtocol() {
    remoteOptions.remoteCache = "http://doesnotexist.com";
    remoteOptions.remoteProxy = "bad-proxy";

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
