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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.api.client.json.GenericJson;
import com.google.api.client.json.gson.GsonFactory;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.Retrier.Backoff;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.grpc.ChannelConnectionFactory;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.util.MutableHandlerRegistry;
import io.reactivex.rxjava3.core.Single;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.junit.After;
import org.junit.Before;

class GrpcCacheClientTestBase {
  protected static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  protected FileSystem fs;
  protected Path execRoot;
  protected FileOutErr outErr;
  protected FakeActionInputFileCache fakeFileCache;
  protected final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  protected final String fakeServerName = "fake server for " + getClass();
  protected Server fakeServer;
  protected RemoteActionExecutionContext context;
  protected RemotePathResolver remotePathResolver;
  protected ListeningScheduledExecutorService retryService;
  private final ArrayList<ReferenceCountedChannel> channels = new ArrayList<>();

  @Before
  public final void setUp() throws Exception {
    // Use a mutable service registry for later registering the service impl for each test case.
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .fallbackHandlerRegistry(serviceRegistry)
            .directExecutor()
            .build()
            .start();
    Chunker.setDefaultChunkSizeForTesting(1000); // Enough for everything to be one chunk.
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/main");
    execRoot.createDirectoryAndParents();
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    remotePathResolver = RemotePathResolver.createDefault(execRoot);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    stdout.getParentDirectory().createDirectoryAndParents();
    stderr.getParentDirectory().createDirectoryAndParents();
    outErr = new FileOutErr(stdout, stderr);
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(
            "none", "none", Digest.getDefaultInstance().getHash(), null);
    context = RemoteActionExecutionContext.create(metadata);
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @After
  public void tearDown() throws Exception {
    channels.forEach(ReferenceCountedChannel::release);
    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  protected GrpcCacheClient newClient(RemoteOptions remoteOptions) throws IOException {
    return newClient(remoteOptions, () -> new ExponentialBackoff(remoteOptions));
  }

  protected GrpcCacheClient newClient(
      RemoteOptions remoteOptions, Supplier<Backoff> backoffSupplier) throws IOException {
    AuthAndTLSOptions authTlsOptions = Options.getDefaults(AuthAndTLSOptions.class);
    authTlsOptions.useGoogleDefaultCredentials = true;
    authTlsOptions.googleCredentials = "/execroot/main/creds.json";
    authTlsOptions.googleAuthScopes = ImmutableList.of("dummy.scope");

    GenericJson json = new GenericJson();
    json.put("type", "authorized_user");
    json.put("client_id", "some_client");
    json.put("client_secret", "foo");
    json.put("refresh_token", "bar");
    Scratch scratch = new Scratch();
    scratch.file(authTlsOptions.googleCredentials, new GsonFactory().toString(json));

    CallCredentialsProvider callCredentialsProvider;
    try (InputStream in = scratch.resolve(authTlsOptions.googleCredentials).getInputStream()) {
      callCredentialsProvider =
          GoogleAuthUtils.newCallCredentialsProvider(
              GoogleAuthUtils.newGoogleCredentialsFromFile(in, authTlsOptions.googleAuthScopes));
    }
    CallCredentials creds = callCredentialsProvider.getCallCredentials();

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            backoffSupplier, RemoteRetrier.RETRIABLE_GRPC_ERRORS, retryService);
    ReferenceCountedChannel channel =
        new ReferenceCountedChannel(
            new ChannelConnectionFactory() {
              @Override
              public Single<? extends ChannelConnection> create() {
                ManagedChannel ch =
                    InProcessChannelBuilder.forName(fakeServerName)
                        .directExecutor()
                        .intercept(new CallCredentialsInterceptor(creds))
                        .intercept(TracingMetadataUtils.newCacheHeadersInterceptor(remoteOptions))
                        .build();
                return Single.just(new ChannelConnection(ch));
              }

              @Override
              public int maxConcurrency() {
                return 100;
              }
            });
    channels.add(channel);
    return new GrpcCacheClient(
        channel, callCredentialsProvider, remoteOptions, retrier, DIGEST_UTIL);
  }

  protected static byte[] downloadBlob(
      RemoteActionExecutionContext context, GrpcCacheClient cacheClient, Digest digest)
      throws IOException, InterruptedException {
    try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      getFromFuture(cacheClient.downloadBlob(context, digest, out));
      return out.toByteArray();
    }
  }

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
}
