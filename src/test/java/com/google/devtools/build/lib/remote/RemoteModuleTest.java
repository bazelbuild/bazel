// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionCacheUpdateCapabilities;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.CapabilitiesGrpc.CapabilitiesImplBase;
import build.bazel.remote.execution.v2.DigestFunction.Value;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.GetCapabilitiesRequest;
import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.remote.execution.v2.SymlinkAbsolutePathStrategy;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.auth.Credentials;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialHelperEnvironment;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.BlazeWorkspace;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.ClientOptions;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommandLinePathFactory;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteModule}. */
@RunWith(JUnit4.class)
public final class RemoteModuleTest {

  private static final ServerCapabilities CACHE_ONLY_CAPS =
      ServerCapabilities.newBuilder()
          .setLowApiVersion(ApiVersion.current.toSemVer())
          .setHighApiVersion(ApiVersion.current.toSemVer())
          .setCacheCapabilities(
              CacheCapabilities.newBuilder()
                  .addDigestFunctions(Value.SHA256)
                  .setActionCacheUpdateCapabilities(
                      ActionCacheUpdateCapabilities.newBuilder().setUpdateEnabled(true).build())
                  .setSymlinkAbsolutePathStrategy(SymlinkAbsolutePathStrategy.Value.ALLOWED)
                  .build())
          .build();

  private static final ServerCapabilities EXEC_AND_CACHE_CAPS =
      ServerCapabilities.newBuilder()
          .setLowApiVersion(ApiVersion.current.toSemVer())
          .setHighApiVersion(ApiVersion.current.toSemVer())
          .setExecutionCapabilities(
              ExecutionCapabilities.newBuilder()
                  .setExecEnabled(true)
                  .setDigestFunction(Value.SHA256)
                  .build())
          .setCacheCapabilities(
              CacheCapabilities.newBuilder().addDigestFunctions(Value.SHA256).build())
          .build();

  private static CommandEnvironment createTestCommandEnvironment(
      RemoteModule remoteModule, RemoteOptions remoteOptions)
      throws IOException, AbruptExitException {
    CoreOptions coreOptions = Options.getDefaults(CoreOptions.class);
    CommonCommandOptions commonCommandOptions = Options.getDefaults(CommonCommandOptions.class);
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    ClientOptions clientOptions = Options.getDefaults(ClientOptions.class);
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);

    AuthAndTLSOptions authAndTLSOptions = Options.getDefaults(AuthAndTLSOptions.class);

    OptionsParsingResult options = mock(OptionsParsingResult.class);
    when(options.getOptions(CoreOptions.class)).thenReturn(coreOptions);
    when(options.getOptions(CommonCommandOptions.class)).thenReturn(commonCommandOptions);
    when(options.getOptions(PackageOptions.class)).thenReturn(packageOptions);
    when(options.getOptions(ClientOptions.class)).thenReturn(clientOptions);
    when(options.getOptions(RemoteOptions.class)).thenReturn(remoteOptions);
    when(options.getOptions(AuthAndTLSOptions.class)).thenReturn(authAndTLSOptions);
    when(options.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);

    String productName = "bazel";
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    ServerDirectories serverDirectories =
        new ServerDirectories(
            scratch.dir("install"), scratch.dir("output"), scratch.dir("user_root"));
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .setProductName(productName)
            .setFileSystem(scratch.getFileSystem())
            .setServerDirectories(serverDirectories)
            .setStartupOptionsProvider(
                OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build())
            .addBlazeModule(new CredentialModule())
            .addBlazeModule(remoteModule)
            .addBlazeModule(new BlockWaitingModule())
            .build();

    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories,
            scratch.dir("/workspace"),
            scratch.dir("/system_javabase"),
            productName);
    BlazeWorkspace workspace = runtime.initWorkspace(directories, BinTools.empty(directories));
    Command command = BuildCommand.class.getAnnotation(Command.class);
    return workspace.initCommand(
        command, options, new ArrayList<>(), 0, 0, ImmutableList.of(), s -> {});
  }

  static class CapabilitiesImpl extends CapabilitiesImplBase {

    private int requestCount;
    private final ServerCapabilities caps;

    CapabilitiesImpl(ServerCapabilities caps) {
      this.caps = caps;
    }

    @Override
    public void getCapabilities(
        GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
      ++requestCount;
      responseObserver.onNext(caps);
      responseObserver.onCompleted();
    }

    int getRequestCount() {
      return requestCount;
    }
  }

  private static Server createFakeServer(String serverName, BindableService... services) {
    MutableHandlerRegistry executionServerRegistry = new MutableHandlerRegistry();
    for (BindableService service : services) {
      executionServerRegistry.addService(ServerInterceptors.intercept(service));
    }
    return InProcessServerBuilder.forName(serverName)
        .fallbackHandlerRegistry(executionServerRegistry)
        .directExecutor()
        .build();
  }

  @Test
  public void testVerifyCapabilities_executionAndCacheForSingleEndpoint() throws Exception {
    CapabilitiesImpl executionServerCapabilitiesImpl = new CapabilitiesImpl(EXEC_AND_CACHE_CAPS);
    String executionServerName = "execution-server";
    Server executionServer = createFakeServer(executionServerName, executionServerCapabilitiesImpl);
    executionServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteExecutor = executionServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      assertThat(executionServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
    } finally {
      executionServer.shutdownNow();
      executionServer.awaitTermination();
    }
  }

  @Test
  public void testVerifyCapabilities_cacheOnlyEndpoint() throws Exception {
    CapabilitiesImpl cacheServerCapabilitiesImpl = new CapabilitiesImpl(CACHE_ONLY_CAPS);
    String cacheServerName = "cache-server";
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteCache = cacheServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      assertThat(cacheServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
    } finally {
      cacheServer.shutdownNow();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testVerifyCapabilities_executionAndCacheForDifferentEndpoints() throws Exception {
    ServerCapabilities caps =
        ServerCapabilities.newBuilder()
            .setLowApiVersion(ApiVersion.current.toSemVer())
            .setHighApiVersion(ApiVersion.current.toSemVer())
            .setExecutionCapabilities(
                ExecutionCapabilities.newBuilder()
                    .setExecEnabled(true)
                    .setDigestFunction(Value.SHA256)
                    .build())
            .setCacheCapabilities(
                CacheCapabilities.newBuilder().addDigestFunctions(Value.SHA256).build())
            .build();
    CapabilitiesImpl executionServerCapabilitiesImpl = new CapabilitiesImpl(caps);
    String executionServerName = "execution-server";
    Server executionServer = createFakeServer(executionServerName, executionServerCapabilitiesImpl);
    executionServer.start();

    CapabilitiesImpl cacheServerCapabilitiesImpl = new CapabilitiesImpl(caps);
    String cacheServerName = "cache-server";
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteExecutor = executionServerName;
      remoteOptions.remoteCache = cacheServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      assertThat(executionServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
      assertThat(cacheServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
    } finally {
      executionServer.shutdownNow();
      cacheServer.shutdownNow();

      executionServer.awaitTermination();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testVerifyCapabilities_executionOnlyAndCacheOnlyEndpoints() throws Exception {
    ServerCapabilities executionOnlyCaps =
        ServerCapabilities.newBuilder()
            .setLowApiVersion(ApiVersion.current.toSemVer())
            .setHighApiVersion(ApiVersion.current.toSemVer())
            .setExecutionCapabilities(
                ExecutionCapabilities.newBuilder()
                    .setExecEnabled(true)
                    .setDigestFunction(Value.SHA256)
                    .build())
            .build();
    CapabilitiesImpl executionServerCapabilitiesImpl = new CapabilitiesImpl(executionOnlyCaps);
    String executionServerName = "execution-server";
    Server executionServer = createFakeServer(executionServerName, executionServerCapabilitiesImpl);
    executionServer.start();

    ServerCapabilities cacheOnlyCaps =
        ServerCapabilities.newBuilder()
            .setLowApiVersion(ApiVersion.current.toSemVer())
            .setHighApiVersion(ApiVersion.current.toSemVer())
            .setCacheCapabilities(
                CacheCapabilities.newBuilder().addDigestFunctions(Value.SHA256).build())
            .build();
    CapabilitiesImpl cacheServerCapabilitiesImpl = new CapabilitiesImpl(cacheOnlyCaps);
    String cacheServerName = "cache-server";
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteExecutor = executionServerName;
      remoteOptions.remoteCache = cacheServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      assertThat(executionServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
      assertThat(cacheServerCapabilitiesImpl.getRequestCount()).isEqualTo(1);
    } finally {
      executionServer.shutdownNow();
      cacheServer.shutdownNow();

      executionServer.awaitTermination();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testLocalFallback_shouldErrorForRemoteCacheWithoutRequiredCapabilities()
      throws Exception {
    ServerCapabilities noneCaps =
        ServerCapabilities.newBuilder()
            .setLowApiVersion(ApiVersion.current.toSemVer())
            .setHighApiVersion(ApiVersion.current.toSemVer())
            .build();
    CapabilitiesImpl cacheServerCapabilitiesImpl = new CapabilitiesImpl(noneCaps);
    String cacheServerName = "cache-server";
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteCache = cacheServerName;
      remoteOptions.remoteLocalFallback = true;
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      assertThrows(AbruptExitException.class, () -> remoteModule.beforeCommand(env));
    } finally {
      cacheServer.shutdownNow();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testLocalFallback_shouldErrorInaccessibleGrpcRemoteCacheIfFlagNotSet()
      throws Exception {
    String cacheServerName = "cache-server";
    CapabilitiesImplBase cacheServerCapabilitiesImpl =
        new CapabilitiesImplBase() {
          @Override
          public void getCapabilities(
              GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
            responseObserver.onError(new UnsupportedOperationException());
          }
        };
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteCache = cacheServerName;
      remoteOptions.remoteLocalFallback = false;
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      assertThrows(AbruptExitException.class, () -> remoteModule.beforeCommand(env));
    } finally {
      cacheServer.shutdownNow();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testLocalFallback_shouldIgnoreInaccessibleGrpcRemoteCache() throws Exception {
    String cacheServerName = "cache-server";
    CapabilitiesImplBase cacheServerCapabilitiesImpl =
        new CapabilitiesImplBase() {
          @Override
          public void getCapabilities(
              GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
            responseObserver.onError(new UnsupportedOperationException());
          }
        };
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteCache = cacheServerName;
      remoteOptions.remoteLocalFallback = true;
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      RemoteActionContextProvider actionContextProvider = remoteModule.getActionContextProvider();
      assertThat(actionContextProvider).isNotNull();
      assertThat(actionContextProvider.getRemoteCache()).isNull();
      assertThat(actionContextProvider.getRemoteExecutionClient()).isNull();
    } finally {
      cacheServer.shutdownNow();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testLocalFallback_shouldIgnoreInaccessibleGrpcRemoteExecutor() throws Exception {
    CapabilitiesImplBase executionServerCapabilitiesImpl =
        new CapabilitiesImplBase() {
          @Override
          public void getCapabilities(
              GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
            responseObserver.onError(new UnsupportedOperationException());
          }
        };
    String executionServerName = "execution-server";
    Server executionServer = createFakeServer(executionServerName, executionServerCapabilitiesImpl);
    executionServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteExecutor = executionServerName;
      remoteOptions.remoteLocalFallback = true;
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      RemoteActionContextProvider actionContextProvider = remoteModule.getActionContextProvider();
      assertThat(actionContextProvider).isNotNull();
      assertThat(actionContextProvider.getRemoteCache()).isNull();
      assertThat(actionContextProvider.getRemoteExecutionClient()).isNull();
    } finally {
      executionServer.shutdownNow();
      executionServer.awaitTermination();
    }
  }

  @Test
  public void testNetrc_netrcWithoutRemoteCache() throws Exception {
    String netrc = "/.netrc";
    FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Scratch scratch = new Scratch(fileSystem);
    scratch.file(netrc, "machine foo.example.org login baruser password barpass");
    AuthAndTLSOptions authAndTLSOptions = Options.getDefaults(AuthAndTLSOptions.class);
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);

    Cache<URI, ImmutableMap<String, ImmutableList<String>>> credentialCache =
        Caffeine.newBuilder().build();

    Credentials credentials =
        RemoteModule.createCredentials(
            CredentialHelperEnvironment.newBuilder()
                .setEventReporter(new Reporter(new EventBus()))
                .setWorkspacePath(fileSystem.getPath("/workspace"))
                .setClientEnvironment(ImmutableMap.of("NETRC", netrc))
                .setHelperExecutionTimeout(Duration.ZERO)
                .build(),
            credentialCache,
            new CommandLinePathFactory(fileSystem, ImmutableMap.of()),
            fileSystem,
            authAndTLSOptions,
            remoteOptions);

    assertThat(credentials).isNotNull();
    assertThat(credentials.getRequestMetadata(URI.create("https://foo.example.org"))).isNotEmpty();
    assertThat(credentials.getRequestMetadata(URI.create("https://bar.example.org"))).isEmpty();
  }

  @Test
  public void testCacheCapabilities_propagatedToRemoteCache() throws Exception {
    CapabilitiesImpl cacheServerCapabilitiesImpl = new CapabilitiesImpl(CACHE_ONLY_CAPS);
    String cacheServerName = "cache-server";
    Server cacheServer = createFakeServer(cacheServerName, cacheServerCapabilitiesImpl);
    cacheServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteCache = cacheServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      RemoteActionContextProvider actionContextProvider = remoteModule.getActionContextProvider();
      assertThat(actionContextProvider).isNotNull();
      assertThat(actionContextProvider.getRemoteCache()).isNotNull();
      assertThat(actionContextProvider.getRemoteCache().getCacheCapabilities())
          .isEqualTo(CACHE_ONLY_CAPS.getCacheCapabilities());
    } finally {
      cacheServer.shutdownNow();
      cacheServer.awaitTermination();
    }
  }

  @Test
  public void testCacheCapabilities_propagatedToRemoteExecutionCache() throws Exception {
    CapabilitiesImpl executionServerCapabilitiesImpl = new CapabilitiesImpl(EXEC_AND_CACHE_CAPS);
    String executionServerName = "execution-server";
    Server executionServer = createFakeServer(executionServerName, executionServerCapabilitiesImpl);
    executionServer.start();

    try {
      RemoteModule remoteModule = new RemoteModule();
      remoteModule.setChannelFactory(
          (target, proxy, options, interceptors) ->
              InProcessChannelBuilder.forName(target).directExecutor().build());

      RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
      remoteOptions.remoteExecutor = executionServerName;

      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      remoteModule.beforeCommand(env);

      assertThat(Thread.interrupted()).isFalse();
      RemoteActionContextProvider actionContextProvider = remoteModule.getActionContextProvider();
      assertThat(actionContextProvider).isNotNull();
      assertThat(actionContextProvider.getRemoteCache()).isNotNull();
      assertThat(actionContextProvider.getRemoteCache().getCacheCapabilities())
          .isEqualTo(EXEC_AND_CACHE_CAPS.getCacheCapabilities());
    } finally {
      executionServer.shutdownNow();
      executionServer.awaitTermination();
    }
  }
}
