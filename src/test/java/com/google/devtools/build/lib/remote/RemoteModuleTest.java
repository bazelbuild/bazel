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
import com.google.devtools.build.lib.remote.circuitbreaker.FailureCircuitBreaker;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/** Tests for {@link RemoteModule}. */
@RunWith(Parameterized.class)
public final class RemoteModuleTest {
  private static final String EXECUTION_SERVER_NAME = "execution-server";
  private static final String CACHE_SERVER_NAME = "cache-server";
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

  private static final ServerCapabilities EXEC_ONLY_CAPS =
      ServerCapabilities.newBuilder()
          .setLowApiVersion(ApiVersion.current.toSemVer())
          .setHighApiVersion(ApiVersion.current.toSemVer())
          .setExecutionCapabilities(
              ExecutionCapabilities.newBuilder()
                  .setExecEnabled(true)
                  .setDigestFunction(Value.SHA256)
                  .build())
          .build();

  private static final ServerCapabilities NONE_CAPS =
      ServerCapabilities.newBuilder()
          .setLowApiVersion(ApiVersion.current.toSemVer())
          .setHighApiVersion(ApiVersion.current.toSemVer())
          .build();

  private static final CapabilitiesImpl INACCESSIBLE_GRPC_REMOTE =
      new CapabilitiesImpl(null) {
        @Override
        public void getCapabilities(
            GetCapabilitiesRequest request, StreamObserver<ServerCapabilities> responseObserver) {
          responseObserver.onError(new UnsupportedOperationException());
        }
      };

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

  private static void assertCircuitBreakerInstance(Retrier.CircuitBreaker circuitBreaker, RemoteOptions remoteOptions) {
    if (remoteOptions.circuitBreakerStrategy == RemoteOptions.CircuitBreakerStrategy.FAILURE) {
      assertThat(circuitBreaker).isInstanceOf(FailureCircuitBreaker.class);
    }
    if (remoteOptions.circuitBreakerStrategy == null) {
      assertThat(circuitBreaker).isEqualTo(Retrier.ALLOW_ALL_CALLS);
    }
  }

  private final Map<String, CapabilitiesImpl> endpointCapabilityMap;
  private final Consumer<RemoteOptions> remoteOptionsConsumer;
  private final Boolean throwsException;
  private final CacheCapabilities cacheCapabilities;
  private final Boolean hasRemoteExecutionCapability;
  private RemoteModule remoteModule;
  private RemoteOptions remoteOptions;

  @Before
  public void initialize() {
    remoteModule = new RemoteModule();
    remoteModule.setChannelFactory(
        (target, proxy, options, interceptors) ->
            InProcessChannelBuilder.forName(target).directExecutor().build());
    remoteOptions = Options.getDefaults(RemoteOptions.class);
  }

  public RemoteModuleTest(Map<String, CapabilitiesImpl> endpointCapabilityMap,
                            Consumer<RemoteOptions> remoteOptionsConsumer,
                            CacheCapabilities cacheCapabilities,
                            Boolean hasRemoteExecutionCapability,
                            Boolean throwsException) {
    this.endpointCapabilityMap = endpointCapabilityMap;
    this.remoteOptionsConsumer = remoteOptionsConsumer;
    this.cacheCapabilities = cacheCapabilities;
    this.hasRemoteExecutionCapability = hasRemoteExecutionCapability;
    this.throwsException = throwsException;
  }

  @Parameterized.Parameters
  public static Collection<Object[]> remoteModuleTestSet() {
    return Arrays.asList(new Object[][]{
        // executionAndCacheForSingleEndpoint and verify RemoteExecutionCache
        {
            Map.of(EXECUTION_SERVER_NAME, new CapabilitiesImpl(EXEC_AND_CACHE_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> remoteOptions.remoteExecutor = EXECUTION_SERVER_NAME),
            EXEC_AND_CACHE_CAPS.getCacheCapabilities(),
            true,
            false
        },
        // executionAndCacheForSingleEndpoint with circuitBreaker
        {
            Map.of(EXECUTION_SERVER_NAME, new CapabilitiesImpl(EXEC_AND_CACHE_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteExecutor = EXECUTION_SERVER_NAME;
              remoteOptions.circuitBreakerStrategy = RemoteOptions.CircuitBreakerStrategy.FAILURE;
            }),
            EXEC_AND_CACHE_CAPS.getCacheCapabilities(),
            true,
            false
        },
        // cacheOnlyEndpoint and verify RemoteCache
        {
            Map.of(CACHE_SERVER_NAME, new CapabilitiesImpl(CACHE_ONLY_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
                remoteOptions.remoteCache = CACHE_SERVER_NAME;
                remoteOptions.circuitBreakerStrategy = RemoteOptions.CircuitBreakerStrategy.FAILURE;
            }),
            CACHE_ONLY_CAPS.getCacheCapabilities(),
            false,
            false
        },
        // executionAndCacheForDifferentEndpoints and verify RemoteExecutionCache
        {
            Map.of(EXECUTION_SERVER_NAME, new CapabilitiesImpl(EXEC_AND_CACHE_CAPS),
                CACHE_SERVER_NAME, new CapabilitiesImpl(EXEC_AND_CACHE_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteExecutor = EXECUTION_SERVER_NAME;
              remoteOptions.remoteCache = CACHE_SERVER_NAME;
              remoteOptions.circuitBreakerStrategy = RemoteOptions.CircuitBreakerStrategy.FAILURE;
            }),
            EXEC_AND_CACHE_CAPS.getCacheCapabilities(),
            true,
            false
        },
        // executionOnlyAndCacheOnlyEndpoints and verify RemoteCache
        {
            Map.of(EXECUTION_SERVER_NAME, new CapabilitiesImpl(EXEC_ONLY_CAPS),
                CACHE_SERVER_NAME, new CapabilitiesImpl(CACHE_ONLY_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteExecutor = EXECUTION_SERVER_NAME;
              remoteOptions.remoteCache = CACHE_SERVER_NAME;
            }),
            CACHE_ONLY_CAPS.getCacheCapabilities(),
            true,
            false
        },
        // shouldErrorForRemoteCacheWithoutRequiredCapabilities
        {
            Map.of(CACHE_SERVER_NAME, new CapabilitiesImpl(NONE_CAPS)),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteCache = CACHE_SERVER_NAME;
              remoteOptions.remoteLocalFallback = true;
            }),
            null,
            false,
            true
        },
        // shouldErrorInaccessibleGrpcRemoteCacheIfRemoteLocalFallbackNotSet
        {
            Map.of(CACHE_SERVER_NAME, INACCESSIBLE_GRPC_REMOTE),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteCache = CACHE_SERVER_NAME;
              remoteOptions.remoteLocalFallback = false;
            }),
            null,
            false,
            true
        },
        // shouldIgnoreInaccessibleGrpcRemoteCacheIfRemoteLocalFallbackSet
        {
            Map.of(CACHE_SERVER_NAME, INACCESSIBLE_GRPC_REMOTE),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteCache = CACHE_SERVER_NAME;
              remoteOptions.remoteLocalFallback = true;
            }),
            null,
            false,
            false
        },
        // shouldIgnoreInaccessibleGrpcRemoteExecutorIfRemoteLocalFallbackSet
        {
            Map.of(EXECUTION_SERVER_NAME, INACCESSIBLE_GRPC_REMOTE),
            (Consumer<RemoteOptions>) ((remoteOptions) -> {
              remoteOptions.remoteExecutor = EXECUTION_SERVER_NAME;
              remoteOptions.remoteLocalFallback = true;
            }),
            null,
            false,
            false
        }

    });
  }

  @Test
  public void testBeforeCommand() throws Exception {
    List<CapabilitiesImpl> capabilitiesList = new ArrayList<>();
    List<Server> serverList = new ArrayList<>();
    for (Map.Entry<String, CapabilitiesImpl> entry : endpointCapabilityMap.entrySet()) {
      String serverName = entry.getKey();
      CapabilitiesImpl capabilities = entry.getValue();
      Server server = createFakeServer(serverName, capabilities);
      server.start();
      capabilitiesList.add(capabilities);
      serverList.add(server);
    }

    try {
      remoteOptionsConsumer.accept(remoteOptions);
      CommandEnvironment env = createTestCommandEnvironment(remoteModule, remoteOptions);

      if (throwsException) {
        assertThrows(AbruptExitException.class, () -> remoteModule.beforeCommand(env));
        return;
      }
      remoteModule.beforeCommand(env);
      assertThat(Thread.interrupted()).isFalse();

      capabilitiesList.forEach(capabilities -> {
        if (!INACCESSIBLE_GRPC_REMOTE.equals(capabilities)) {
          assertThat(capabilities.getRequestCount()).isEqualTo(1);
        }
      });

      RemoteActionContextProvider actionContextProvider = remoteModule.getActionContextProvider();
      assertThat(actionContextProvider).isNotNull();
      GrpcCacheClient grpcCacheClient;
      GrpcRemoteExecutor grpcRemoteExecutor;
      if (cacheCapabilities != null) {
        assertThat(actionContextProvider.getRemoteCache()).isNotNull();
        assertThat(actionContextProvider.getRemoteCache().getCacheCapabilities()).isNotNull();
        assertThat(actionContextProvider.getRemoteCache().getCacheCapabilities())
            .isEqualTo(cacheCapabilities);
        grpcCacheClient = (GrpcCacheClient) actionContextProvider.getRemoteCache().cacheProtocol;
        assertCircuitBreakerInstance(grpcCacheClient.getRetrier().getCircuitBreaker(), remoteOptions);
      } else {
        assertThat(actionContextProvider.getRemoteCache()).isNull();
      }

      if (hasRemoteExecutionCapability) {
        assertThat(actionContextProvider.getRemoteExecutionClient()).isNotNull();
        grpcRemoteExecutor = (GrpcRemoteExecutor) actionContextProvider.getRemoteExecutionClient();
        assertCircuitBreakerInstance(grpcRemoteExecutor.getRetrier().getCircuitBreaker(), remoteOptions);
      } else {
        assertThat(actionContextProvider.getRemoteExecutionClient()).isNull();
      }

    } finally {
      serverList.forEach(Server::shutdownNow);
      for (Server server : serverList) {
        server.awaitTermination();
      }
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
}
