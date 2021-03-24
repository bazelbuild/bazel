// Copyright 2016 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.auth.Credentials;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.authandtls.Netrc;
import com.google.devtools.build.lib.authandtls.NetrcCredentials;
import com.google.devtools.build.lib.authandtls.NetrcParser;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.remote.RemoteServerCapabilities.ServerCapabilitiesRequirement;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.downloader.GrpcRemoteDownloader;
import com.google.devtools.build.lib.remote.logging.LoggingInterceptor;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutorFactory;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution;
import com.google.devtools.build.lib.server.FailureDetails.RemoteExecution.Code;
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.reactivex.rxjava3.plugins.RxJavaPlugins;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Executors;

/** RemoteModule provides distributed cache and remote execution for Bazel. */
public final class RemoteModule extends BlazeModule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private AsynchronousFileOutputStream rpcLogFile;

  private final ListeningScheduledExecutorService retryScheduler =
      MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

  private RemoteActionContextProvider actionContextProvider;
  private RemoteActionInputFetcher actionInputFetcher;
  private RemoteOutputsMode remoteOutputsMode;
  private RemoteOutputService remoteOutputService;

  private ChannelFactory channelFactory =
      new ChannelFactory() {
        @Override
        public ManagedChannel newChannel(
            String target,
            String proxy,
            AuthAndTLSOptions options,
            List<ClientInterceptor> interceptors)
            throws IOException {
          return GoogleAuthUtils.newChannel(
              target, proxy, options, interceptors.isEmpty() ? null : interceptors);
        }
      };

  private final BuildEventArtifactUploaderFactoryDelegate
      buildEventArtifactUploaderFactoryDelegate = new BuildEventArtifactUploaderFactoryDelegate();

  private final RepositoryRemoteExecutorFactoryDelegate repositoryRemoteExecutorFactoryDelegate =
      new RepositoryRemoteExecutorFactoryDelegate();

  private final MutableSupplier<Downloader> remoteDownloaderSupplier = new MutableSupplier<>();

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder) {
    builder.addBuildEventArtifactUploaderFactory(
        buildEventArtifactUploaderFactoryDelegate, "remote");
    builder.setRepositoryRemoteExecutorFactory(repositoryRemoteExecutorFactoryDelegate);
    builder.setDownloaderSupplier(remoteDownloaderSupplier);
  }

  /** Returns whether remote execution should be available. */
  public static boolean shouldEnableRemoteExecution(RemoteOptions options) {
    return !Strings.isNullOrEmpty(options.remoteExecutor);
  }

  /** Returns whether remote downloading should be available. */
  private static boolean shouldEnableRemoteDownloader(RemoteOptions options) {
    return !Strings.isNullOrEmpty(options.remoteDownloader);
  }

  private static void verifyServerCapabilities(
      RemoteOptions remoteOptions,
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteRetrier retrier,
      CommandEnvironment env,
      DigestUtil digestUtil,
      ServerCapabilitiesRequirement requirement)
      throws AbruptExitException, IOException {
    RemoteServerCapabilities rsc =
        new RemoteServerCapabilities(
            remoteOptions.remoteInstanceName,
            channel,
            credentials,
            remoteOptions.remoteTimeout.getSeconds(),
            retrier);
    ServerCapabilities capabilities = null;
    try {
      capabilities = rsc.get(env.getBuildRequestId(), env.getCommandId().toString());
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return;
    }
    checkClientServerCompatibility(
        capabilities,
        remoteOptions,
        digestUtil.getDigestFunction(),
        env.getReporter(),
        requirement);
  }

  private void initHttpAndDiskCache(
      CommandEnvironment env,
      AuthAndTLSOptions authAndTlsOptions,
      RemoteOptions remoteOptions,
      DigestUtil digestUtil) {
    Credentials creds;
    try {
      creds =
          newCredentials(
              env.getClientEnv(),
              env.getRuntime().getFileSystem(),
              env.getReporter(),
              authAndTlsOptions,
              remoteOptions);
    } catch (IOException e) {
      handleInitFailure(env, e, Code.CREDENTIALS_INIT_FAILURE);
      return;
    }
    RemoteCacheClient cacheClient;
    try {
      cacheClient =
          RemoteCacheClientFactory.create(
              remoteOptions,
              creds,
              Preconditions.checkNotNull(env.getWorkingDirectory(), "workingDirectory"),
              digestUtil);
    } catch (IOException e) {
      handleInitFailure(env, e, Code.CACHE_INIT_FAILURE);
      return;
    }
    RemoteCache remoteCache = new RemoteCache(cacheClient, remoteOptions, digestUtil);
    actionContextProvider =
        RemoteActionContextProvider.createForRemoteCaching(
            env, remoteCache, /* retryScheduler= */ null, digestUtil);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    Preconditions.checkState(actionContextProvider == null, "actionContextProvider must be null");
    Preconditions.checkState(actionInputFetcher == null, "actionInputFetcher must be null");
    Preconditions.checkState(remoteOutputsMode == null, "remoteOutputsMode must be null");

    RemoteOptions remoteOptions = env.getOptions().getOptions(RemoteOptions.class);
    if (remoteOptions == null) {
      // Quit if no supported command is being used. See getCommandOptions for details.
      return;
    }

    remoteOutputsMode = remoteOptions.remoteOutputsMode;

    AuthAndTLSOptions authAndTlsOptions = env.getOptions().getOptions(AuthAndTLSOptions.class);
    DigestHashFunction hashFn = env.getRuntime().getFileSystem().getDigestFunction();
    DigestUtil digestUtil = new DigestUtil(hashFn);

    boolean verboseFailures = false;
    ExecutionOptions executionOptions = env.getOptions().getOptions(ExecutionOptions.class);
    if (executionOptions != null) {
      verboseFailures = executionOptions.verboseFailures;
    }

    boolean enableDiskCache = RemoteCacheClientFactory.isDiskCache(remoteOptions);
    boolean enableHttpCache = RemoteCacheClientFactory.isHttpCache(remoteOptions);
    boolean enableRemoteExecution = shouldEnableRemoteExecution(remoteOptions);
    // If --remote_cache is empty but --remote_executor is not, endpoint for cache should be the one
    // for execution.
    if (enableRemoteExecution && Strings.isNullOrEmpty(remoteOptions.remoteCache)) {
      remoteOptions.remoteCache = remoteOptions.remoteExecutor;
    }
    boolean enableGrpcCache = GrpcCacheClient.isRemoteCacheOptions(remoteOptions);
    boolean enableRemoteDownloader = shouldEnableRemoteDownloader(remoteOptions);

    if (enableRemoteDownloader && !enableGrpcCache) {
      throw createOptionsExitException(
          "The remote downloader can only be used in combination with gRPC caching",
          FailureDetails.RemoteOptions.Code.DOWNLOADER_WITHOUT_GRPC_CACHE);
    }

    if (!enableDiskCache && !enableHttpCache && !enableGrpcCache && !enableRemoteExecution) {
      // Quit if no remote caching or execution was enabled.
      return;
    }

    if ((enableHttpCache || enableDiskCache) && enableRemoteExecution) {
      throw createOptionsExitException(
          "Cannot combine gRPC based remote execution with disk caching or HTTP-based caching",
          FailureDetails.RemoteOptions.Code.EXECUTION_WITH_INVALID_CACHE);
    }

    env.getEventBus().register(this);
    String invocationId = env.getCommandId().toString();
    String buildRequestId = env.getBuildRequestId();
    env.getReporter().handle(Event.info(String.format("Invocation ID: %s", invocationId)));

    RxJavaPlugins.setErrorHandler(
        error -> env.getReporter().handle(Event.error(Throwables.getStackTraceAsString(error))));

    Path logDir =
        env.getOutputBase().getRelative(env.getRuntime().getProductName() + "-remote-logs");
    cleanAndCreateRemoteLogsDir(logDir);

    if ((enableHttpCache || enableDiskCache) && !enableGrpcCache) {
      initHttpAndDiskCache(env, authAndTlsOptions, remoteOptions, digestUtil);
      return;
    }

    ClientInterceptor loggingInterceptor = null;
    if (remoteOptions.experimentalRemoteGrpcLog != null) {
      try {
        rpcLogFile =
            new AsynchronousFileOutputStream(
                env.getWorkingDirectory().getRelative(remoteOptions.experimentalRemoteGrpcLog));
      } catch (IOException e) {
        handleInitFailure(env, e, Code.RPC_LOG_FAILURE);
        return;
      }
      loggingInterceptor = new LoggingInterceptor(rpcLogFile, env.getRuntime().getClock());
    }

    ReferenceCountedChannel execChannel = null;
    ReferenceCountedChannel cacheChannel = null;
    ReferenceCountedChannel downloaderChannel = null;

    // The number of concurrent requests for one connection to a gRPC server is limited by
    // MAX_CONCURRENT_STREAMS which is normally being 100+. We assume 50 concurrent requests for
    // each connection should be fairly well. The number of connections opened by one channel is
    // based on the resolved IPs of that server. We assume servers normally have 2 IPs. So the
    // max concurrency per connection is 100.
    int maxConcurrencyPerConnection = 100;

    if (enableRemoteExecution) {
      ImmutableList.Builder<ClientInterceptor> interceptors = ImmutableList.builder();
      interceptors.add(TracingMetadataUtils.newExecHeadersInterceptor(remoteOptions));
      if (loggingInterceptor != null) {
        interceptors.add(loggingInterceptor);
      }
      execChannel =
          new ReferenceCountedChannel(
              new GoogleChannelConnectionFactory(
                  channelFactory,
                  remoteOptions.remoteExecutor,
                  remoteOptions.remoteProxy,
                  authAndTlsOptions,
                  interceptors.build(),
                  maxConcurrencyPerConnection));

      // Create a separate channel if --remote_executor and --remote_cache point to different
      // endpoints.
      if (remoteOptions.remoteCache.equals(remoteOptions.remoteExecutor)) {
        cacheChannel = execChannel.retain();
      }
    }

    if (cacheChannel == null) {
      ImmutableList.Builder<ClientInterceptor> interceptors = ImmutableList.builder();
      interceptors.add(TracingMetadataUtils.newCacheHeadersInterceptor(remoteOptions));
      if (loggingInterceptor != null) {
        interceptors.add(loggingInterceptor);
      }
      cacheChannel =
          new ReferenceCountedChannel(
              new GoogleChannelConnectionFactory(
                  channelFactory,
                  remoteOptions.remoteCache,
                  remoteOptions.remoteProxy,
                  authAndTlsOptions,
                  interceptors.build(),
                  maxConcurrencyPerConnection));
    }

    if (enableRemoteDownloader) {
      // Create a separate channel if --remote_downloader and --remote_cache point to different
      // endpoints.
      if (remoteOptions.remoteDownloader.equals(remoteOptions.remoteCache)) {
        downloaderChannel = cacheChannel.retain();
      } else {
        ImmutableList.Builder<ClientInterceptor> interceptors = ImmutableList.builder();
        if (loggingInterceptor != null) {
          interceptors.add(loggingInterceptor);
        }
        downloaderChannel =
            new ReferenceCountedChannel(
                new GoogleChannelConnectionFactory(
                    channelFactory,
                    remoteOptions.remoteDownloader,
                    remoteOptions.remoteProxy,
                    authAndTlsOptions,
                    interceptors.build(),
                    maxConcurrencyPerConnection));
      }
    }

    CallCredentialsProvider callCredentialsProvider;
    try {
      callCredentialsProvider =
          GoogleAuthUtils.newCallCredentialsProvider(
              newCredentials(
                  env.getClientEnv(),
                  env.getRuntime().getFileSystem(),
                  env.getReporter(),
                  authAndTlsOptions,
                  remoteOptions));
    } catch (IOException e) {
      handleInitFailure(env, e, Code.CREDENTIALS_INIT_FAILURE);
      return;
    }

    CallCredentials credentials = callCredentialsProvider.getCallCredentials();

    RemoteRetrier retrier =
        new RemoteRetrier(
            remoteOptions,
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryScheduler,
            Retrier.ALLOW_ALL_CALLS);

    // We only check required capabilities for a given endpoint.
    //
    // If --remote_executor and --remote_cache point to the same endpoint, we require that
    // endpoint has both execution and cache capabilities.
    //
    // If they point to different endpoints, we check the endpoint with execution or cache
    // capabilities respectively.
    try {
      if (execChannel != null) {
        if (cacheChannel != execChannel) {
          verifyServerCapabilities(
              remoteOptions,
              execChannel,
              credentials,
              retrier,
              env,
              digestUtil,
              ServerCapabilitiesRequirement.EXECUTION);
          verifyServerCapabilities(
              remoteOptions,
              cacheChannel,
              credentials,
              retrier,
              env,
              digestUtil,
              ServerCapabilitiesRequirement.CACHE);
        } else {
          verifyServerCapabilities(
              remoteOptions,
              execChannel,
              credentials,
              retrier,
              env,
              digestUtil,
              ServerCapabilitiesRequirement.EXECUTION_AND_CACHE);
        }
      } else {
        verifyServerCapabilities(
            remoteOptions,
            cacheChannel,
            credentials,
            retrier,
            env,
            digestUtil,
            ServerCapabilitiesRequirement.CACHE);
      }
    } catch (IOException e) {
      String errorMessage =
          "Failed to query remote execution capabilities: " + Utils.grpcAwareErrorMessage(e);
      if (remoteOptions.remoteLocalFallback) {
        if (verboseFailures) {
          errorMessage += System.lineSeparator() + Throwables.getStackTraceAsString(e);
        }
        env.getReporter().handle(Event.warn(errorMessage));
        return;
      } else {
        if (verboseFailures) {
          env.getReporter().handle(Event.error(Throwables.getStackTraceAsString(e)));
        }
        throw createExitException(
            errorMessage, ExitCode.REMOTE_ERROR, Code.CAPABILITIES_QUERY_FAILURE);
      }
    }

    String remoteBytestreamUriPrefix = remoteOptions.remoteBytestreamUriPrefix;
    if (Strings.isNullOrEmpty(remoteBytestreamUriPrefix)) {
      remoteBytestreamUriPrefix = cacheChannel.authority();
      if (!Strings.isNullOrEmpty(remoteOptions.remoteInstanceName)) {
        remoteBytestreamUriPrefix += "/" + remoteOptions.remoteInstanceName;
      }
    }

    ByteStreamUploader uploader =
        new ByteStreamUploader(
            remoteOptions.remoteInstanceName,
            cacheChannel.retain(),
            callCredentialsProvider,
            remoteOptions.remoteTimeout.getSeconds(),
            retrier);

    cacheChannel.release();
    RemoteCacheClient cacheClient =
        new GrpcCacheClient(
            cacheChannel.retain(),
            callCredentialsProvider,
            remoteOptions,
            retrier,
            digestUtil,
            uploader.retain());
    uploader.release();
    buildEventArtifactUploaderFactoryDelegate.init(
        new ByteStreamBuildEventArtifactUploaderFactory(
            uploader, cacheClient, remoteBytestreamUriPrefix, buildRequestId, invocationId));

    if (enableRemoteExecution) {
      RemoteExecutionClient remoteExecutor;
      if (remoteOptions.remoteExecutionKeepalive) {
        RemoteRetrier execRetrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_ERRORS, // Handle NOT_FOUND internally
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        remoteExecutor =
            new ExperimentalGrpcRemoteExecutor(
                remoteOptions, execChannel.retain(), callCredentialsProvider, execRetrier);
      } else {
        RemoteRetrier execRetrier =
            new RemoteRetrier(
                remoteOptions,
                RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
                retryScheduler,
                Retrier.ALLOW_ALL_CALLS);
        remoteExecutor =
            new GrpcRemoteExecutor(execChannel.retain(), callCredentialsProvider, execRetrier);
      }
      execChannel.release();
      RemoteExecutionCache remoteCache =
          new RemoteExecutionCache(cacheClient, remoteOptions, digestUtil);
      actionContextProvider =
          RemoteActionContextProvider.createForRemoteExecution(
              env, remoteCache, remoteExecutor, retryScheduler, digestUtil, logDir);
      repositoryRemoteExecutorFactoryDelegate.init(
          new RemoteRepositoryRemoteExecutorFactory(
              remoteCache,
              remoteExecutor,
              digestUtil,
              buildRequestId,
              invocationId,
              remoteOptions.remoteInstanceName,
              remoteOptions.remoteAcceptCached));
    } else {
      if (enableDiskCache) {
        try {
          cacheClient =
              RemoteCacheClientFactory.createDiskAndRemoteClient(
                  env.getWorkingDirectory(),
                  remoteOptions.diskCache,
                  remoteOptions.remoteVerifyDownloads,
                  digestUtil,
                  cacheClient,
                  remoteOptions);
        } catch (IOException e) {
          handleInitFailure(env, e, Code.CACHE_INIT_FAILURE);
          return;
        }
      }

      RemoteCache remoteCache = new RemoteCache(cacheClient, remoteOptions, digestUtil);
      actionContextProvider =
          RemoteActionContextProvider.createForRemoteCaching(
              env, remoteCache, retryScheduler, digestUtil);
    }

    if (enableRemoteDownloader) {
      remoteDownloaderSupplier.set(
          new GrpcRemoteDownloader(
              buildRequestId,
              invocationId,
              downloaderChannel.retain(),
              Optional.ofNullable(credentials),
              retrier,
              cacheClient,
              remoteOptions));
      downloaderChannel.release();
    }
  }

  private static void handleInitFailure(
      CommandEnvironment env, IOException e, Code remoteExecutionCode) {
    env.getReporter().handle(Event.error(e.getMessage()));
    env.getBlazeModuleEnvironment()
        .exit(
            createExitException(
                "Error initializing RemoteModule",
                ExitCode.COMMAND_LINE_ERROR,
                remoteExecutionCode));
  }

  private static ImmutableList<Artifact> getRunfiles(ConfiguredTarget buildTarget) {
    FilesToRunProvider runfilesProvider = buildTarget.getProvider(FilesToRunProvider.class);
    if (runfilesProvider == null) {
      return ImmutableList.of();
    }
    RunfilesSupport runfilesSupport = runfilesProvider.getRunfilesSupport();
    if (runfilesSupport == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Artifact> runfilesBuilder = ImmutableList.builder();
    for (Artifact runfile : runfilesSupport.getRunfiles().getArtifacts().toList()) {
      if (runfile.isSourceArtifact()) {
        continue;
      }
      runfilesBuilder.add(runfile);
    }
    return runfilesBuilder.build();
  }

  private static ImmutableList<ActionInput> getTestOutputs(ConfiguredTarget testTarget) {
    TestProvider testProvider = testTarget.getProvider(TestProvider.class);
    if (testProvider == null) {
      return ImmutableList.of();
    }
    return testProvider.getTestParams().getOutputs();
  }

  private static NestedSet<Artifact> getArtifactsToBuild(
      ConfiguredTarget buildTarget, TopLevelArtifactContext topLevelArtifactContext) {
    return TopLevelArtifactHelper.getAllArtifactsToBuild(buildTarget, topLevelArtifactContext)
        .getImportantArtifacts();
  }

  private static boolean isTestRule(ConfiguredTarget configuredTarget) {
    if (configuredTarget instanceof RuleConfiguredTarget) {
      RuleConfiguredTarget ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;
      return TargetUtils.isTestRuleName(ruleConfiguredTarget.getRuleClassString());
    }
    return false;
  }

  @Override
  public void afterAnalysis(
      CommandEnvironment env,
      BuildRequest request,
      BuildOptions buildOptions,
      AnalysisResult analysisResult) {
    // The actionContextProvider may be null if remote execution is disabled or if there was an
    // error during initialization.
    if (remoteOutputsMode != null
        && remoteOutputsMode.downloadToplevelOutputsOnly()
        && actionContextProvider != null) {
      boolean isTestCommand = env.getCommandName().equals("test");
      TopLevelArtifactContext artifactContext = request.getTopLevelArtifactContext();
      Set<ActionInput> filesToDownload = new HashSet<>();
      for (ConfiguredTarget configuredTarget : analysisResult.getTargetsToBuild()) {
        if (isTestCommand && isTestRule(configuredTarget)) {
          // When running a test download the test.log and test.xml. These are never symlinks.
          filesToDownload.addAll(getTestOutputs(configuredTarget));
        } else {
          fetchSymlinkDependenciesRecursively(
              analysisResult.getActionGraph(),
              filesToDownload,
              getArtifactsToBuild(configuredTarget, artifactContext).toList());
          fetchSymlinkDependenciesRecursively(
              analysisResult.getActionGraph(), filesToDownload, getRunfiles(configuredTarget));
        }
      }
      actionContextProvider.setFilesToDownload(ImmutableSet.copyOf(filesToDownload));
    }
  }

  // This is a short-term fix for top-level outputs that are symlinks. Unfortunately, we cannot
  // reliably tell after analysis whether actions will create symlinks (the RE protocol allows any
  // action to generate and return symlinks), but at least we can handle basic C++ rules with this
  // change.
  // TODO(ulfjack): I think we should separate downloading files from action execution. That would
  // also resolve issues around action invalidation - we currently invalidate actions to trigger
  // downloads of top-level outputs when the top-level targets change.
  private static void fetchSymlinkDependenciesRecursively(
      ActionGraph actionGraph, Set<ActionInput> builder, List<Artifact> inputs) {
    for (Artifact input : inputs) {
      // Only fetch recursively if we don't have the file to avoid visiting nodes multiple times.
      if (builder.add(input)) {
        fetchSymlinkDependenciesRecursively(actionGraph, builder, input);
      }
    }
  }

  private static void fetchSymlinkDependenciesRecursively(
      ActionGraph actionGraph, Set<ActionInput> builder, Artifact artifact) {
    if (!(actionGraph.getGeneratingAction(artifact) instanceof ActionExecutionMetadata)) {
      // The top-level artifact could be a tree artifact, in which case the generating action may
      // be an ActionTemplate, which does not implement ActionExecutionMetadata. We don't handle
      // this case right now, so exit.
      return;
    }
    ActionExecutionMetadata action =
        (ActionExecutionMetadata) actionGraph.getGeneratingAction(artifact);
    if (action.mayInsensitivelyPropagateInputs()) {
      List<Artifact> inputs = action.getInputs().toList();
      if (inputs.size() > 5) {
        logger.atWarning().log(
            "Action with a lot of inputs insensitively propagates them; this could be performance"
                + " problem");
      }
      fetchSymlinkDependenciesRecursively(actionGraph, builder, inputs);
    }
  }

  private static void cleanAndCreateRemoteLogsDir(Path logDir) throws AbruptExitException {
    try {
      // Clean out old logs files.
      if (logDir.exists()) {
        logDir.deleteTree();
      }
      logDir.createDirectory();
    } catch (IOException e) {
      String message = String.format("Could not create base directory for remote logs: %s", logDir);
      throw createExitException(
          message, ExitCode.LOCAL_ENVIRONMENTAL_ERROR, Code.LOG_DIR_CLEANUP_FAILURE);
    }
  }

  private static void checkClientServerCompatibility(
      ServerCapabilities capabilities,
      RemoteOptions remoteOptions,
      DigestFunction.Value digestFunction,
      Reporter reporter,
      ServerCapabilitiesRequirement requirement)
      throws AbruptExitException {
    RemoteServerCapabilities.ClientServerCompatibilityStatus st =
        RemoteServerCapabilities.checkClientServerCompatibility(
            capabilities, remoteOptions, digestFunction, requirement);
    for (String warning : st.getWarnings()) {
      reporter.handle(Event.warn(warning));
    }
    List<String> errors = st.getErrors();
    for (int i = 0; i < errors.size() - 1; ++i) {
      reporter.handle(Event.error(errors.get(i)));
    }
    if (!errors.isEmpty()) {
      String lastErrorMessage = errors.get(errors.size() - 1);
      throw createExitException(
          lastErrorMessage, ExitCode.REMOTE_ERROR, Code.CLIENT_SERVER_INCOMPATIBLE);
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    IOException failure = null;
    Code failureCode = null;
    String failureMessage = null;

    try {
      closeRpcLogFile();
    } catch (IOException e) {
      failure = e;
      failureCode = Code.RPC_LOG_FAILURE;
      failureMessage = "Partially wrote rpc log file";
      logger.atWarning().withCause(e).log(failureMessage);
    }

    buildEventArtifactUploaderFactoryDelegate.reset();
    repositoryRemoteExecutorFactoryDelegate.reset();
    remoteDownloaderSupplier.set(null);
    actionContextProvider = null;
    actionInputFetcher = null;
    remoteOutputsMode = null;
    remoteOutputService = null;

    if (failure != null) {
      throw createExitException(failureMessage, ExitCode.LOCAL_ENVIRONMENTAL_ERROR, failureCode);
    }
  }

  private void closeRpcLogFile() throws IOException {
    if (rpcLogFile != null) {
      AsynchronousFileOutputStream oldLogFile = rpcLogFile;
      rpcLogFile = null;
      oldLogFile.close();
    }
  }

  @Override
  public void registerSpawnStrategies(
      SpawnStrategyRegistry.Builder registryBuilder, CommandEnvironment env) {
    if (actionContextProvider == null) {
      return;
    }
    RemoteOptions remoteOptions =
        Preconditions.checkNotNull(
            env.getOptions().getOptions(RemoteOptions.class), "RemoteOptions");
    registryBuilder.setRemoteLocalFallbackStrategyIdentifier(
        remoteOptions.remoteLocalFallbackStrategy);
    actionContextProvider.registerRemoteSpawnStrategyIfApplicable(registryBuilder);
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    if (actionContextProvider == null) {
      return;
    }
    actionContextProvider.registerSpawnCache(registryBuilder);
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    Preconditions.checkState(actionInputFetcher == null, "actionInputFetcher must be null");
    Preconditions.checkNotNull(remoteOutputsMode, "remoteOutputsMode must not be null");

    if (actionContextProvider == null) {
      return;
    }
    builder.addExecutorLifecycleListener(actionContextProvider);
    RemoteOptions remoteOptions =
        Preconditions.checkNotNull(
            env.getOptions().getOptions(RemoteOptions.class), "RemoteOptions");
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    if (!remoteOutputsMode.downloadAllOutputs()) {
      actionInputFetcher =
          new RemoteActionInputFetcher(
              env.getBuildRequestId(),
              env.getCommandId().toString(),
              actionContextProvider.getRemoteCache(),
              env.getExecRoot());
      builder.setActionInputPrefetcher(actionInputFetcher);
      remoteOutputService.setActionInputFetcher(actionInputFetcher);
    }
  }

  @Override
  public OutputService getOutputService() {
    Preconditions.checkState(remoteOutputService == null, "remoteOutputService must be null");
    if (remoteOutputsMode != null && !remoteOutputsMode.downloadAllOutputs()) {
      remoteOutputService = new RemoteOutputService();
    }
    return remoteOutputService;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of("build", "fetch", "query", "sync", "test").contains(command.name())
        ? ImmutableList.of(RemoteOptions.class, AuthAndTLSOptions.class)
        : ImmutableList.of();
  }

  private static class BuildEventArtifactUploaderFactoryDelegate
      implements BuildEventArtifactUploaderFactory {

    private volatile BuildEventArtifactUploaderFactory uploaderFactory;

    public void init(BuildEventArtifactUploaderFactory uploaderFactory) {
      Preconditions.checkState(this.uploaderFactory == null);
      this.uploaderFactory = uploaderFactory;
    }

    public void reset() {
      this.uploaderFactory = null;
    }

    @Override
    public BuildEventArtifactUploader create(CommandEnvironment env)
        throws InvalidPackagePathSymlinkException {
      BuildEventArtifactUploaderFactory uploaderFactory0 = this.uploaderFactory;
      if (uploaderFactory0 == null) {
        return new LocalFilesArtifactUploader();
      }
      return uploaderFactory0.create(env);
    }
  }

  private static AbruptExitException createOptionsExitException(
      String message, FailureDetails.RemoteOptions.Code remoteExecutionCode) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setRemoteOptions(
                    FailureDetails.RemoteOptions.newBuilder().setCode(remoteExecutionCode))
                .build()));
  }

  private static AbruptExitException createExitException(
      String message, ExitCode exitCode, Code remoteExecutionCode) {
    return new AbruptExitException(
        DetailedExitCode.of(
            exitCode,
            FailureDetail.newBuilder()
                .setMessage(message)
                .setRemoteExecution(RemoteExecution.newBuilder().setCode(remoteExecutionCode))
                .build()));
  }

  private static class RepositoryRemoteExecutorFactoryDelegate
      implements RepositoryRemoteExecutorFactory {

    private volatile RepositoryRemoteExecutorFactory delegate;

    public void init(RepositoryRemoteExecutorFactory delegate) {
      Preconditions.checkState(this.delegate == null);
      this.delegate = delegate;
    }

    public void reset() {
      this.delegate = null;
    }

    @Override
    public RepositoryRemoteExecutor create() {
      RepositoryRemoteExecutorFactory delegate = this.delegate;
      if (delegate == null) {
        return null;
      }
      return delegate.create();
    }
  }

  @VisibleForTesting
  void setChannelFactory(ChannelFactory channelFactory) {
    this.channelFactory = channelFactory;
  }

  @VisibleForTesting
  RemoteActionContextProvider getActionContextProvider() {
    return actionContextProvider;
  }

  /**
   * Create a new {@link Credentials} object by parsing the .netrc file with following order to
   * search it:
   *
   * <ol>
   *   <li>If environment variable $NETRC exists, use it as the path to the .netrc file
   *   <li>Fallback to $HOME/.netrc
   * </ol>
   *
   * @return the {@link Credentials} object or {@code null} if there is no .netrc file.
   * @throws IOException in case the credentials can't be constructed.
   */
  @VisibleForTesting
  static Credentials newCredentialsFromNetrc(Map<String, String> clientEnv, FileSystem fileSystem)
      throws IOException {
    String netrcFileString =
        Optional.ofNullable(clientEnv.get("NETRC"))
            .orElseGet(
                () ->
                    Optional.ofNullable(clientEnv.get("HOME"))
                        .map(home -> home + "/.netrc")
                        .orElse(null));
    if (netrcFileString == null) {
      return null;
    }

    Path netrcFile = fileSystem.getPath(netrcFileString);
    if (netrcFile.exists()) {
      try {
        Netrc netrc = NetrcParser.parseAndClose(netrcFile.getInputStream());
        return new NetrcCredentials(netrc);
      } catch (IOException e) {
        throw new IOException(
            "Failed to parse " + netrcFile.getPathString() + ": " + e.getMessage(), e);
      }
    } else {
      return null;
    }
  }

  /**
   * Create a new {@link Credentials} with following order:
   *
   * <ol>
   *   <li>If authentication enabled by flags, use it to create credentials
   *   <li>Use .netrc to provide credentials if exists
   *   <li>Otherwise, return {@code null}
   * </ol>
   *
   * @throws IOException in case the credentials can't be constructed.
   */
  @VisibleForTesting
  static Credentials newCredentials(
      Map<String, String> clientEnv,
      FileSystem fileSystem,
      Reporter reporter,
      AuthAndTLSOptions authAndTlsOptions,
      RemoteOptions remoteOptions)
      throws IOException {
    Credentials creds = GoogleAuthUtils.newCredentials(authAndTlsOptions);

    // Fallback to .netrc if it exists
    if (creds == null) {
      try {
        creds = newCredentialsFromNetrc(clientEnv, fileSystem);
      } catch (IOException e) {
        reporter.handle(Event.warn(e.getMessage()));
      }

      try {
        if (creds != null
            && remoteOptions.remoteCache != null
            && Ascii.toLowerCase(remoteOptions.remoteCache).startsWith("http://")
            && !creds.getRequestMetadata(new URI(remoteOptions.remoteCache)).isEmpty()) {
          reporter.handle(
              Event.warn(
                  "Username and password from .netrc is transmitted in plaintext to "
                      + remoteOptions.remoteCache
                      + ". Please consider using an HTTPS endpoint."));
        }
      } catch (URISyntaxException e) {
        throw new IOException(e.getMessage(), e);
      }
    }

    return creds;
  }
}
