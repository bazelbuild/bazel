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
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.downloader.GrpcRemoteDownloader;
import com.google.devtools.build.lib.remote.logging.LoggingInterceptor;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.NetworkTime;
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
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.Context;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
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

  private void verifyServerCapabilities(
      RemoteOptions remoteOptions,
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteRetrier retrier,
      CommandEnvironment env,
      DigestUtil digestUtil)
      throws AbruptExitException {
    RemoteServerCapabilities rsc =
        new RemoteServerCapabilities(
            remoteOptions.remoteInstanceName,
            channel,
            credentials,
            remoteOptions.remoteTimeout,
            retrier);
    ServerCapabilities capabilities = null;
    try {
      capabilities = rsc.get(env.getCommandId().toString(), env.getBuildRequestId());
    } catch (IOException e) {
      env.getReporter().handle(Event.error(Throwables.getStackTraceAsString(e)));
      throw createExitException(
          "Failed to query remote execution capabilities: " + Utils.grpcAwareErrorMessage(e),
          ExitCode.REMOTE_ERROR,
          Code.CAPABILITIES_QUERY_FAILURE);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return;
    }
    checkClientServerCompatibility(
        capabilities, remoteOptions, digestUtil.getDigestFunction(), env.getReporter());
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

    boolean enableDiskCache = RemoteCacheClientFactory.isDiskCache(remoteOptions);
    boolean enableHttpCache = RemoteCacheClientFactory.isHttpCache(remoteOptions);
    boolean enableGrpcCache = GrpcCacheClient.isRemoteCacheOptions(remoteOptions);
    boolean enableRemoteExecution = shouldEnableRemoteExecution(remoteOptions);
    boolean enableRemoteDownloader = shouldEnableRemoteDownloader(remoteOptions);

    if (enableRemoteDownloader && !enableGrpcCache) {
      throw createOptionsExitException(
          "The remote downloader can only be used in combination with gRPC caching",
          ExitCode.COMMAND_LINE_ERROR,
          FailureDetails.RemoteOptions.Code.DOWNLOADER_WITHOUT_GRPC_CACHE);
    }

    if (!enableDiskCache && !enableHttpCache && !enableGrpcCache && !enableRemoteExecution) {
      // Quit if no remote caching or execution was enabled.
      return;
    }

    if ((enableHttpCache || enableDiskCache) && enableRemoteExecution) {
      throw createOptionsExitException(
          "Cannot combine gRPC based remote execution with disk caching or HTTP-based caching",
          ExitCode.COMMAND_LINE_ERROR,
          FailureDetails.RemoteOptions.Code.EXECUTION_WITH_INVALID_CACHE);
    }

    env.getEventBus().register(this);
    String invocationId = env.getCommandId().toString();
    String buildRequestId = env.getBuildRequestId();
    env.getReporter().handle(Event.info(String.format("Invocation ID: %s", invocationId)));

    Path logDir =
        env.getOutputBase().getRelative(env.getRuntime().getProductName() + "-remote-logs");
    cleanAndCreateRemoteLogsDir(logDir);

    if ((enableHttpCache || enableDiskCache) && !enableGrpcCache) {
      Credentials creds;
      try {
        creds = GoogleAuthUtils.newCredentials(authAndTlsOptions);
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
    if (enableRemoteExecution) {
      ImmutableList.Builder<ClientInterceptor> interceptors = ImmutableList.builder();
      interceptors.add(TracingMetadataUtils.newExecHeadersInterceptor(remoteOptions));
      if (loggingInterceptor != null) {
        interceptors.add(loggingInterceptor);
      }
      interceptors.add(new NetworkTime.Interceptor());
      try {
        execChannel =
            RemoteCacheClientFactory.createGrpcChannel(
                remoteOptions.remoteExecutor,
                remoteOptions.remoteProxy,
                authAndTlsOptions,
                interceptors.build());
      } catch (IOException e) {
        handleInitFailure(env, e, Code.EXEC_CHANNEL_INIT_FAILURE);
        return;
      }
      // Create a separate channel if --remote_executor and --remote_cache point to different
      // endpoints.
      if (Strings.isNullOrEmpty(remoteOptions.remoteCache)
          || remoteOptions.remoteCache.equals(remoteOptions.remoteExecutor)) {
        cacheChannel = execChannel.retain();
      }
    }

    if (cacheChannel == null) {
      ImmutableList.Builder<ClientInterceptor> interceptors = ImmutableList.builder();
      interceptors.add(TracingMetadataUtils.newCacheHeadersInterceptor(remoteOptions));
      if (loggingInterceptor != null) {
        interceptors.add(loggingInterceptor);
      }
      interceptors.add(new NetworkTime.Interceptor());
      try {
        cacheChannel =
            RemoteCacheClientFactory.createGrpcChannel(
                remoteOptions.remoteCache,
                remoteOptions.remoteProxy,
                authAndTlsOptions,
                interceptors.build());
      } catch (IOException e) {
        handleInitFailure(env, e, Code.CACHE_CHANNEL_INIT_FAILURE);
        return;
      }
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
        try {
          downloaderChannel =
              RemoteCacheClientFactory.createGrpcChannel(
                  remoteOptions.remoteDownloader,
                  remoteOptions.remoteProxy,
                  authAndTlsOptions,
                  interceptors.build());
        } catch (IOException e) {
          handleInitFailure(env, e, Code.DOWNLOADER_CHANNEL_INIT_FAILURE);
          return;
        }
      }
    }

    CallCredentials credentials;
    try {
      credentials = GoogleAuthUtils.newCallCredentials(authAndTlsOptions);
    } catch (IOException e) {
      handleInitFailure(env, e, Code.CREDENTIALS_INIT_FAILURE);
      return;
    }
    RemoteRetrier retrier =
        new RemoteRetrier(
            remoteOptions,
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryScheduler,
            Retrier.ALLOW_ALL_CALLS);

    // We always query the execution server for capabilities, if it is defined. A remote
    // execution/cache system should have all its servers to return the capabilities pertaining
    // to the system as a whole.
    if (execChannel != null) {
      verifyServerCapabilities(remoteOptions, execChannel, credentials, retrier, env, digestUtil);
    }
    if (cacheChannel != execChannel) {
      verifyServerCapabilities(remoteOptions, cacheChannel, credentials, retrier, env, digestUtil);
    }

    ByteStreamUploader uploader =
        new ByteStreamUploader(
            remoteOptions.remoteInstanceName,
            cacheChannel.retain(),
            credentials,
            remoteOptions.remoteTimeout,
            retrier);

    cacheChannel.release();
    RemoteCacheClient cacheClient =
        new GrpcCacheClient(
            cacheChannel.retain(),
            credentials,
            remoteOptions,
            retrier,
            digestUtil,
            uploader.retain());
    uploader.release();
    Context requestContext =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, invocationId, "bes-upload");
    buildEventArtifactUploaderFactoryDelegate.init(
        new ByteStreamBuildEventArtifactUploaderFactory(
            uploader,
            cacheClient,
            cacheChannel.authority(),
            requestContext,
            remoteOptions.remoteInstanceName));

    Context repoContext =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, invocationId, "repository_rule");

    if (enableRemoteExecution) {
      RemoteRetrier execRetrier =
          new RemoteRetrier(
              remoteOptions,
              RemoteRetrier.RETRIABLE_GRPC_EXEC_ERRORS,
              retryScheduler,
              Retrier.ALLOW_ALL_CALLS);
      GrpcRemoteExecutor remoteExecutor =
          new GrpcRemoteExecutor(execChannel.retain(), credentials, execRetrier, remoteOptions);
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
              repoContext,
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
              downloaderChannel.retain(),
              Optional.ofNullable(credentials),
              retrier,
              repoContext,
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

  private static NestedSet<? extends ActionInput> getArtifactsToBuild(
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
      Iterable<ConfiguredTarget> configuredTargets) {
    if (remoteOutputsMode != null && remoteOutputsMode.downloadToplevelOutputsOnly()) {
      Preconditions.checkState(actionContextProvider != null, "actionContextProvider was null");
      boolean isTestCommand = env.getCommandName().equals("test");
      TopLevelArtifactContext artifactContext = request.getTopLevelArtifactContext();
      ImmutableSet.Builder<ActionInput> filesToDownload = ImmutableSet.builder();
      for (ConfiguredTarget configuredTarget : configuredTargets) {
        if (isTestCommand && isTestRule(configuredTarget)) {
          // When running a test download the test.log and test.xml.
          filesToDownload.addAll(getTestOutputs(configuredTarget));
        } else {
          filesToDownload.addAll(getArtifactsToBuild(configuredTarget, artifactContext).toList());
          filesToDownload.addAll(getRunfiles(configuredTarget));
        }
      }
      actionContextProvider.setFilesToDownload(filesToDownload.build());
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

  private void checkClientServerCompatibility(
      ServerCapabilities capabilities,
      RemoteOptions remoteOptions,
      DigestFunction.Value digestFunction,
      Reporter reporter)
      throws AbruptExitException {
    RemoteServerCapabilities.ClientServerCompatibilityStatus st =
        RemoteServerCapabilities.checkClientServerCompatibility(
            capabilities, remoteOptions, digestFunction);
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

    try {
      deleteDownloadedInputs();
    } catch (IOException e) {
      failure = e;
      failureCode = Code.DOWNLOADED_INPUTS_DELETION_FAILURE;
      failureMessage = "Failed to delete downloaded inputs";
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

  /**
   * Delete any input files that have been fetched from the remote cache during the build. This is
   * so that Bazel's view of the output base is identical with the output base after a build i.e.
   * files that Bazel thinks exist only remotely actually do.
   */
  private void deleteDownloadedInputs() throws IOException {
    if (actionInputFetcher == null) {
      return;
    }
    IOException deletionFailure = null;
    for (Path file : actionInputFetcher.downloadedFiles()) {
      try {
        file.delete();
      } catch (IOException e) {
        logger.atSevere().withCause(e).log(
            "Failed to delete remote output '%s' from the output base.", file);
        deletionFailure = e;
      }
    }
    if (deletionFailure != null) {
      throw deletionFailure;
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
      RequestMetadata requestMetadata =
          RequestMetadata.newBuilder()
              .setCorrelatedInvocationsId(env.getBuildRequestId())
              .setToolInvocationId(env.getCommandId().toString())
              .build();
      actionInputFetcher =
          new RemoteActionInputFetcher(
              actionContextProvider.getRemoteCache(), env.getExecRoot(), requestMetadata);
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
      String message, ExitCode exitCode, FailureDetails.RemoteOptions.Code remoteExecutionCode) {
    return new AbruptExitException(
        DetailedExitCode.of(
            exitCode,
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
}
