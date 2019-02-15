// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.isNullOrEmpty;
import static java.lang.String.format;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceTransport.ExitFunction;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventStreamer;
import com.google.devtools.build.lib.runtime.BuildEventTransportFactory;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CountingArtifactGroupNamer;
import com.google.devtools.build.lib.runtime.SynchronizedOutputStream;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Module responsible for the Build Event Transport (BEP) and Build Event Service (BES)
 * functionality.
 */
public abstract class BuildEventServiceModule<T extends BuildEventServiceOptions>
    extends BlazeModule {

  private static final Logger logger = Logger.getLogger(BuildEventServiceModule.class.getName());

  private OutErr outErr;
  private BuildEventStreamer streamer;
  private boolean keepClient;

  /** Whether an error in the Build Event Service upload causes the build to fail. */
  protected boolean errorsShouldFailTheBuild() {
    return true;
  }

  /** Report errors in the command line and possibly fail the build. */
  protected void reportError(
      EventHandler commandLineReporter,
      ModuleEnvironment moduleEnvironment,
      AbruptExitException exception) {
    commandLineReporter.handle(Event.error(exception.getMessage()));
    moduleEnvironment.exit(exception);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(
        optionsClass(),
        AuthAndTLSOptions.class,
        BuildEventStreamOptions.class,
        BuildEventProtocolOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment commandEnvironment) {
    T besOptions = commandEnvironment.getOptions().getOptions(optionsClass());

    // Reset to null in case afterCommand was not called.
    this.outErr = null;
    if (!whitelistedCommands(besOptions).contains(commandEnvironment.getCommandName())) {
      return;
    }

    streamer = tryCreateStreamer(commandEnvironment);
    if (streamer != null) {
      commandEnvironment.getReporter().addHandler(streamer);
      commandEnvironment.getEventBus().register(streamer);
      int bufferSize = besOptions.besOuterrBufferSize;
      int chunkSize = besOptions.besOuterrChunkSize;

      final SynchronizedOutputStream out = new SynchronizedOutputStream(bufferSize, chunkSize);
      final SynchronizedOutputStream err = new SynchronizedOutputStream(bufferSize, chunkSize);
      this.outErr = OutErr.create(out, err);
      streamer.registerOutErrProvider(
          new BuildEventStreamer.OutErrProvider() {
            @Override
            public Iterable<String> getOut() {
              return out.readAndReset();
            }

            @Override
            public Iterable<String> getErr() {
              return err.readAndReset();
            }
          });
      err.registerStreamer(streamer);
      out.registerStreamer(streamer);
      logger.fine("BuildEventStreamer created and registered successfully.");
    }
  }

  @Override
  public OutErr getOutputListener() {
    return outErr;
  }

  @Override
  public void blazeShutdownOnCrash() {
    if (streamer != null) {
      logger.warning("Attempting to close BES streamer on crash");
      streamer.close(AbortReason.INTERNAL);
    }
  }

  @Override
  public void afterCommand() {
    if (streamer != null) {
      if (!streamer.isClosed()) {
        // This should not occur, but close with an internal error if a {@link BuildEventStreamer}
        // bug manifests as an unclosed streamer.
        logger.warning("Attempting to close BES streamer after command");
        String msg = "BES was not properly closed";
        LoggingUtil.logToRemote(Level.WARNING, msg, new IllegalStateException(msg));
        streamer.close(AbortReason.INTERNAL);
      }
      this.streamer = null;
    }

    if (!keepClient) {
      clearBesClient();
    }
    this.outErr = null;
  }

  /** Returns {@code null} if no stream could be created. */
  @Nullable
  @VisibleForTesting
  BuildEventStreamer tryCreateStreamer(CommandEnvironment env) {
    BuildEventStreamOptions buildEventStreamOptions =
        env.getOptions().getOptions(BuildEventStreamOptions.class);
    this.keepClient = buildEventStreamOptions.keepBackendConnections;

    BuildEventProtocolOptions protocolOptions =
        checkNotNull(
            env.getOptions().getOptions(BuildEventProtocolOptions.class),
            "Could not get BuildEventProtocolOptions.");
    Supplier<BuildEventArtifactUploader> uploaderSupplier =
        Suppliers.memoize(
            () ->
                env.getRuntime()
                    .getBuildEventArtifactUploaderFactoryMap()
                    .select(protocolOptions.buildEventUploadStrategy)
                    .create(env));

    CountingArtifactGroupNamer namer = new CountingArtifactGroupNamer();
    try {
      BuildEventTransport besTransport;
      try {
        besTransport = tryCreateBesTransport(env, uploaderSupplier, namer);
      } catch (IOException | OptionsParsingException e) {
        String message = "Failed to create BuildEventTransport: " + e;
        logger.log(Level.WARNING, message, e);
        ExitCode exitCode =
            (e instanceof OptionsParsingException)
                ? ExitCode.COMMAND_LINE_ERROR
                : ExitCode.TRANSIENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR;
        reportError(
            env.getReporter(),
            env.getBlazeModuleEnvironment(),
            new AbruptExitException(message, exitCode, e));
        clearBesClient();
        return null;
      }

      ImmutableSet<BuildEventTransport> bepTransports =
          BuildEventTransportFactory.createFromOptions(
              env, env.getBlazeModuleEnvironment()::exit, protocolOptions, uploaderSupplier, namer);

      ImmutableSet.Builder<BuildEventTransport> transportsBuilder =
          ImmutableSet.<BuildEventTransport>builder().addAll(bepTransports);
      if (besTransport != null) {
        transportsBuilder.add(besTransport);
      }

      ImmutableSet<BuildEventTransport> transports = transportsBuilder.build();
      if (!transports.isEmpty()) {
        return new BuildEventStreamer(
            transports, env.getReporter(), buildEventStreamOptions, namer);
      }
    } catch (Exception e) {
      reportError(
          env.getReporter(),
          env.getBlazeModuleEnvironment(),
          new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e));
    }
    return null;
  }

  @Nullable
  private BuildEventTransport tryCreateBesTransport(
      CommandEnvironment env,
      Supplier<BuildEventArtifactUploader> uploaderSupplier,
      ArtifactGroupNamer namer)
      throws IOException, OptionsParsingException {
    OptionsParsingResult optionsProvider = env.getOptions();
    T besOptions =
        checkNotNull(
            optionsProvider.getOptions(optionsClass()), "Could not get BuildEventServiceOptions.");
    AuthAndTLSOptions authTlsOptions =
        checkNotNull(
            optionsProvider.getOptions(AuthAndTLSOptions.class),
            "Could not get AuthAndTLSOptions.");
    BuildEventProtocolOptions protocolOptions =
        checkNotNull(
            optionsProvider.getOptions(BuildEventProtocolOptions.class),
            "Could not get BuildEventProtocolOptions.");

    if (isNullOrEmpty(besOptions.besBackend)) {
      logger.fine("BuildEventServiceTransport is disabled.");
      clearBesClient();
      return null;
    } else {
      logger.fine(
          format(
              "Will create BuildEventServiceTransport streaming to '%s'", besOptions.besBackend));

      String invocationId = env.getCommandId().toString();
      final String besResultsUrl;
      if (!Strings.isNullOrEmpty(besOptions.besResultsUrl)) {
        besResultsUrl =
            besOptions.besResultsUrl.endsWith("/")
                ? besOptions.besResultsUrl + invocationId
                : besOptions.besResultsUrl + "/" + invocationId;
        env.getReporter().handle(Event.info("Streaming Build Event Protocol to " + besResultsUrl));
      } else {
        besResultsUrl = null;
        env.getReporter()
            .handle(
                Event.info(
                    format(
                        "Streaming Build Event Protocol to %s build_request_id: %s "
                            + "invocation_id: %s",
                        besOptions.besBackend, env.getBuildRequestId(), invocationId)));
      }

      BuildEventServiceClient client = getBesClient(besOptions, authTlsOptions);
      BuildEventArtifactUploader artifactUploader = uploaderSupplier.get();

      BuildEventServiceProtoUtil besProtoUtil =
          new BuildEventServiceProtoUtil(
              env.getBuildRequestId(),
              invocationId,
              besOptions.projectId,
              env.getCommandName(),
              keywords(besOptions, env.getRuntime().getStartupOptionsProvider()));

      BuildEventTransport besTransport =
          new BuildEventServiceTransport.Builder()
              .closeTimeout(besOptions.besTimeout)
              .publishLifecycleEvents(besOptions.besLifecycleEvents)
              .setEventBus(env.getEventBus())
              .build(
                  client,
                  artifactUploader,
                  protocolOptions,
                  besProtoUtil,
                  env.getRuntime().getClock(),
                  bazelExitFunction(
                      env.getReporter(), env.getBlazeModuleEnvironment(), besResultsUrl),
                  namer);
      logger.fine("BuildEventServiceTransport was created successfully");
      return besTransport;
    }
  }

  protected abstract Class<T> optionsClass();

  protected abstract BuildEventServiceClient getBesClient(
      T besOptions, AuthAndTLSOptions authAndTLSOptions)
      throws IOException, OptionsParsingException;

  protected abstract void clearBesClient();

  protected abstract Set<String> whitelistedCommands(T besOptions);

  protected Set<String> keywords(
      T besOptions, @Nullable OptionsParsingResult startupOptionsProvider) {
    return besOptions.besKeywords.stream()
        .map(keyword -> "user_keyword=" + keyword)
        .collect(ImmutableSet.toImmutableSet());
  }

  private ExitFunction bazelExitFunction(
      EventHandler commandLineReporter, ModuleEnvironment moduleEnvironment, String besResultsUrl) {
    return (String message, Throwable cause, ExitCode exitCode) -> {
      if (exitCode == ExitCode.SUCCESS) {
        Preconditions.checkState(cause == null, cause);
        commandLineReporter.handle(Event.info("Build Event Protocol upload finished successfully"));
        if (besResultsUrl != null) {
          commandLineReporter.handle(
              Event.info("Build Event Protocol results available at " + besResultsUrl));
        }
      } else {
        Preconditions.checkState(cause != null, cause);
        if (errorsShouldFailTheBuild()) {
          commandLineReporter.handle(Event.error(message));
          moduleEnvironment.exit(new AbruptExitException(exitCode, cause));
        } else {
          commandLineReporter.handle(Event.warn(message));
        }
        if (besResultsUrl != null) {
          if (!Strings.isNullOrEmpty(besResultsUrl)) {
            commandLineReporter.handle(
                Event.info(
                    "Partial Build Event Protocol results may be available at " + besResultsUrl));
          }
        }
      }
    };
  }
}
