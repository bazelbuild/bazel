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
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.devtools.build.lib.buildeventservice.BuildEventServiceTransport.UPLOAD_FAILED_MESSAGE;
import static java.lang.String.format;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventStreamer;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.SynchronizedOutputStream;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Module responsible for the {@link BuildEventTransport} and its {@link BuildEventStreamer}.
 *
 * Implementors of this class have to overwrite {@link #optionsClass()} and
 * {@link #createBesClient(BuildEventServiceOptions)}.
 */
public abstract class BuildEventServiceModule<T extends BuildEventServiceOptions>
    extends BlazeModule {

  private static final Logger logger = Logger.getLogger(BuildEventServiceModule.class.getName());

  private CommandEnvironment commandEnvironment;
  private SynchronizedOutputStream out;
  private SynchronizedOutputStream err;

  private static class BuildEventRecorder {
    private final List<BuildEvent> events = new ArrayList<>();

    @Subscribe
    public void buildEvent(BuildEvent event) {
      events.add(event);
    }

    List<BuildEvent> getEvents() {
      return events;
    }
  }

  private BuildEventRecorder buildEventRecorder;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(optionsClass(), AuthAndTLSOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment commandEnvironment)
      throws AbruptExitException {
    this.commandEnvironment = commandEnvironment;
    this.buildEventRecorder = new BuildEventRecorder();
    commandEnvironment.getEventBus().register(buildEventRecorder);
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    checkState(commandEnvironment != null, "Methods called out of order");
    BuildEventStreamer streamer =
        tryCreateStreamer(
            optionsProvider,
            commandEnvironment.getReporter(),
            commandEnvironment.getBlazeModuleEnvironment(),
            commandEnvironment.getRuntime().getClock(),
            commandEnvironment.getClientEnv().get("BAZEL_INTERNAL_BUILD_REQUEST_ID"),
            commandEnvironment.getCommandId().toString());
    if (streamer != null) {
      commandEnvironment.getReporter().addHandler(streamer);
      commandEnvironment.getEventBus().register(streamer);

      final SynchronizedOutputStream theOut = this.out;
      final SynchronizedOutputStream theErr = this.err;
      // out and err should be non-null at this point, as getOutputListener is supposed to
      // be always called before handleOptions. But let's still prefer a stream with no
      // stdout/stderr over an aborted build.
      streamer.registerOutErrProvider(
          new BuildEventStreamer.OutErrProvider() {
            @Override
            public String getOut() {
              if (theOut == null) {
                return null;
              }
              return theOut.readAndReset();
            }

            @Override
            public String getErr() {
              if (theErr == null) {
                return null;
              }
              return theErr.readAndReset();
            }
          });
      if (theErr != null) {
        theErr.registerStreamer(streamer);
      }
      if (theOut != null) {
        theOut.registerStreamer(streamer);
      }
      for (BuildEvent event : buildEventRecorder.getEvents()) {
        streamer.buildEvent(event);
      }
      logger.fine("BuildEventStreamer created and registered successfully.");
    } else {
      // If there is no streamer to consume the output, we should not try to accumulate it.
      this.out.setDiscardAll();
      this.err.setDiscardAll();
    }
    commandEnvironment.getEventBus().unregister(buildEventRecorder);
    this.buildEventRecorder = null;
  }

  @Override
  public OutErr getOutputListener() {
    this.out = new SynchronizedOutputStream();
    this.err = new SynchronizedOutputStream();
    return OutErr.create(this.out, this.err);
  }

  /**
   * Returns {@code null} if no stream could be created.
   *
   * @param buildRequestId  if {@code null} or {@code ""} a random UUID is used instead.
   */
  @Nullable
  private BuildEventStreamer tryCreateStreamer(
      OptionsProvider optionsProvider,
      EventHandler commandLineReporter,
      ModuleEnvironment moduleEnvironment,
      Clock clock,
      String buildRequestId,
      String invocationId) {
    T besOptions = null;
    try {
      besOptions =
          checkNotNull(
              optionsProvider.getOptions(optionsClass()),
              "Could not get BuildEventServiceOptions.");
      AuthAndTLSOptions authTlsOptions =
          checkNotNull(optionsProvider.getOptions(AuthAndTLSOptions.class),
              "Could not get AuthAndTLSOptions.");
      if (isNullOrEmpty(besOptions.besBackend)) {
        logger.fine("BuildEventServiceTransport is disabled.");
      } else {
        logger.fine(format("Will create BuildEventServiceTransport streaming to '%s'",
            besOptions.besBackend));

        buildRequestId = isNullOrEmpty(buildRequestId)
            ? UUID.randomUUID().toString()
            : buildRequestId;
        commandLineReporter.handle(
            Event.info(
                format(
                    "Streaming Build Event Protocol to %s build_request_id: %s invocation_id: %s",
                    besOptions.besBackend, buildRequestId, invocationId)));

        BuildEventTransport besTransport =
            new BuildEventServiceTransport(
                createBesClient(besOptions, authTlsOptions),
                besOptions.besTimeout,
                besOptions.besBestEffort,
                besOptions.besLifecycleEvents,
                buildRequestId,
                invocationId,
                moduleEnvironment,
                clock,
                commandEnvironment.getRuntime().getPathToUriConverter(),
                commandLineReporter,
                besOptions.projectId);
        logger.fine("BuildEventServiceTransport was created successfully");
        return new BuildEventStreamer(ImmutableSet.of(besTransport),
            commandEnvironment.getReporter());
      }
    } catch (Exception e) {
      if (besOptions != null && besOptions.besBestEffort) {
        commandLineReporter.handle(Event.warn(format(UPLOAD_FAILED_MESSAGE, e.getMessage())));
      } else {
        commandLineReporter.handle(Event.error(format(UPLOAD_FAILED_MESSAGE, e.getMessage())));
        moduleEnvironment.exit(new AbruptExitException(ExitCode.PUBLISH_ERROR));
      }
    }
    return null;
  }

  protected abstract Class<T> optionsClass();

  protected abstract BuildEventServiceClient createBesClient(T besOptions,
      AuthAndTLSOptions authAndTLSOptions);
}