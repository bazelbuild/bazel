// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.buildeventstream.transports.BuildEventTransportFactory.createFromOptions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/** Module responsible for configuring BuildEventStreamer and transports. */
public class BuildEventStreamerModule extends BlazeModule {

  private CommandEnvironment commandEnvironment;

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
    return ImmutableList.<Class<? extends OptionsBase>>of(BuildEventStreamOptions.class);
  }

  @Override
  public void checkEnvironment(CommandEnvironment commandEnvironment) {
    this.commandEnvironment = commandEnvironment;
    this.buildEventRecorder = new BuildEventRecorder();
    commandEnvironment.getEventBus().register(buildEventRecorder);
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    checkState(commandEnvironment != null, "Methods called out of order");
    Optional<BuildEventStreamer> maybeStreamer =
        tryCreateStreamer(optionsProvider, commandEnvironment.getBlazeModuleEnvironment());
    if (maybeStreamer.isPresent()) {
      BuildEventStreamer streamer = maybeStreamer.get();
      commandEnvironment.getReporter().addHandler(streamer);
      commandEnvironment.getEventBus().register(streamer);
      for (BuildEvent event : buildEventRecorder.getEvents()) {
        streamer.buildEvent(event);
      }
    }
    commandEnvironment.getEventBus().unregister(buildEventRecorder);
    this.buildEventRecorder = null;
  }

  @VisibleForTesting
  Optional<BuildEventStreamer> tryCreateStreamer(
      OptionsProvider optionsProvider, ModuleEnvironment moduleEnvironment) {
    try {
      PathConverter pathConverter;
      if (commandEnvironment == null) {
        pathConverter = new PathConverter() {
            @Override
            public String apply(Path path) {
              return path.getPathString();
            }
          };
      } else {
        pathConverter = commandEnvironment.getRuntime().getPathToUriConverter();
      }
      BuildEventStreamOptions besOptions =
          checkNotNull(
              optionsProvider.getOptions(BuildEventStreamOptions.class),
              "Could not get BuildEventStreamOptions");
      ImmutableSet<BuildEventTransport> buildEventTransports
          = createFromOptions(besOptions, pathConverter);
      if (!buildEventTransports.isEmpty()) {
        BuildEventStreamer streamer = new BuildEventStreamer(buildEventTransports);
        return Optional.of(streamer);
      }
    } catch (IOException e) {
      moduleEnvironment.exit(new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e));
    }
    return Optional.absent();
  }
}
