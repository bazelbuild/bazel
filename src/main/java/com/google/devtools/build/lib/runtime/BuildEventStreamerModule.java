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
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;


/** Module responsible for configuring BuildEventStreamer and transports. */
public class BuildEventStreamerModule extends BlazeModule {

  private CommandEnvironment commandEnvironment;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.<Class<? extends OptionsBase>>of(BuildEventStreamOptions.class);
  }

  @Override
  public void beforeCommand(Command command, CommandEnvironment commandEnvironment)
      throws AbruptExitException {
    this.commandEnvironment = commandEnvironment;
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    checkState(commandEnvironment != null, "Methods called out of order");
    Optional<BuildEventStreamer> streamer =
        tryCreateStreamer(optionsProvider, commandEnvironment.getBlazeModuleEnvironment());
    if (streamer.isPresent()) {
      commandEnvironment.getReporter().addHandler(streamer.get());
      commandEnvironment.getEventBus().register(streamer.get());
    }
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
