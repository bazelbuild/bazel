// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.common.io.Files;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ResolvedEvent;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.common.options.OptionsBase;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/** Module providing the collection of the resolved values for the repository rules executed. */
public final class RepositoryResolvedModule extends BlazeModule {
  public static final String EXPORTED_NAME = "resolved";

  private static final Logger logger = Logger.getLogger(RepositoryResolvedModule.class.getName());
  private Map<String, Object> resolvedValues;
  private String resolvedFile;
  private ImmutableList<String> orderedNames;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableSet.of("sync", "fetch", "build", "query").contains(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(RepositoryResolvedOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    orderedNames = ImmutableList.<String>of();
    RepositoryResolvedOptions options =
        env.getOptions().getOptions(RepositoryResolvedOptions.class);
    if (options != null && !Strings.isNullOrEmpty(options.repositoryResolvedFile)) {
      this.resolvedFile = options.repositoryResolvedFile;
      env.getEventBus().register(this);
      this.resolvedValues = new LinkedHashMap<String, Object>();
    } else {
      this.resolvedFile = null;
    }
  }

  @Override
  public void afterCommand() {
    if (resolvedFile != null) {
      ImmutableList.Builder<Object> resultBuilder = new ImmutableList.Builder<>();
      // Fill the result builder; first all known repositories in order, then the
      // rest in the order we knew about them.
      for (String name : orderedNames) {
        if (resolvedValues.containsKey(name)) {
          resultBuilder.add(resolvedValues.get(name));
          resolvedValues.remove(name);
        }
      }
      for (Object resolved : resolvedValues.values()) {
        resultBuilder.add(resolved);
      }
      try (Writer writer = Files.newWriter(new File(resolvedFile), StandardCharsets.UTF_8)) {
        writer.write(
            EXPORTED_NAME
                + " = "
                + Printer.getWorkspacePrettyPrinter().repr(resultBuilder.build()));
      } catch (IOException e) {
        logger.warning("IO Error writing to file " + resolvedFile + ": " + e);
      }
    }

    this.resolvedValues = null;
  }

  @Subscribe
  public void repositoryOrderEvent(RepositoryOrderEvent event) {
    orderedNames = event.getOrderedNames();
  }

  @Subscribe
  public void resolved(ResolvedEvent event) {
    if (resolvedValues != null) {
      resolvedValues.put(event.getName(), event.getResolvedInformation());
    }
  }
}
