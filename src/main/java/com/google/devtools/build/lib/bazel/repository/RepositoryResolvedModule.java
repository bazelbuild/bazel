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
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.common.options.OptionsBase;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

/** Module providing the collection of the resolved values for the repository rules executed. */
public final class RepositoryResolvedModule extends BlazeModule {
  public static final String EXPORTED_NAME = "resolved";

  private static final Logger logger = Logger.getLogger(RepositoryResolvedModule.class.getName());
  private ImmutableList.Builder<Object> resultBuilder;
  private String resolvedFile;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableSet.of("sync", "fetch", "build", "query").contains(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(RepositoryResolvedOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    RepositoryResolvedOptions options =
        env.getOptions().getOptions(RepositoryResolvedOptions.class);
    if (options != null && !Strings.isNullOrEmpty(options.repositoryResolvedFile)) {
      this.resolvedFile = options.repositoryResolvedFile;
      env.getEventBus().register(this);
      this.resultBuilder = new ImmutableList.Builder<>();
    } else {
      this.resolvedFile = null;
    }
  }

  @Override
  public void afterCommand() {
    if (resolvedFile != null) {
      try (Writer writer = Files.newWriter(new File(resolvedFile), StandardCharsets.UTF_8)) {
        writer.write(
            EXPORTED_NAME
                + " = "
                + Printer.getWorkspacePrettyPrinter().repr(resultBuilder.build()));
      } catch (IOException e) {
        logger.warning("IO Error writing to file " + resolvedFile + ": " + e);
      }
    }

    this.resultBuilder = null;
  }

  @Subscribe
  public void repositoryResolved(RepositoryResolvedEvent event) {
    if (resultBuilder != null) {
      resultBuilder.add(event.getResolvedInformation());
    }
  }
}
