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
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Files;
import com.google.devtools.build.lib.cmdline.Label;
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

/** Module providing the collection of the resolved values for the repository rules executed. */
public final class RepositoryResolvedModule extends BlazeModule {
  public static final String EXPORTED_NAME = "resolved";

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
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
            EXPORTED_NAME + " = " + getWorkspacePrettyPrinter().repr(resultBuilder.build()));
        writer.close();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("IO Error writing to file %s", resolvedFile);
      }
    }

    this.resolvedValues = null;
  }

  /**
   * Returns a pretty printer that represents values in a form usable in WORKSPACE files.
   *
   * <p>In WORKSPACE files, the Label constructor is not available. Fortunately, in all places where
   * a label is needed, we can pass the canonical string associated with this label.
   */
  private static Printer.PrettyPrinter getWorkspacePrettyPrinter() {
    return new Printer.PrettyPrinter(new StringBuilder()) {
      @Override
      public Printer.BasePrinter repr(Object o) {
        if (o instanceof Label) {
          this.repr(((Label) o).getCanonicalForm());
        } else {
          super.repr(o);
        }
        return this;
      }
    };
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
