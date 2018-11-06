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
package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.query2.CommonQueryOptions;
import com.google.devtools.build.lib.query2.QueryOutputEvent;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;

/** Helper for interfacing with the BEP from the `query`, `cquery`, and `aquery` commands. */
public interface QueryBEPHelper extends AutoCloseable {
  OutputStream getOutputStreamForQueryOutput();

  void afterQueryOutputIsWritten();

  @Override
  void close() throws IOException;

  static BuildEventId getBuildEventIdOfQueryOutputEvent() {
    return QueryOutputEvent.BUILD_EVENT_ID;
  }

  static QueryBEPHelper create(
      CommandEnvironment env, CommonQueryOptions commonQueryOptions) throws IOException {
    if (commonQueryOptions.uploadQueryOutputUsingBEP) {
      Path queryOutputFilePath = env.getOutputBase().getChild("tmp_query_output");
      OutputStream queryOutputStream = queryOutputFilePath.getOutputStream();
      return new QueryBEPHelperForUploadingQueryOutputUsingBEP(
          env.getCommandName(),
          env.getEventBus(),
          queryOutputFilePath,
          queryOutputStream);
    } else {
      return new QueryBEPHelperForStdoutOutput(
          env.getEventBus(),
          env.getReporter().getOutErr().getOutputStream());
    }
  }

  static QueryBEPHelperForNonBuildingCommand createForNonBuildingCommand(
      CommandEnvironment env, CommonQueryOptions commonQueryOptions) throws  IOException {
    QueryBEPHelper delegate = create(env, commonQueryOptions);
    return new QueryBEPHelperForNonBuildingCommand(env, delegate);
  }

  @VisibleForTesting
  static QueryBEPHelper createForUnitTests(
      String commandName,
      EventBus eventBus,
      CommonQueryOptions commonQueryOptions,
      Path queryOutputFilePath,
      OutputStream stdoutOutputStream) throws IOException {
    if (commonQueryOptions.uploadQueryOutputUsingBEP) {
      OutputStream queryOutputStream = queryOutputFilePath.getOutputStream();
      return new QueryBEPHelperForUploadingQueryOutputUsingBEP(
          commandName,
          eventBus,
          queryOutputFilePath,
          queryOutputStream);
    } else {
      return new QueryBEPHelperForStdoutOutput(eventBus, stdoutOutputStream);
    }
  }

  /**
   * Implementation of {@link QueryBEPHelper} for the situation where we want to write query output
   * to stdout on the console.
   */
  class QueryBEPHelperForStdoutOutput implements QueryBEPHelper {
    private final EventBus eventBus;
    private final OutputStream stdoutOutputStream;

    private QueryBEPHelperForStdoutOutput(EventBus eventBus, OutputStream stdoutOutputStream) {
      this.eventBus = eventBus;
      this.stdoutOutputStream = stdoutOutputStream;
    }

    @Override
    public OutputStream getOutputStreamForQueryOutput() {
      return stdoutOutputStream;
    }

    @Override
    public void afterQueryOutputIsWritten() {
      eventBus.post(QueryOutputEvent.forOutputWrittenToStdout());
    }

    @Override
    public void close() {
      // Nothing to do here. The CommandEnvironment owns the stdout OutputStream.
    }
  }

  /**
   * Implementation of {@link QueryBEPHelper} for the situation where we want to write query output
   * to a temporary file and then upload that temporary file via BEP.
   */
  class QueryBEPHelperForUploadingQueryOutputUsingBEP implements QueryBEPHelper {
    private final String commandName;
    private final EventBus eventBus;
    private final Path queryOutputFilePath;
    private final OutputStream queryOutputStream;

    private QueryBEPHelperForUploadingQueryOutputUsingBEP(
        String commandName,
        EventBus eventBus,
        Path queryOutputFilePath,
        OutputStream queryOutputStream) {
      this.commandName = commandName;
      this.eventBus = eventBus;
      this.queryOutputFilePath = queryOutputFilePath;
      this.queryOutputStream = queryOutputStream;
    }

    @Override
    public OutputStream getOutputStreamForQueryOutput() {
      return queryOutputStream;
    }

    @Override
    public void afterQueryOutputIsWritten() {
      eventBus.post(QueryOutputEvent.forOutputWrittenToFile(commandName, queryOutputFilePath));
    }

    @Override
    public void close() throws IOException {
      queryOutputStream.close();
    }
  }

  /** Implementation of {@link QueryBEPHelper} for commands that don't "build". */
  class QueryBEPHelperForNonBuildingCommand implements QueryBEPHelper {
    private final CommandEnvironment env;
    private final QueryBEPHelper delegate;

    private QueryBEPHelperForNonBuildingCommand(CommandEnvironment env, QueryBEPHelper delegate) {
      this.env = env;
      this.delegate = delegate;
    }

    public void beforeQueryOutputIsWritten() {
      env.getEventBus()
          .post(
              NoBuildEvent.newBuilder()
                  .setCommand(env.getCommandName())
                  .setStartTimeMillis(env.getCommandStartTime())
                  .addAdditionalChildrenEvents(
                      ImmutableList.of(getBuildEventIdOfQueryOutputEvent()))
                  .setSeparateFinishedEvent(true)
                  .build());
    }

    @Override
    public OutputStream getOutputStreamForQueryOutput() {
      return delegate.getOutputStreamForQueryOutput();
    }

    @Override
    public void afterQueryOutputIsWritten() {
      delegate.afterQueryOutputIsWritten();
    }

    public void afterExitCodeIsDetermined(ExitCode exitCode) {
      env.getEventBus().post(
          new NoBuildRequestFinishedEvent(
              exitCode, env.getRuntime().getClock().currentTimeMillis()));
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }
  }
}
