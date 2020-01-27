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
import java.io.IOException;
import java.io.OutputStream;

/**
 * Helper for managing the {@link OutputStream} to which query/cquery/aquery results should be
 * written.
 */
public interface QueryRuntimeHelper extends AutoCloseable {
  /**
   * Returns the {@link OutputStream} to which to write query results. This {@link
   * QueryRuntimeHelper} instance, not the caller, is responsible for closing the {@link
   * OutputStream}.
   */
  OutputStream getOutputStreamForQueryOutput();

  /**
   * Should be called after the query is successfully evaluated and the entire query output is
   * written to the {@link OutputStream} returned by {@link #getOutputStreamForQueryOutput}.
   *
   * <p>In particular, this method shouldn't be called if query evaluation fails.
   */
  void afterQueryOutputIsWritten() throws IOException, InterruptedException;

  /** Must be called at some point near the end of the life of the query command. */
  @Override
  void close() throws IOException;

  /** Factory for {@link QueryRuntimeHelper} instances. */
  interface Factory {
    QueryRuntimeHelper create(CommandEnvironment env) throws IOException, CommandLineException;

    class CommandLineException extends Exception {
      public CommandLineException(String message) {
        super(message);
      }
    }
  }

  /**
   * A {@link Factory} for {@link StdoutQueryRuntimeHelper} instances that simply wrap the given
   * {@link CommandEnvironment} instance's stdout.
   *
   * <p>This is intended to be the default {@link Factory}.
   */
  class StdoutQueryRuntimeHelperFactory implements Factory {
    public static final StdoutQueryRuntimeHelperFactory INSTANCE =
        new StdoutQueryRuntimeHelperFactory();

    private StdoutQueryRuntimeHelperFactory() {}

    @Override
    public QueryRuntimeHelper create(CommandEnvironment env) {
      return createInternal(env.getReporter().getOutErr().getOutputStream());
    }

    public QueryRuntimeHelper createInternal(OutputStream stdoutOutputStream) {
      return new StdoutQueryRuntimeHelper(stdoutOutputStream);
    }

    /** A QueryRuntimeHelper that simply wraps a {@link OutputStream} for stdout. */
    @VisibleForTesting
    public static class StdoutQueryRuntimeHelper implements QueryRuntimeHelper {
      private final OutputStream stdoutOutputStream;

      private StdoutQueryRuntimeHelper(OutputStream stdoutOutputStream) {
        this.stdoutOutputStream = stdoutOutputStream;
      }

      @Override
      public OutputStream getOutputStreamForQueryOutput() {
        return stdoutOutputStream;
      }

      @Override
      public void afterQueryOutputIsWritten() {}

      @Override
      public void close() throws IOException {}
    }
  }
}
