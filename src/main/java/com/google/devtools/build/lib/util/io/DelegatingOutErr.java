// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util.io;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * An {@link OutErr} specialization that supports subscribing / removing
 * sinks, using {@link #addSink(OutErr)} and {@link #removeSink(OutErr)}.
 * A sink is a destination to which the {@link DelegatingOutErr} will write.
 *
 * Also, we can hook up {@link System#out} / {@link System#err} as sources.
 */
public final class DelegatingOutErr extends OutErr {

  /**
   * Create a new instance to which no sinks have subscribed (basically just
   * like a {@code /dev/null}.
   */
  public DelegatingOutErr() {
    super(new DelegatingOutputStream(), new DelegatingOutputStream());
  }


  private final DelegatingOutputStream outSink() {
    return (DelegatingOutputStream) getOutputStream();
  }

  private final DelegatingOutputStream errSink() {
    return (DelegatingOutputStream) getErrorStream();
  }

  /**
   * Add a sink, that is, after calling this method, {@code outErrSink} will
   * receive all output / errors written to {@code this} object.
   */
  public void addSink(OutErr outErrSink) {
    outSink().addSink(outErrSink.getOutputStream());
    errSink().addSink(outErrSink.getErrorStream());
  }

  /**
   * Remove the sink, that is, after calling this method, {@code outErrSink}
   * will no longer receive output / errors written to {@code this} object.
   */
  public void removeSink(OutErr outErrSink) {
    outSink().removeSink(outErrSink.getOutputStream());
    errSink().removeSink(outErrSink.getErrorStream());
  }

  private static class DelegatingOutputStream extends OutputStream {

    private final List<OutputStream> sinks = new ArrayList<>();

    public void addSink(OutputStream sink) {
      sinks.add(Preconditions.checkNotNull(sink));
    }

    public void removeSink(OutputStream sink) {
      sinks.remove(sink);
    }

    @Override
    public void write(int b) throws IOException {
      for (OutputStream sink : sinks) {
        sink.write(b);
      }
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
      for (OutputStream sink : sinks) {
        sink.write(b, off, len);
      }
    }

    @Override
    public void write(byte[] b) throws IOException {
      for (OutputStream sink : sinks) {
        sink.write(b);
      }
    }

    @Override
    public void close() throws IOException {
      // Ensure that we close all sinks even if one throws.
      IOException firstException = null;
      for (OutputStream sink : sinks) {
        try {
          sink.close();
        } catch (IOException e) {
          if (firstException == null) {
            firstException = e;
          } else {
            firstException.addSuppressed(e);
          }
        }
      }

      if (firstException != null) {
        throw firstException;
      }
    }

    @Override
    public void flush() throws IOException {
      for (OutputStream sink : sinks) {
        sink.flush();
      }
    }
  }
}
