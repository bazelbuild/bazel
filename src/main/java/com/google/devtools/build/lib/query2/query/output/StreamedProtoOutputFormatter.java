// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.stream.StreamSupport;

/**
 * An output formatter that outputs a protocol buffer representation of a query result and outputs
 * the proto bytes to the output print stream. By taking the bytes and calling {@code mergeFrom()}
 * on a {@code Build.QueryResult} object the full result can be reconstructed.
 */
public class StreamedProtoOutputFormatter extends ProtoOutputFormatter {
  @Override
  public String getName() {
    return "streamed_proto";
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options, LabelPrinter labelPrinter) {
    return new OutputFormatterCallback<Target>() {
      private final LabelPrinter ourLabelPrinter = labelPrinter;

      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        try {
          StreamSupport.stream(partialResult.spliterator(), /* parallel= */ true)
              .map(this::toProto)
              .map(StreamedProtoOutputFormatter::writeDelimited)
              .forEach(this::writeToOutputStreamThreadSafe);
        } catch (WrappedIOException e) {
          throw e.getCause();
        } catch (WrappedInterruptedException e) {
          throw e.getCause();
        }
      }

      private Build.Target toProto(Target target) {
        try {
          return toTargetProtoBuffer(target, ourLabelPrinter);
        } catch (InterruptedException e) {
          throw new WrappedInterruptedException(e);
        }
      }

      private synchronized void writeToOutputStreamThreadSafe(ByteArrayOutputStream bout) {
        try {
          bout.writeTo(out);
        } catch (IOException e) {
          throw new WrappedIOException(e);
        }
      }
    };
  }

  private static ByteArrayOutputStream writeDelimited(Build.Target targetProtoBuffer) {
    try {
      var bout = new ByteArrayOutputStream(targetProtoBuffer.getSerializedSize() + 10);
      targetProtoBuffer.writeDelimitedTo(bout);
      return bout;
    } catch (IOException e) {
      throw new WrappedIOException(e);
    }
  }

  private static class WrappedIOException extends RuntimeException {
    private WrappedIOException(IOException cause) {
      super(cause);
    }

    @Override
    public synchronized IOException getCause() {
      return (IOException) super.getCause();
    }
  }

  private static class WrappedInterruptedException extends RuntimeException {
    private WrappedInterruptedException(InterruptedException cause) {
      super(cause);
    }

    @Override
    public synchronized InterruptedException getCause() {
      return (InterruptedException) super.getCause();
    }
  }
}
