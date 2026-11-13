// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import javax.annotation.Nullable;

/**
 * An output formatter that prints a list of targets according to ndjson spec to the output print
 * stream.
 */
public class StreamedJSONProtoOutputFormatter extends ProtoOutputFormatter {
  @Override
  public String getName() {
    return "streamed_jsonproto";
  }

  private final JsonFormat.Printer jsonPrinter =
      JsonFormat.printer().omittingInsignificantWhitespace();

  /** Writes one unconfigured target as a single JSON line. */
  public void writeTarget(Target target, LabelPrinter labelPrinter, OutputStream out)
      throws IOException, InterruptedException {
    writeTargetJson(target, labelPrinter, out);
    out.write('\n');
  }

  /** Writes a snapshot row even if the target's transitive hash is unknown. */
  public void writeHashedTarget(
      Target target, LabelPrinter labelPrinter, @Nullable String digest, OutputStream out)
      throws IOException, InterruptedException {
    // Hashes contain only hex digits. Target labels and attributes use the normal JSON printer.
    String hash = digest == null ? "null" : "\"" + digest + "\"";
    out.write(("{\"hash\":" + hash + ",\"target\":").getBytes(StandardCharsets.UTF_8));
    writeTargetJson(target, labelPrinter, out);
    out.write('}');
    out.write('\n');
  }

  private void writeTargetJson(Target target, LabelPrinter labelPrinter, OutputStream out)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    out.write(
        jsonPrinter
            .print(toTargetProtoBuffer(target, labelPrinter))
            .getBytes(StandardCharsets.UTF_8));
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options, LabelPrinter labelPrinter) {
    return new OutputFormatterCallback<Target>() {
      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        for (Target target : partialResult) {
          writeTarget(target, labelPrinter, out);
        }
      }
    };
  }
}
