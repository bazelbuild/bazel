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

import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * An output formatter that prints a list of targets according to ndjson spec to the output print
 * stream.
 */
public class StreamedJSONProtoOutputFormatter extends ProtoOutputFormatter {
  @Override
  public String getName() {
    return "streamed_jsonproto";
  }

  private final JsonFormat.Printer jsonPrinter = JsonFormat.printer();

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options, RepositoryMapping mainRepoMapping) {
    return new OutputFormatterCallback<Target>() {
      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        for (Target target : partialResult) {
          out.write(
              jsonPrinter
                  .omittingInsignificantWhitespace()
                  .print(toTargetProtoBuffer(target))
                  .getBytes(StandardCharsets.UTF_8));
          out.write(System.lineSeparator().getBytes(StandardCharsets.UTF_8));
        }
      }
    };
  }
}
