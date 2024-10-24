// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.execlog;

import com.google.devtools.build.execlog.ConverterOptions.FormatAndPath;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.SpawnLogReconstructor;
import com.google.devtools.build.lib.exec.StableSort;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.BinaryInputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.JsonInputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.BinaryOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.JsonOutputStreamWrapper;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;

/** A tool to convert between Bazel execution log formats. */
final class ExecLogConverter {
  private ExecLogConverter() {}

  private static MessageInputStream<SpawnExec> getMessageInputStream(FormatAndPath log)
      throws IOException {
    InputStream in = Files.newInputStream(log.path());
    switch (log.format()) {
      case BINARY:
        return new BinaryInputStreamWrapper<>(in, SpawnExec.getDefaultInstance());
      case JSON:
        return new JsonInputStreamWrapper<>(in, SpawnExec.getDefaultInstance());
      case COMPACT:
        return new SpawnLogReconstructor(in);
    }
    throw new AssertionError("unsupported input format");
  }

  private static MessageOutputStream<SpawnExec> getMessageOutputStream(FormatAndPath log)
      throws IOException {
    OutputStream out = Files.newOutputStream(log.path());
    switch (log.format()) {
      case BINARY:
        return new BinaryOutputStreamWrapper<>(out);
      case JSON:
        return new JsonOutputStreamWrapper<>(out);
      case COMPACT:
        // unsupported
    }
    throw new AssertionError("unsupported output format");
  }

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.builder().optionsClasses(ConverterOptions.class).build();
    op.parseAndExitUponError(args);

    ConverterOptions options = op.getOptions(ConverterOptions.class);

    if (options.input == null) {
      System.err.println("--input must be specified.");
      System.exit(1);
    }

    if (options.output == null) {
      System.err.println("--output must be specified.");
      System.exit(1);
    }

    if (!Files.exists(options.input.path()) && !Files.isReadable(options.input.path())) {
      System.err.println(
          "Input path '" + options.input.path() + "' does not exist or is not readable.");
      System.exit(1);
    }

    try (MessageInputStream<SpawnExec> in = getMessageInputStream(options.input);
        MessageOutputStream<SpawnExec> out = getMessageOutputStream(options.output)) {
      if (options.sort) {
        StableSort.stableSort(in, out);
      } else {
        SpawnExec ex;
        while ((ex = in.read()) != null) {
          out.write(ex);
        }
      }
    }
  }
}
