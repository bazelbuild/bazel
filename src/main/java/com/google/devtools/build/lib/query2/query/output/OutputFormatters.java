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

package com.google.devtools.build.lib.query2.query.output;

import static java.util.stream.Collectors.joining;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import javax.annotation.Nullable;

/** Encapsulates available {@link OutputFormatter}s and selection logic. */
public class OutputFormatters {

  private OutputFormatters() {}

  /** Returns all available {@link OutputFormatter}s. */
  public static ImmutableList<OutputFormatter> getDefaultFormatters() {
    return ImmutableList.of(
        new LabelOutputFormatter(false),
        new LabelOutputFormatter(true),
        new BuildOutputFormatter(),
        new MinrankOutputFormatter(),
        new MaxrankOutputFormatter(),
        new PackageOutputFormatter(),
        new LocationOutputFormatter(),
        new GraphOutputFormatter(),
        new XmlOutputFormatter(),
        new ProtoOutputFormatter(),
        new StreamedJSONProtoOutputFormatter(),
        new StreamedProtoOutputFormatter());
  }

  /** Returns the names of all {@link OutputFormatter}s in the input. */
  public static String formatterNames(Iterable<OutputFormatter> formatters) {
    return Streams.stream(formatters).map(OutputFormatter::getName).collect(joining(", "));
  }

  /** Returns the name of all streaming {@link OutputFormatter}s in the input. */
  public static String streamingFormatterNames(Iterable<OutputFormatter> formatters) {
    return Streams.stream(formatters)
        .filter(Predicates.instanceOf(StreamedFormatter.class))
        .map(OutputFormatter::getName)
        .collect(joining(", "));
  }

  /** Returns the {@link OutputFormatter} for the specified type. */
  @Nullable
  public static OutputFormatter getFormatter(Iterable<OutputFormatter> formatters, String type) {
    for (OutputFormatter formatter : formatters) {
      if (formatter.getName().equals(type)) {
        return formatter;
      }
    }

    return null;
  }

}
