// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.profiler;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Represents a single trace event in a JSON profile.
 *
 * <p>Format is documented in
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
@AutoValue
public abstract class TraceEvent {
  public static TraceEvent create(
      @Nullable String category,
      String name,
      @Nullable Duration timestamp,
      @Nullable Duration duration,
      long threadId,
      @Nullable String primaryOutputPath,
      @Nullable String targetLabel) {
    return new AutoValue_TraceEvent(
        category, name, timestamp, duration, threadId, primaryOutputPath, targetLabel);
  }

  @Nullable
  public abstract String category();

  public abstract String name();

  @Nullable
  public abstract Duration timestamp();

  @Nullable
  public abstract Duration duration();

  public abstract long threadId();

  // Only applicable to action-related TraceEvents.
  @Nullable
  public abstract String primaryOutputPath();

  @Nullable
  public abstract String targetLabel();

  private static TraceEvent createFromJsonReader(JsonReader reader) throws IOException {
    String category = null;
    String name = null;
    Duration timestamp = null;
    Duration duration = null;
    long threadId = -1;
    String primaryOutputPath = null;
    String targetLabel = null;

    reader.beginObject();
    while (reader.hasNext()) {
      switch (reader.nextName()) {
        case "cat":
          category = reader.nextString();
          break;
        case "name":
          name = reader.nextString();
          break;
        case "ts":
          // Duration has no microseconds :-/.
          timestamp = Duration.ofNanos(reader.nextLong() * 1000);
          break;
        case "dur":
          duration = Duration.ofNanos(reader.nextLong() * 1000);
          break;
        case "tid":
          threadId = reader.nextLong();
          break;
        case "out":
          primaryOutputPath = reader.nextString();
          break;
        case "args":
          ImmutableMap<String, Object> args = parseMap(reader);
          Object target = args.get("target");
          targetLabel = target instanceof String ? (String) target : null;
          break;
        default:
          reader.skipValue();
      }
    }
    reader.endObject();
    return TraceEvent.create(
        category, name, timestamp, duration, threadId, primaryOutputPath, targetLabel);
  }

  private static ImmutableMap<String, Object> parseMap(JsonReader reader) throws IOException {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    reader.beginObject();
    while (reader.peek() != JsonToken.END_OBJECT) {
      String name = reader.nextName();
      Object val = parseSingleValueRecursively(reader);
      builder.put(name, val);
    }
    reader.endObject();

    return builder.build();
  }

  private static ImmutableList<Object> parseArray(JsonReader reader) throws IOException {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();

    reader.beginArray();
    while (reader.peek() != JsonToken.END_ARRAY) {
      Object val = parseSingleValueRecursively(reader);
      builder.add(val);
    }
    reader.endArray();

    return builder.build();
  }

  private static Object parseSingleValueRecursively(JsonReader reader) throws IOException {
    JsonToken nextToken = reader.peek();
    switch (nextToken) {
      case BOOLEAN:
        return reader.nextBoolean();
      case NULL:
        reader.nextNull();
        return null;
      case NUMBER:
        // Json's only numeric type is number, using Double to accommodate all types
        return reader.nextDouble();
      case STRING:
        return reader.nextString();
      case BEGIN_OBJECT:
        return parseMap(reader);
      case BEGIN_ARRAY:
        return parseArray(reader);
      default:
        throw new IOException("Unexpected token " + nextToken.name());
    }
  }

  public static List<TraceEvent> parseTraceEvents(JsonReader reader) throws IOException {
    List<TraceEvent> traceEvents = new ArrayList<>();
    reader.beginArray();
    while (reader.hasNext()) {
      traceEvents.add(createFromJsonReader(reader));
    }
    reader.endArray();
    return traceEvents;
  }
}
