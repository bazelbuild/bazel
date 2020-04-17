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
import com.google.gson.stream.JsonReader;
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
      @Nullable String primaryOutputPath) {
    return new AutoValue_TraceEvent(
        category, name, timestamp, duration, threadId, primaryOutputPath);
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

  private static TraceEvent createFromJsonReader(JsonReader reader) throws IOException {
    String category = null;
    String name = null;
    Duration timestamp = null;
    Duration duration = null;
    long threadId = -1;
    String primaryOutputPath = null;

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
        default:
          reader.skipValue();
      }
    }
    reader.endObject();
    return TraceEvent.create(category, name, timestamp, duration, threadId, primaryOutputPath);
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
