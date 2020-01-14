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

import com.google.common.base.Optional;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.InfoListener;
import com.google.devtools.build.lib.profiler.output.PhaseText;
import com.google.devtools.build.lib.profiler.statistics.CriticalPathStatistics;
import com.google.devtools.build.lib.profiler.statistics.PhaseSummaryStatistics;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Utility class to handle parsing the JSON trace profiles.
 *
 * <p>The format itself is documented in
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
public final class JsonProfile {
  private JsonProfile() {}

  public static void parseProfileAndDumpStats(
      Reporter reporter,
      String dumpMode,
      PrintStream out,
      File profileFile,
      InfoListener infoListener)
      throws IOException {
    boolean gzipped = profileFile.getName().endsWith(".gz");
    InputStream inputStream = new FileInputStream(profileFile);
    if (gzipped) {
      inputStream = new GZIPInputStream(inputStream);
    }

    parseProfileAndDumpStats(reporter, dumpMode, out, inputStream, infoListener);
  }

  public static void parseProfileAndDumpStats(
      Reporter reporter,
      String dumpMode,
      PrintStream out,
      InputStream inputStream,
      InfoListener infoListener)
      throws IOException {
    if (dumpMode != null) {
      reporter.handle(
          Event.warn("--dump has not been implemented yet for the JSON profile, ignoring."));
    }

    try (JsonReader reader =
        new JsonReader(
            new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8)))) {
      if (reader.peek() == JsonToken.BEGIN_OBJECT) {
        reader.beginObject();
        while (reader.hasNext()) {
          String objectKey = reader.nextName();
          if ("otherData".equals(objectKey)) {
            parseAndAnnounceBuildMetadata(infoListener, reader);
          } else if ("traceEvents".equals(objectKey)) {
            List<TraceEvent> traceEvents = TraceEvent.parseTraceEvents(reader);
            PhaseSummaryStatistics phaseSummaryStatistics = new PhaseSummaryStatistics();
            TraceEvent lastPhaseEvent = null;
            for (TraceEvent traceEvent : traceEvents) {
              if (ProfilerTask.PHASE.description.equals(traceEvent.category())) {
                if (lastPhaseEvent != null) {
                  phaseSummaryStatistics.addProfilePhase(
                      ProfilePhase.getPhaseFromDescription(lastPhaseEvent.name()),
                      traceEvent.timestamp().minus(lastPhaseEvent.timestamp()));
                }
                lastPhaseEvent = traceEvent;
              }
            }
            if (lastPhaseEvent != null) {
              TraceEvent lastEvent = Iterables.getLast(traceEvents);
              phaseSummaryStatistics.addProfilePhase(
                  ProfilePhase.getPhaseFromDescription(lastPhaseEvent.name()),
                  lastEvent.timestamp().minus(lastPhaseEvent.timestamp()));
            }

            new PhaseText(
                    out,
                    phaseSummaryStatistics,
                    /* phaseStatistics= */ Optional.absent(),
                    Optional.of(new CriticalPathStatistics(traceEvents)))
                .print();
          } else {
            reader.skipValue();
          }
        }
      }
    } catch (IOException e) {
      reporter.handle(Event.error("Failed to analyze profile file(s): " + e.getMessage()));
      throw e;
    }
  }

  private static void parseAndAnnounceBuildMetadata(InfoListener infoListener, JsonReader reader)
      throws IOException {
    reader.beginObject();
    String buildId = null;
    String date = null;
    String outputBase = null;
    while (reader.hasNext()) {
      switch (reader.nextName()) {
        case "build_id":
          buildId = reader.nextString();
          break;
        case "date":
          date = reader.nextString();
          break;
        case "output_base":
          outputBase = reader.nextString();
          break;
        default:
          reader.skipValue();
      }
    }
    reader.endObject();

    infoListener.info(
        "Profile created on " + date + ", build ID: " + buildId + ", output base: " + outputBase);
  }
}
