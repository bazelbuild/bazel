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
package com.google.devtools.build.lib.profiler.grapher;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.profiler.CpuUsageTimeSeries;
import com.google.gson.stream.JsonReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * An experimental tool to turn Json profiles into graphs. Do not depend on the continuing existence
 * of this tool.
 *
 * <p>Run this tool like so: bazel run \
 * //src/main/java/com/google/devtools/build/lib/profiler:profiler-grapher \
 * -- /path/to/command.profile > /tmp/tmp.csv
 *
 * <p>Plot the resulting CSV with gnuplot like so: gnuplot -p -e \
 * "set datafile sep ','; plot for [col=1:3] '/tmp/tmp.csv' using col lw 2 with lines \
 * title columnheader"
 */
public class ProfileGrapher {
  public static final long DEFAULT_BUCKET_SIZE_MILLIS = 1000;

  // Decode a JSON object and flattens any nested JSON objects.
  private static Map<String, Object> decodeJsonObject(JsonReader reader) throws IOException {
    reader.beginObject();
    Map<String, Object> data = new HashMap<>();
    while (reader.hasNext()) {
      String name = reader.nextName();
      Object value;
      switch (reader.peek()) {
        case BOOLEAN:
          value = reader.nextBoolean();
          break;
        case NUMBER:
          value = reader.nextDouble();
          break;
        case STRING:
          value = reader.nextString();
          break;
        case BEGIN_OBJECT:
          value = null;
          Map<String, Object> childData = decodeJsonObject(reader);
          for (Map.Entry<String, Object> entry : childData.entrySet()) {
            data.put(name + "." + entry.getKey(), entry.getValue());
          }
          break;
        default:
          reader.skipValue();
          continue;
      }
      data.put(name, value);
    }
    reader.endObject();
    return data;
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 1) {
      System.err.println("Requires filename of a Bazel profile in Chrome event trace format.");
      System.exit(1);
    }
    String filename = args[0];

    // TODO(twerth): Make it possible to select the set of profiler task descriptions on the command
    // line.
    ImmutableList<String> profilerTaskDescriptions =
        ImmutableList.of(
            "Remote execution process wall time",
            "action processing",
            "Remote execution file fetching");
    Map<String, CpuUsageTimeSeries> seriesMap = new LinkedHashMap<>();
    for (String profilerTaskDescription : profilerTaskDescriptions) {
      seriesMap.put(profilerTaskDescription, new CpuUsageTimeSeries(0, DEFAULT_BUCKET_SIZE_MILLIS));
    }

    long maxEndTime = 0;
    // TODO(twerth): Support gzip compressed profiles.
    try (JsonReader reader =
        new JsonReader(
            new BufferedReader(
                new InputStreamReader(
                    Files.newInputStream(Paths.get(filename)), StandardCharsets.UTF_8)))) {
      reader.beginArray();
      while (reader.hasNext()) {
        Map<String, Object> data = decodeJsonObject(reader);
        Object name = data.get("name");
        if ("cpu counters".equals(name)) {
          seriesMap.putIfAbsent(
              "cpu counters", new CpuUsageTimeSeries(0, DEFAULT_BUCKET_SIZE_MILLIS));
          CpuUsageTimeSeries series = seriesMap.get(name);
          Double ts = (Double) data.get("ts");
          long startTimeMillis = Math.round(ts.doubleValue() / 1000);
          Double cpuValue = Double.valueOf((String) data.get("args.cpu"));
          series.addRange(startTimeMillis, startTimeMillis + DEFAULT_BUCKET_SIZE_MILLIS, cpuValue);
        } else {
          Object cat = data.get("cat");
          CpuUsageTimeSeries series = seriesMap.get(cat);
          if (series != null) {
            Long endTimeMillis = decodeAndAdd(series, data);
            if (endTimeMillis != null) {
              maxEndTime = Math.max(maxEndTime, endTimeMillis);
            }
          }
        }
      }
      reader.endArray();
    }

    // Instead of generating a CSV here, we could generate Json and use the Google Charts API to
    // generate interactive web graphs (https://developers.google.com/chart/).
    int len = (int) (maxEndTime / DEFAULT_BUCKET_SIZE_MILLIS) + 1;
    double[][] numbers = new double[seriesMap.size()][];
    List<Map.Entry<String, CpuUsageTimeSeries>> allSeries = new ArrayList<>(seriesMap.entrySet());
    // Write the titles in the first line of the CSV
    StringJoiner stringJoiner = new StringJoiner(",");
    for (int i = 0; i < numbers.length; i++) {
      stringJoiner.add(allSeries.get(i).getKey());
      numbers[i] = allSeries.get(i).getValue().toDoubleArray(len);
    }
    System.out.println(stringJoiner.toString());

    for (int i = 0; i < numbers[0].length; i++) {
      stringJoiner = new StringJoiner(",");
      for (int j = 0; j < numbers.length; j++) {
        stringJoiner.add(String.valueOf(numbers[j][i]));
      }
      System.out.println(stringJoiner.toString());
    }
  }

  /**
   * Decodes the start time and duration from the data, adds it to the given series and returns the
   * end time in milliseconds if it was possible to decode the data, otherwise null.
   */
  private static Long decodeAndAdd(CpuUsageTimeSeries series, Map<String, Object> data) {
    Double ts = (Double) data.get("ts");
    Double dur = (Double) data.get("dur");
    if (ts == null || dur == null) {
      return null;
    }
    long durationMillis = Math.round(dur.doubleValue() / 1000);
    long startTimeMillis = Math.round(ts.doubleValue() / 1000);
    long endTimeMillis = startTimeMillis + durationMillis;
    series.addRange(startTimeMillis, endTimeMillis);
    return endTimeMillis;
  }
}
