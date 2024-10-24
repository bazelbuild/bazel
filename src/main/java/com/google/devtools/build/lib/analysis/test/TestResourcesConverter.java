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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.Map;

public class TestResourcesConverter
    extends Converter.Contextless<Pair<String, Map<TestSize, Double>>> {
  private static final Converters.AssignmentConverter assignmentConverter =
      new Converters.AssignmentConverter();
  private static final ResourceConverter.DoubleConverter resourceConverter =
      new ResourceConverter.DoubleConverter(
          /* keywords= */ ImmutableMap.of(
              ResourceConverter.HOST_CPUS_KEYWORD,
                  () -> (double) ResourceConverter.HOST_CPUS_SUPPLIER.get(),
              ResourceConverter.HOST_RAM_KEYWORD,
                  () -> (double) ResourceConverter.HOST_RAM_SUPPLIER.get()),
          /* minValue= */ 0.0,
          /* maxValue= */ Double.MAX_VALUE);

  @Override
  public String getTypeDescription() {
    return "a resource name followed by equal and 1 float or 4 float, e.g memory=10,30,60,100";
  }

  @Override
  public Pair<String, Map<TestSize, Double>> convert(String input) throws OptionsParsingException {
    Map.Entry<String, String> assignment = assignmentConverter.convert(input);
    ArrayList<Double> values = new ArrayList<>(TestSize.values().length);
    for (String s : Splitter.on(",").splitToList(assignment.getValue())) {
      values.add(resourceConverter.convert(s));
    }

    if (values.size() != 1 && values.size() != TestSize.values().length) {
      throw new OptionsParsingException("Invalid number of comma-separated entries in " + input);
    }

    EnumMap<TestSize, Double> amounts = Maps.newEnumMap(TestSize.class);
    for (TestSize size : TestSize.values()) {
      amounts.put(size, values.get(Math.min(values.size() - 1, size.ordinal())));
    }
    return Pair.of(assignment.getKey(), amounts);
  }
}
