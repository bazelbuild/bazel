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

package com.google.devtools.build.lib.query2.cquery;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.AttributeReader;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.TargetOutputter;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.IOException;
import java.io.OutputStream;

/** Cquery implementation of BUILD-style output. */
class BuildOutputFormatterCallback extends CqueryThreadsafeCallback {
  BuildOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<KeyedConfiguredTarget> accessor) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
  }

  @Override
  public String getName() {
    return "build";
  }

  /** {@link AttributeReader} implementation that returns the exact value an attribute takes. */
  private static class CqueryAttributeReader implements AttributeReader {
    private final ConfiguredAttributeMapper attributeMap;

    CqueryAttributeReader(ConfiguredAttributeMapper attributeMap) {
      this.attributeMap = attributeMap;
    }

    /**
     * Cquery knows which select path is taken so it knows the exact value the attribute takes. Note
     * that null values are also possible - these are represented as an empty value list.
     */
    @Override
    public Iterable<Object> getPossibleValues(Rule rule, Attribute attr) {
      Object actualValue = attributeMap.get(attr.getName(), attr.getType());
      return actualValue == null ? ImmutableList.of() : ImmutableList.of(actualValue);
    }
  }

  private ConfiguredAttributeMapper getAttributeMap(KeyedConfiguredTarget kct)
      throws InterruptedException {
    Rule associatedRule = accessor.getTarget(kct).getAssociatedRule();
    if (associatedRule == null) {
      return null;
    } else if (kct.getConfiguredTarget() instanceof OutputFileConfiguredTarget) {
      return ConfiguredAttributeMapper.of(
          associatedRule,
          accessor.getGeneratingConfiguredTarget(kct).getConfigConditions(),
          kct.getConfigurationChecksum());
    } else {
      return ConfiguredAttributeMapper.of(
          associatedRule, kct.getConfigConditions(), kct.getConfigurationChecksum());
    }
  }

  @Override
  public void processOutput(Iterable<KeyedConfiguredTarget> partialResult)
      throws InterruptedException, IOException {
    BuildOutputFormatter.TargetOutputter outputter =
        new TargetOutputter(
            printStream,
            // This tells TargetOutputter which attributes to print as selects without resolving
            // those selects. For cquery we never have to do this since we can always resolve
            // selects. Going forward we could expand this to show both the complete select
            // and which path is chosen, which people may find even more informative.
            (rule, attr) -> false,
            System.lineSeparator());
    for (KeyedConfiguredTarget configuredTarget : partialResult) {
      Target target = accessor.getTarget(configuredTarget);
      outputter.output(target, new CqueryAttributeReader(getAttributeMap(configuredTarget)));
    }
  }
}
