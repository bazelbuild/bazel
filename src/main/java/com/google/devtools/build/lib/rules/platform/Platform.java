// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Map;

/** Defines a platform for execution contexts. */
public class Platform implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    PlatformInfo.Builder platformBuilder = PlatformInfo.builder();

    if (ruleContext.attributes().get(PlatformRule.HOST_PLATFORM_ATTR, Type.BOOLEAN)) {
      // Create default constraints based on the current OS and CPU values.
      autodetectHostConstraints(ruleContext, platformBuilder);
    } else {
      platformBuilder.addConstraints(
          ConstraintValue.constraintValues(
              ruleContext.getPrerequisites(PlatformRule.CONSTRAINT_VALUES_ATTR, Mode.DONT_CHECK)));
    }

    Map<String, String> remoteExecutionProperties =
        ruleContext.attributes().get(PlatformRule.REMOTE_EXECUTION_PROPS_ATTR, Type.STRING_DICT);
    platformBuilder.addRemoteExecutionProperties(remoteExecutionProperties);

    PlatformInfo platformInfo;
    try {
      platformInfo = platformBuilder.build();
    } catch (PlatformInfo.DuplicateConstraintException e) {
      // Report the error and return null.
      ruleContext.attributeError(PlatformRule.CONSTRAINT_VALUES_ATTR, e.getMessage());
      return null;
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addNativeDeclaredProvider(platformInfo)
        .build();
  }

  private void autodetectHostConstraints(
      RuleContext ruleContext, PlatformInfo.Builder platformBuilder) {

    // Add the CPU.
    CPU cpu = CPU.getCurrent();
    Iterable<ConstraintValueInfo> cpuConstraintValues =
        ConstraintValue.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.HOST_CPU_CONSTRAINTS_ATTR, Mode.DONT_CHECK));
    for (ConstraintValueInfo constraint : cpuConstraintValues) {
      if (cpu.getCanonicalName().equals(constraint.label().getName())) {
        platformBuilder.addConstraint(constraint);
        break;
      }
    }

    // Add the OS.
    OS os = OS.getCurrent();
    Iterable<ConstraintValueInfo> osConstraintValues =
        ConstraintValue.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.HOST_OS_CONSTRAINTS_ATTR, Mode.DONT_CHECK));
    for (ConstraintValueInfo constraint : osConstraintValues) {
      if (os.getCanonicalName().equals(constraint.label().getName())) {
        platformBuilder.addConstraint(constraint);
        break;
      }
    }
  }

  /** Retrieves and casts the provider from the given target. */
  public static PlatformInfo platform(TransitiveInfoCollection target) {
    Object provider = target.get(PlatformInfo.SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof PlatformInfo);
    return (PlatformInfo) provider;
  }

  /** Retrieves and casts the providers from the given targets. */
  public static Iterable<PlatformInfo> platforms(
      Iterable<? extends TransitiveInfoCollection> targets) {
    return Iterables.transform(
        targets,
        new Function<TransitiveInfoCollection, PlatformInfo>() {
          @Override
          public PlatformInfo apply(TransitiveInfoCollection target) {
            return platform(target);
          }
        });
  }
}
