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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.AutoCpuConverter;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import java.util.Map;

/** Defines a platform for execution contexts. */
public class Platform implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    PlatformOptions platformOptions =
        ruleContext.getConfiguration().getOptions().get(PlatformOptions.class);

    PlatformInfo.Builder platformBuilder = PlatformInfo.builder().setLabel(ruleContext.getLabel());

    ImmutableList<PlatformInfo> parentPlatforms =
        PlatformProviderUtils.platforms(
            ruleContext.getPrerequisites(PlatformRule.PARENTS_PLATFORM_ATTR));

    if (parentPlatforms.size() > 1) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.PARENTS_PLATFORM_ATTR,
          PlatformRule.PARENTS_PLATFORM_ATTR + " attribute must have a single value");
    }
    PlatformInfo parentPlatform = Iterables.getFirst(parentPlatforms, null);
    platformBuilder.setParent(parentPlatform);

    if (!platformOptions.autoConfigureHostPlatform) {
      // If the flag is set, the constraints are defaulted by @local_config_platform.
      setDefaultConstraints(platformBuilder, ruleContext);
    }

    // Add the declared constraints. Because setting the host_platform or target_platform attribute
    // to true on a platform automatically includes the detected CPU and OS constraints, if the
    // constraint_values attribute tries to add those, this will throw an error.
    platformBuilder.addConstraints(
        PlatformProviderUtils.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.CONSTRAINT_VALUES_ATTR)));

    String remoteExecutionProperties =
        ruleContext.attributes().get(PlatformRule.REMOTE_EXECUTION_PROPS_ATTR, Type.STRING);
    platformBuilder.setRemoteExecutionProperties(remoteExecutionProperties);

    Map<String, String> execProperties =
        ruleContext.attributes().get(PlatformRule.EXEC_PROPS_ATTR, Type.STRING_DICT);
    if (execProperties != null && !execProperties.isEmpty()) {
      platformBuilder.setExecProperties(ImmutableMap.copyOf(execProperties));
    }

    PlatformInfo platformInfo;
    try {
      platformInfo = platformBuilder.build();
    } catch (ConstraintCollection.DuplicateConstraintException e) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.CONSTRAINT_VALUES_ATTR, e.getMessage());
    } catch (PlatformInfo.ExecPropertiesException e) {
      throw ruleContext.throwWithAttributeError(PlatformRule.EXEC_PROPS_ATTR, e.getMessage());
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addNativeDeclaredProvider(platformInfo)
        .build();
  }

  private void setDefaultConstraints(PlatformInfo.Builder platformBuilder, RuleContext ruleContext)
      throws RuleErrorException {
    Boolean isHostPlatform =
        ruleContext.attributes().get(PlatformRule.HOST_PLATFORM_ATTR, Type.BOOLEAN);
    Boolean isTargetPlatform =
        ruleContext.attributes().get(PlatformRule.TARGET_PLATFORM_ATTR, Type.BOOLEAN);
    if (isHostPlatform && isTargetPlatform) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.HOST_PLATFORM_ATTR,
          "A single platform cannot have both host_platform and target_platform set.");
    } else if (isHostPlatform) {
      // Create default constraints based on the current host OS and CPU values.
      String cpuOption = ruleContext.getConfiguration().getHostCpu();
      autodetectConstraints(cpuOption, ruleContext, platformBuilder);
    } else if (isTargetPlatform) {
      // Create default constraints based on the current OS and CPU values.
      String cpuOption = ruleContext.getConfiguration().getCpu();
      autodetectConstraints(cpuOption, ruleContext, platformBuilder);
    }
  }

  private void autodetectConstraints(
      String cpuOption, RuleContext ruleContext, PlatformInfo.Builder platformBuilder) {

    Pair<CPU, OS> cpuValues = AutoCpuConverter.reverse(cpuOption);

    // Add the CPU.
    CPU cpu = cpuValues.getFirst();
    ImmutableList<ConstraintValueInfo> cpuConstraintValues =
        PlatformProviderUtils.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.CPU_CONSTRAINTS_ATTR));
    for (ConstraintValueInfo constraint : cpuConstraintValues) {
      if (cpu.getCanonicalName().equals(constraint.label().getName())) {
        platformBuilder.addConstraint(constraint);
        break;
      }
    }

    // Add the OS.
    OS os = cpuValues.getSecond();
    ImmutableList<ConstraintValueInfo> osConstraintValues =
        PlatformProviderUtils.constraintValues(
            ruleContext.getPrerequisites(PlatformRule.OS_CONSTRAINTS_ATTR));
    for (ConstraintValueInfo constraint : osConstraintValues) {
      if (os.getCanonicalName().equals(constraint.label().getName())) {
        platformBuilder.addConstraint(constraint);
        break;
      }
    }
  }
}
