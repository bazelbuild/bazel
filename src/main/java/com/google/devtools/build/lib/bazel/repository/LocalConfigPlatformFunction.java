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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ResolvedEvent;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Create a local repository that describes the auto-detected host platform. */
public class LocalConfigPlatformFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return LocalConfigPlatformRule.class;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<String, String> markerData,
      SkyKey key)
      throws RepositoryFunctionException {

    CPU hostCpu = CPU.getCurrent();
    OS hostOs = OS.getCurrent();

    String name = rule.getName();
    try {
      outputDirectory.createDirectoryAndParents();
      RepositoryFunction.writeFile(outputDirectory, "WORKSPACE", workspaceFileContent(name));
      RepositoryFunction.writeFile(outputDirectory, "BUILD.bazel", buildFileContent(name));
      RepositoryFunction.writeFile(
          outputDirectory, "constraints.bzl", constraintFileContent(hostCpu, hostOs));
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not create content for " + name + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }

    // Save in the resolved repository file.
    env.getListener()
        .post(
            new ResolvedEvent() {
              @Override
              public String getName() {
                return name;
              }

              @Override
              public Object getResolvedInformation() {
                String repr = String.format("local_config_platform(name = '%s')", name);
                return ImmutableMap.<String, Object>builder()
                    .put(ResolvedHashesFunction.ORIGINAL_RULE_CLASS, LocalConfigPlatformRule.NAME)
                    .put(
                        ResolvedHashesFunction.ORIGINAL_ATTRIBUTES,
                        ImmutableMap.<String, Object>builder().put("name", name).build())
                    .put(ResolvedHashesFunction.NATIVE, repr)
                    .build();
              }
            });

    // Return the needed info.
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @Nullable
  static String cpuToConstraint(CPU cpu) {
    switch (cpu) {
      case X86_32:
        return "@platforms//cpu:x86_32";
      case X86_64:
        return "@platforms//cpu:x86_64";
      case PPC:
        return "@platforms//cpu:ppc";
      case ARM:
        return "@platforms//cpu:arm";
      case AARCH64:
        return "@platforms//cpu:aarch64";
      case S390X:
        return "@platforms//cpu:s390x";
      default:
        // Unknown, so skip it.
        return null;
    }
  }

  @Nullable
  static String osToConstraint(OS os) {
    switch (os) {
      case DARWIN:
        return "@platforms//os:osx";
      case FREEBSD:
        return "@platforms//os:freebsd";
      case OPENBSD:
        return "@platforms//os:openbsd";
      case LINUX:
        return "@platforms//os:linux";
      case WINDOWS:
        return "@platforms//os:windows";
      default:
        // Unknown, so skip it.
        return null;
    }
  }

  private static String workspaceFileContent(String repositoryName) {
    return format(
        ImmutableList.of(
            "# DO NOT EDIT: automatically generated WORKSPACE file for local_config_platform",
            "workspace(name = \"%s\")"),
        repositoryName);
  }

  private static String buildFileContent(String repositoryName) {
    return format(
        ImmutableList.of(
            "# DO NOT EDIT: automatically generated BUILD file for local_config_platform",
            "package(default_visibility = ['//visibility:public'])",
            "load(':constraints.bzl', 'HOST_CONSTRAINTS')",
            "platform(name = 'host',",
            "  # Auto-detected host platform constraints.",
            "  constraint_values = HOST_CONSTRAINTS,",
            ")"),
        repositoryName);
  }

  private static String constraintFileContent(CPU hostCpu, OS hostOs) {
    List<String> contents = new ArrayList<>();
    contents.add(
        "# DO NOT EDIT: automatically generated constraints list for local_config_platform");
    contents.add("# Auto-detected host platform constraints.");
    contents.add("HOST_CONSTRAINTS = [");

    String cpuConstraint = cpuToConstraint(hostCpu);
    if (cpuConstraint != null) {
      contents.add("  '" + cpuConstraint + "',");
    }
    String osConstraint = osToConstraint(hostOs);
    if (osConstraint != null) {
      contents.add("  '" + osConstraint + "',");
    }
    contents.add("]");

    return format(contents);
  }

  private static String format(List<String> lines, Object... params) {
    // Add a newline between each line, and also after the final line.
    String content = lines.stream().collect(Collectors.joining("\n", "", "\n"));
    return String.format(content, params);
  }
}
