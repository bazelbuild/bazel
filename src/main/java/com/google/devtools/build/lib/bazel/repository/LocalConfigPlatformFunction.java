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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Create a local repository that describes the auto-detected host platform. */
public class LocalConfigPlatformFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Environment env, FileSystem fileSystem,
      Rule rule) {
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
      Map<String, String> markerData)
      throws RepositoryFunctionException {

    CPU hostCpu = CPU.getCurrent();
    OS hostOs = OS.getCurrent();

    try {
      outputDirectory.createDirectoryAndParents();
      RepositoryFunction.writeFile(
          outputDirectory, "WORKSPACE", workspaceFileContent(rule.getName()));
      RepositoryFunction.writeFile(
          outputDirectory, "BUILD.bazel", buildFileContent(rule.getName()));
      RepositoryFunction.writeFile(
          outputDirectory, "constraints.bzl", constraintFileContent(hostCpu, hostOs));
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not create content for " + rule.getName() + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }

    // Return the needed info.
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @Nullable
  static String cpuToConstraint(CPU cpu) {
    switch (cpu) {
      case X86_32:
        return "@bazel_tools//platforms:x86_32";
      case X86_64:
        return "@bazel_tools//platforms:x86_64";
      case PPC:
        return "@bazel_tools//platforms:ppc";
      case ARM:
        return "@bazel_tools//platforms:arm";
      case AARCH64:
        return "@bazel_tools//platforms:aarch64";
      case S390X:
        return "@bazel_tools//platforms:s390x";
      default:
        // Unknown, so skip it.
        return null;
    }
  }

  @Nullable
  static String osToConstraint(OS os) {
    switch (os) {
      case DARWIN:
        return "@bazel_tools//platforms:osx";
      case FREEBSD:
        return "@bazel_tools//platforms:freebsd";
      case LINUX:
        return "@bazel_tools//platforms:linux";
      case WINDOWS:
        return "@bazel_tools//platforms:windows";
      default:
        // Unknown, so skip it.
        return null;
    }
  }

  private static String workspaceFileContent(String repositoryName) {
    return format(
        ImmutableList.of(
            "# DO NOT EDIT: automatically generated WORKSPACE file for local_config_platforms",
            "workspace(name = \"%s\")"),
        repositoryName);
  }

  private static String buildFileContent(String repositoryName) {
    return format(
        ImmutableList.of(
            "# DO NOT EDIT: automatically generated BUILD file for local_config_platforms",
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
        "# DO NOT EDIT: automatically generated constraints list for local_config_platforms");
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
