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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;

/** Builds the action to package the resources for a Java rule into a jar. */
public class ResourceJarActionBuilder {
  private Artifact outputJar;
  private Map<PathFragment, Artifact> resources = ImmutableMap.of();
  private NestedSet<Artifact> resourceJars = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private List<Artifact> classpathResources = ImmutableList.of();
  private List<Artifact> messages = ImmutableList.of();
  private JavaToolchainProvider javaToolchain;
  private NestedSet<Artifact> javabase;

  public ResourceJarActionBuilder setOutputJar(Artifact outputJar) {
    this.outputJar = outputJar;
    return this;
  }

  public ResourceJarActionBuilder setClasspathResources(
      ImmutableList<Artifact> classpathResources) {
    this.classpathResources = classpathResources;
    return this;
  }

  public ResourceJarActionBuilder setResources(Map<PathFragment, Artifact> resources) {
    this.resources = resources;
    return this;
  }

  public ResourceJarActionBuilder setResourceJars(NestedSet<Artifact> resourceJars) {
    this.resourceJars = resourceJars;
    return this;
  }

  public ResourceJarActionBuilder setTranslations(ImmutableList<Artifact> translations) {
    this.messages = translations;
    return this;
  }

  public ResourceJarActionBuilder setJavaToolchain(JavaToolchainProvider javaToolchain) {
    this.javaToolchain = javaToolchain;
    return this;
  }

  public ResourceJarActionBuilder setJavabase(NestedSet<Artifact> javabase) {
    this.javabase = javabase;
    return this;
  }

  public void build(JavaSemantics semantics, RuleContext ruleContext) {
    checkNotNull(outputJar, "outputJar must not be null");
    checkNotNull(javabase, "javabase must not be null");
    checkNotNull(javaToolchain, "javaToolchain must not be null");

    Artifact singleJar = javaToolchain.getSingleJar();
    if (singleJar == null) {
      singleJar = ruleContext.getPrerequisiteArtifact("$singlejar", Mode.HOST);
    }
    SpawnAction.Builder builder = new SpawnAction.Builder();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              javaToolchain.getJvmOptions())
          .addTransitiveInputs(javabase);
    } else {
      builder.setExecutable(singleJar);
    }
    CustomCommandLine.Builder command =
        CustomCommandLine.builder()
            .add("--normalize")
            .add("--dont_change_compression")
            .add("--exclude_build_data")
            .addExecPath("--output", outputJar);
    if (!resourceJars.isEmpty()) {
      command.add("--sources").addExecPaths(resourceJars);
    }
    if (!resources.isEmpty() || !messages.isEmpty()) {
      command.add("--resources");
      for (Map.Entry<PathFragment, Artifact> resource : resources.entrySet()) {
        addAsResourcePrefixedExecPath(resource.getKey(), resource.getValue(), command);
      }
      for (Artifact message : messages) {
        addAsResourcePrefixedExecPath(
            semantics.getDefaultJavaResourcePath(message.getRootRelativePath()), message, command);
      }
    }
    if (!classpathResources.isEmpty()) {
      command.add("--classpath_resources").addExecPaths(classpathResources);
    }
    // TODO(b/37444705): remove this logic and always call useParameterFile once the bug is fixed
    // Most resource jar actions are very small and expanding the argument list for
    // ParamFileHelper#getParamsFileMaybe is expensive, so avoid doing that work if
    // we definitely don't need a params file.
    // This heuristic could be much more aggressive, but we don't ever want to skip
    // the params file in situations where it is required for --min_param_file_size.
    if (sizeGreaterThanOrEqual(
            Iterables.concat(messages, resources.values(), resourceJars, classpathResources), 10)
        || ruleContext.getConfiguration().getMinParamFileSize() < 10000) {
      builder.useParameterFile(ParameterFileType.SHELL_QUOTED);
    }
    ruleContext.registerAction(
        builder
            .addOutput(outputJar)
            .addInputs(messages)
            .addInputs(resources.values())
            .addTransitiveInputs(resourceJars)
            .addInputs(classpathResources)
            .setCommandLine(command.build())
            .setProgressMessage("Building Java resource jar")
            .setMnemonic("JavaResourceJar")
            .build(ruleContext));
  }

  boolean sizeGreaterThanOrEqual(Iterable<?> elements, int size) {
    return Streams.stream(elements).limit(size).count() == size;
  }

  private static void addAsResourcePrefixedExecPath(
      PathFragment resourcePath, Artifact artifact, CustomCommandLine.Builder builder) {
    PathFragment execPath = artifact.getExecPath();
    if (execPath.equals(resourcePath)) {
      builder.addPaths("%s", resourcePath);
    } else {
      builder.addPaths("%s:%s", execPath, resourcePath);
    }
  }
}
