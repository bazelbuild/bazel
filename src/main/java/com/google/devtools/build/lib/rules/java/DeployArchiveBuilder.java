// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Utility for configuring an action to generate a deploy archive. */
public class DeployArchiveBuilder {
  /**
   * Memory consumption of SingleJar is about 250 bytes per entry in the output file. Unfortunately,
   * the JVM tends to kill the process with an OOM long before we're at the limit. In the most
   * recent example, 400 MB of memory was enough for about 500,000 entries.
   */
  private static final int SINGLEJAR_MEMORY_MB = 1600;

  private static final String SINGLEJAR_MAX_MEMORY = "-Xmx" + SINGLEJAR_MEMORY_MB + "m";

  private static final ResourceSet DEPLOY_ACTION_RESOURCE_SET =
      ResourceSet.createWithRamCpu(/*memoryMb = */ SINGLEJAR_MEMORY_MB, /*cpuUsage = */ 1);

  private final RuleContext ruleContext;

  private final IterablesChain.Builder<Artifact> runtimeJarsBuilder = IterablesChain.builder();

  private final JavaSemantics semantics;

  private JavaTargetAttributes attributes;
  private boolean includeBuildData;
  private Compression compression = Compression.UNCOMPRESSED;
  @Nullable private Artifact runfilesMiddleman;
  private Artifact outputJar;
  @Nullable private String javaStartClass;
  private ImmutableList<String> deployManifestLines = ImmutableList.of();
  @Nullable private Artifact launcher;
  @Nullable private Function<Artifact, Artifact> derivedJars = null;
  private boolean checkDesugarDeps;
  private OneVersionEnforcementLevel oneVersionEnforcementLevel = OneVersionEnforcementLevel.OFF;
  @Nullable private Artifact oneVersionWhitelistArtifact;

  /** Type of compression to apply to output archive. */
  public enum Compression {

    /** Output should be compressed */
    COMPRESSED,

    /** Output should not be compressed */
    UNCOMPRESSED;
  }

  /** Creates a builder using the configuration of the rule as the action configuration. */
  public DeployArchiveBuilder(JavaSemantics semantics, RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.semantics = semantics;
  }

  /** Sets the processed attributes of the rule generating the deploy archive. */
  public DeployArchiveBuilder setAttributes(JavaTargetAttributes attributes) {
    this.attributes = attributes;
    return this;
  }

  /** Sets whether to include build-data.properties in the deploy archive. */
  public DeployArchiveBuilder setIncludeBuildData(boolean includeBuildData) {
    this.includeBuildData = includeBuildData;
    return this;
  }

  /** Sets whether to enable compression of the output deploy archive. */
  public DeployArchiveBuilder setCompression(Compression compress) {
    this.compression = Preconditions.checkNotNull(compress);
    return this;
  }

  /**
   * Sets additional dependencies to be added to the action that creates the deploy jar so that we
   * force the runtime dependencies to be built.
   */
  public DeployArchiveBuilder setRunfilesMiddleman(@Nullable Artifact runfilesMiddleman) {
    this.runfilesMiddleman = runfilesMiddleman;
    return this;
  }

  /** Sets the artifact to create with the action. */
  public DeployArchiveBuilder setOutputJar(Artifact outputJar) {
    this.outputJar = Preconditions.checkNotNull(outputJar);
    return this;
  }

  /** Sets the class to launch the Java application. */
  public DeployArchiveBuilder setJavaStartClass(@Nullable String javaStartClass) {
    this.javaStartClass = javaStartClass;
    return this;
  }

  /** Adds additional jars that should be on the classpath at runtime. */
  public DeployArchiveBuilder addRuntimeJars(Iterable<Artifact> jars) {
    this.runtimeJarsBuilder.add(jars);
    return this;
  }

  /** Sets the list of extra lines to add to the archive's MANIFEST.MF file. */
  public DeployArchiveBuilder setDeployManifestLines(Iterable<String> deployManifestLines) {
    this.deployManifestLines = ImmutableList.copyOf(deployManifestLines);
    return this;
  }

  /** Sets the optional launcher to be used as the executable for this deploy JAR */
  public DeployArchiveBuilder setLauncher(@Nullable Artifact launcher) {
    this.launcher = launcher;
    return this;
  }

  public DeployArchiveBuilder setDerivedJarFunction(Function<Artifact, Artifact> derivedJars) {
    this.derivedJars = derivedJars;
    return this;
  }

  /** Whether singlejar should process META-INF/desugar_deps files and fail upon inconsistencies. */
  public DeployArchiveBuilder setCheckDesugarDeps(boolean checkDesugarDeps) {
    this.checkDesugarDeps = checkDesugarDeps;
    return this;
  }

  /** Whether or not singlejar would attempt to enforce one version of java classes in the jar */
  public DeployArchiveBuilder setOneVersionEnforcementLevel(
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      @Nullable Artifact oneVersionWhitelistArtifact) {
    this.oneVersionEnforcementLevel = oneVersionEnforcementLevel;
    this.oneVersionWhitelistArtifact = oneVersionWhitelistArtifact;
    return this;
  }

  public static CustomCommandLine.Builder defaultSingleJarCommandLineWithoutOneVersion(
      Artifact outputJar,
      String javaMainClass,
      ImmutableList<String> deployManifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> classpathResources,
      NestedSet<Artifact> runtimeClasspath,
      boolean includeBuildData,
      Compression compress,
      Artifact launcher,
      boolean usingNativeSinglejar) {
    return defaultSingleJarCommandLine(
        outputJar,
        javaMainClass,
        deployManifestLines,
        buildInfoFiles,
        classpathResources,
        runtimeClasspath,
        includeBuildData,
        compress,
        launcher,
        usingNativeSinglejar,
        OneVersionEnforcementLevel.OFF,
        null);
  }

  public static CustomCommandLine.Builder defaultSingleJarCommandLine(
      Artifact outputJar,
      String javaMainClass,
      ImmutableList<String> deployManifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> classpathResources,
      NestedSet<Artifact> runtimeClasspath,
      boolean includeBuildData,
      Compression compress,
      Artifact launcher,
      boolean usingNativeSinglejar,
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      @Nullable Artifact oneVersionWhitelistArtifact) {

    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    if (compress == Compression.COMPRESSED) {
      args.add("--compression");
    }
    args.add("--normalize");
    if (javaMainClass != null) {
      args.add("--main_class", javaMainClass);
    }

    if (!deployManifestLines.isEmpty()) {
      args.add("--deploy_manifest_lines");
      args.addAll(deployManifestLines);
    }

    if (buildInfoFiles != null) {
      for (Artifact artifact : buildInfoFiles) {
        args.addExecPath("--build_info_file", artifact);
      }
    }
    if (!includeBuildData) {
      args.add("--exclude_build_data");
    }
    if (launcher != null) {
      args.addExecPath("--java_launcher", launcher);
    }

    args.addExecPaths("--classpath_resources", classpathResources);
    if (runtimeClasspath != null) {
      if (usingNativeSinglejar) {
        args.addAll(
            "--sources", OneVersionCheckActionBuilder.jarAndTargetVectorArg(runtimeClasspath));
      } else {
        args.addExecPaths("--sources", runtimeClasspath);
      }
    }
    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF && usingNativeSinglejar) {
      args.add("--enforce_one_version");
      // RuleErrors should have been added in Builder.build() before this command
      // line is invoked.
      Preconditions.checkNotNull(oneVersionWhitelistArtifact);
      args.addExecPath("--one_version_whitelist", oneVersionWhitelistArtifact);
      if (oneVersionEnforcementLevel == OneVersionEnforcementLevel.WARNING) {
        args.add("--succeed_on_found_violations");
      }
    }
    return args;
  }

  private static NestedSet<Artifact> getArchiveInputs(
      JavaTargetAttributes attributes,
      NestedSet<Artifact> runtimeClasspathForArchive,
      @Nullable Function<Artifact, Artifact> derivedJarFunction) {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    if (derivedJarFunction != null) {
      inputs.addAll(
          runtimeClasspathForArchive.toList().stream()
              .map(derivedJarFunction)
              .collect(toImmutableList()));
    } else {
      inputs.addTransitive(runtimeClasspathForArchive);
    }
    // TODO(bazel-team): Remove?  Resources not used as input to singlejar action
    inputs.addAll(attributes.getResources().values());
    inputs.addAll(attributes.getClassPathResources());
    return inputs.build();
  }

  /** Builds the action as configured. */
  public void build() throws InterruptedException {
    ImmutableList<Artifact> classpathResources = attributes.getClassPathResources();
    Set<String> classPathResourceNames = new HashSet<>();
    for (Artifact artifact : classpathResources) {
      String name = artifact.getExecPath().getBaseName();
      if (!classPathResourceNames.add(name)) {
        ruleContext.attributeError(
            "classpath_resources",
            "entries must have different file names (duplicate: " + name + ")");
        return;
      }
    }

    Iterable<Artifact> runtimeJars = runtimeJarsBuilder.build();

    NestedSet<Artifact> runtimeClasspathForArchive = attributes.getRuntimeClassPathForArchive();

    // TODO(kmb): Consider not using getArchiveInputs, specifically because we don't want/need to
    // transform anything but the runtimeClasspath and b/c we currently do it twice here and below
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.addTransitive(getArchiveInputs(attributes, runtimeClasspathForArchive, derivedJars));

    if (derivedJars != null) {
      inputs.addAll(Streams.stream(runtimeJars).map(derivedJars).collect(toImmutableList()));
    } else {
      inputs.addAll(runtimeJars);
    }
    if (runfilesMiddleman != null) {
      inputs.add(runfilesMiddleman);
    }

    ImmutableList<Artifact> buildInfoArtifacts = ruleContext.getBuildInfo(JavaBuildInfoFactory.KEY);
    inputs.addAll(buildInfoArtifacts);

    NestedSetBuilder<Artifact> runtimeClasspath = NestedSetBuilder.stableOrder();
    if (derivedJars != null) {
      runtimeClasspath.addAll(
          Iterables.transform(
              Iterables.concat(runtimeJars, runtimeClasspathForArchive.toList()), derivedJars));
    } else {
      runtimeClasspath.addAll(runtimeJars);
      runtimeClasspath.addTransitive(runtimeClasspathForArchive);
    }

    if (launcher != null) {
      inputs.add(launcher);
    }

    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF) {
      if (oneVersionWhitelistArtifact == null) {
        OneVersionCheckActionBuilder.addRuleErrorForMissingArtifacts(
            ruleContext, JavaToolchainProvider.from(ruleContext));
        return;
      }
      inputs.add(oneVersionWhitelistArtifact);
    }
    // If singlejar's name ends with .jar, it is Java application, otherwise it is native.
    // TODO(asmundak): once https://github.com/bazelbuild/bazel/issues/2241 is fixed (that is,
    // the native singlejar is used on windows) remove support for the Java implementation
    Artifact singlejar = JavaToolchainProvider.from(ruleContext).getSingleJar();
    boolean usingNativeSinglejar = !singlejar.getFilename().endsWith(".jar");

    CommandLine commandLine =
        semantics.buildSingleJarCommandLine(
            CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext)
                .getToolchainIdentifier(),
            outputJar,
            javaStartClass,
            deployManifestLines,
            buildInfoArtifacts,
            classpathResources,
            runtimeClasspath.build(),
            includeBuildData,
            compression,
            launcher,
            usingNativeSinglejar,
            oneVersionEnforcementLevel,
            oneVersionWhitelistArtifact);
    if (checkDesugarDeps) {
      commandLine = CommandLine.concat(commandLine, ImmutableList.of("--check_desugar_deps"));
    }

    List<String> jvmArgs = ImmutableList.of(SINGLEJAR_MAX_MEMORY);

    if (!usingNativeSinglejar) {
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addTransitiveInputs(inputs.build())
              .addTransitiveInputs(JavaRuntimeInfo.forHost(ruleContext).javaBaseInputsMiddleman())
              .addOutput(outputJar)
              .setResources(DEPLOY_ACTION_RESOURCE_SET)
              .setJarExecutable(JavaCommon.getHostJavaExecutable(ruleContext), singlejar, jvmArgs)
              .addCommandLine(
                  commandLine,
                  ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
              .setProgressMessage("Building deploy jar %s", outputJar.prettyPrint())
              .setMnemonic("JavaDeployJar")
              .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED)
              .build(ruleContext));
    } else {
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addTransitiveInputs(inputs.build())
              .addOutput(outputJar)
              .setResources(DEPLOY_ACTION_RESOURCE_SET)
              .setExecutable(singlejar)
              .addCommandLine(
                  commandLine,
                  ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
              .setProgressMessage("Building deploy jar %s", outputJar.prettyPrint())
              .setMnemonic("JavaDeployJar")
              .build(ruleContext));
    }
  }
}
