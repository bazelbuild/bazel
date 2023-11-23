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
import static java.util.Objects.requireNonNull;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.HashSet;
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

  private static final ResourceSet DEPLOY_ACTION_RESOURCE_SET =
      ResourceSet.createWithRamCpu(/* memoryMb= */ SINGLEJAR_MEMORY_MB, /* cpu= */ 1);

  private final RuleContext ruleContext;

  private final NestedSetBuilder<Artifact> runtimeJarsBuilder = NestedSetBuilder.stableOrder();

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
  @Nullable private Artifact oneVersionAllowlistArtifact;
  @Nullable private Artifact sharedArchive;
  private boolean multiReleaseDeployJars;
  @Nullable private PathFragment javaHome;
  @Nullable private Artifact libModules;
  private NestedSet<Artifact> hermeticInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private NestedSet<String> addExports;
  private NestedSet<String> addOpens;

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
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setAttributes(JavaTargetAttributes attributes) {
    this.attributes = attributes;
    return this;
  }

  /** Sets whether to include build-data.properties in the deploy archive. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setIncludeBuildData(boolean includeBuildData) {
    this.includeBuildData = includeBuildData;
    return this;
  }

  /** Sets whether to enable compression of the output deploy archive. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setCompression(Compression compress) {
    this.compression = Preconditions.checkNotNull(compress);
    return this;
  }

  /**
   * Sets additional dependencies to be added to the action that creates the deploy jar so that we
   * force the runtime dependencies to be built.
   */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setRunfilesMiddleman(@Nullable Artifact runfilesMiddleman) {
    this.runfilesMiddleman = runfilesMiddleman;
    return this;
  }

  /** Sets the artifact to create with the action. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setOutputJar(Artifact outputJar) {
    this.outputJar = Preconditions.checkNotNull(outputJar);
    return this;
  }

  /** Sets the class to launch the Java application. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setJavaStartClass(@Nullable String javaStartClass) {
    this.javaStartClass = javaStartClass;
    return this;
  }

  /** Adds additional jars that should be on the classpath at runtime. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder addRuntimeJars(NestedSet<Artifact> jars) {
    this.runtimeJarsBuilder.addTransitive(jars);
    return this;
  }

  /** Adds additional jars that should be on the classpath at runtime. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder addRuntimeJars(Iterable<Artifact> jars) {
    this.runtimeJarsBuilder.addAll(jars);
    return this;
  }

  /** Sets the list of extra lines to add to the archive's MANIFEST.MF file. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setDeployManifestLines(ImmutableList<String> deployManifestLines) {
    this.deployManifestLines = Preconditions.checkNotNull(deployManifestLines);
    return this;
  }

  /** Sets the optional launcher to be used as the executable for this deploy JAR */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setLauncher(@Nullable Artifact launcher) {
    this.launcher = launcher;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setDerivedJarFunction(Function<Artifact, Artifact> derivedJars) {
    this.derivedJars = derivedJars;
    return this;
  }

  /** Whether singlejar should process META-INF/desugar_deps files and fail upon inconsistencies. */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setCheckDesugarDeps(boolean checkDesugarDeps) {
    this.checkDesugarDeps = checkDesugarDeps;
    return this;
  }

  /** Whether or not singlejar would attempt to enforce one version of java classes in the jar */
  @CanIgnoreReturnValue
  public DeployArchiveBuilder setOneVersionEnforcementLevel(
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      @Nullable Artifact oneVersionAllowlistArtifact) {
    this.oneVersionEnforcementLevel = oneVersionEnforcementLevel;
    this.oneVersionAllowlistArtifact = oneVersionAllowlistArtifact;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setMultiReleaseDeployJars(boolean multiReleaseDeployJars) {
    this.multiReleaseDeployJars = multiReleaseDeployJars;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setSharedArchive(@Nullable Artifact sharedArchive) {
    this.sharedArchive = sharedArchive;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setJavaHome(PathFragment javaHome) {
    this.javaHome = requireNonNull(javaHome);
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setLibModules(@Nullable Artifact libModules) {
    this.libModules = libModules;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setHermeticInputs(NestedSet<Artifact> hermeticInputs) {
    this.hermeticInputs = requireNonNull(hermeticInputs);
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setAddExports(NestedSet<String> addExports) {
    this.addExports = addExports;
    return this;
  }

  @CanIgnoreReturnValue
  public DeployArchiveBuilder setAddOpens(NestedSet<String> addOpens) {
    this.addOpens = addOpens;
    return this;
  }

  public static CustomCommandLine.Builder defaultSingleJarCommandLineWithoutOneVersion(
      Artifact outputJar,
      Label label,
      String javaMainClass,
      ImmutableList<String> deployManifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> classpathResources,
      NestedSet<Artifact> runtimeClasspath,
      boolean includeBuildData,
      Compression compress,
      Artifact launcher,
      boolean multiReleaseDeployJars,
      PathFragment javaHome,
      Artifact libModules,
      NestedSet<Artifact> hermeticInputs,
      NestedSet<String> addExports,
      NestedSet<String> addOpens) {
    return defaultSingleJarCommandLine(
        outputJar,
        label,
        javaMainClass,
        deployManifestLines,
        buildInfoFiles,
        classpathResources,
        runtimeClasspath,
        includeBuildData,
        compress,
        launcher,
        OneVersionEnforcementLevel.OFF,
        null,
        /* multiReleaseDeployJars= */ multiReleaseDeployJars,
        javaHome,
        libModules,
        hermeticInputs,
        addExports,
        addOpens);
  }

  public static CustomCommandLine.Builder defaultSingleJarCommandLine(
      Artifact outputJar,
      Label label,
      String javaMainClass,
      ImmutableList<String> deployManifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> classpathResources,
      NestedSet<Artifact> runtimeClasspath,
      boolean includeBuildData,
      Compression compress,
      Artifact launcher,
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      @Nullable Artifact oneVersionAllowlistArtifact,
      boolean multiReleaseDeployJars,
      PathFragment javaHome,
      Artifact libModules,
      NestedSet<Artifact> hermeticInputs,
      NestedSet<String> addExports,
      NestedSet<String> addOpens) {

    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    args.add("--build_target", label.getCanonicalForm());
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
      args.addAll(
          "--sources", OneVersionCheckActionBuilder.jarAndTargetVectorArg(runtimeClasspath));
    }
    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF
        && oneVersionAllowlistArtifact != null) {
      args.add("--enforce_one_version");
      args.addExecPath("--one_version_whitelist", oneVersionAllowlistArtifact);
      if (oneVersionEnforcementLevel == OneVersionEnforcementLevel.WARNING) {
        args.add("--succeed_on_found_violations");
      }
    }
    if (multiReleaseDeployJars) {
      args.add("--multi_release");
    }
    args.addPath("--hermetic_java_home", javaHome);
    args.addExecPath("--jdk_lib_modules", libModules);
    args.addExecPaths("--resources", hermeticInputs);
    args.addAll("--add_exports", addExports);
    args.addAll("--add_opens", addOpens);
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
  public void build() throws InterruptedException, RuleErrorException {
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

    NestedSet<Artifact> runtimeJars = runtimeJarsBuilder.build();

    NestedSet<Artifact> runtimeClasspathForArchive = attributes.getRuntimeClassPathForArchive();

    // TODO(kmb): Consider not using getArchiveInputs, specifically because we don't want/need to
    // transform anything but the runtimeClasspath and b/c we currently do it twice here and below
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    inputs.addTransitive(getArchiveInputs(attributes, runtimeClasspathForArchive, derivedJars));

    if (derivedJars != null) {
      inputs.addAll(Iterables.transform(runtimeJars.toList(), derivedJars));
    } else {
      inputs.addTransitive(runtimeJars);
    }
    if (runfilesMiddleman != null) {
      inputs.add(runfilesMiddleman);
    }
    ImmutableList<Artifact> buildInfoArtifacts = ImmutableList.of();
    int stamp = 0;
    if (ruleContext.attributes().has("stamp", Type.INTEGER)) {
      stamp = ruleContext.attributes().get("stamp", Type.INTEGER).toIntUnchecked();
    }
    if (ruleContext.attributes().has("stamp", BuildType.TRISTATE)) {
      stamp = ruleContext.attributes().get("stamp", BuildType.TRISTATE).toInt();
    }
    try {
      buildInfoArtifacts = semantics.getBuildInfo(ruleContext, stamp);
    } catch (RuleErrorException e) {
      throw new InterruptedException("Translating BuildInfo files failed: " + e);
    }
    inputs.addAll(buildInfoArtifacts);

    NestedSetBuilder<Artifact> runtimeClasspath = NestedSetBuilder.stableOrder();
    if (derivedJars != null) {
      runtimeClasspath.addAll(Iterables.transform(runtimeJars.toList(), derivedJars));
      runtimeClasspath.addAll(
          Iterables.transform(runtimeClasspathForArchive.toList(), derivedJars));
    } else {
      runtimeClasspath.addTransitive(runtimeJars);
      runtimeClasspath.addTransitive(runtimeClasspathForArchive);
    }

    if (launcher != null) {
      inputs.add(launcher);
    }

    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF
        && oneVersionAllowlistArtifact != null) {
      inputs.add(oneVersionAllowlistArtifact);
    }
    if (sharedArchive != null) {
      inputs.add(sharedArchive);
    }
    inputs.addTransitive(hermeticInputs);
    if (libModules != null) {
      inputs.add(libModules);
    }

    FilesToRunProvider singlejar = JavaToolchainProvider.from(ruleContext).getSingleJar();

    String toolchainIdentifier = null;
    try {
      toolchainIdentifier = CppHelper.getToolchain(ruleContext).getToolchainIdentifier();
    } catch (RuleErrorException e) {
      // Something went wrong loading the toolchain, which is an exceptional condition.
      throw new IllegalStateException("Unable to load cc toolchain", e);
    }
    CommandLine commandLine =
        semantics.buildSingleJarCommandLine(
            toolchainIdentifier,
            outputJar,
            ruleContext.getLabel(),
            javaStartClass,
            deployManifestLines,
            buildInfoArtifacts,
            classpathResources,
            runtimeClasspath.build(),
            includeBuildData,
            compression,
            launcher,
            oneVersionEnforcementLevel,
            oneVersionAllowlistArtifact,
            sharedArchive,
            /* multiReleaseDeployJars= */ multiReleaseDeployJars,
            javaHome,
            libModules,
            hermeticInputs,
            addExports,
            addOpens);
    if (checkDesugarDeps) {
      commandLine = CommandLine.concat(commandLine, ImmutableList.of("--check_desugar_deps"));
    }

    ImmutableMap.Builder<String, String> executionInfo = ImmutableMap.builder();
    executionInfo.putAll(
        TargetUtils.getExecutionInfo(ruleContext.getRule(), ruleContext.isAllowTagsPropagation()));

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutput(outputJar)
            .setResources(DEPLOY_ACTION_RESOURCE_SET)
            .setExecutable(singlejar)
            .addCommandLine(
                commandLine,
                ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).setUseAlways(true).build())
            .setProgressMessage("Building deploy jar %{output}")
            .setMnemonic("JavaDeployJar")
            .setExecutionInfo(executionInfo.buildOrThrow())
            .build(ruleContext));
  }
}
