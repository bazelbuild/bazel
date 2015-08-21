// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.java;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.DirectDependencyProvider;
import com.google.devtools.build.lib.rules.java.DirectDependencyProvider.Dependency;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaPrimaryClassProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Semantics for Bazel Java rules
 */
public class BazelJavaSemantics implements JavaSemantics {

  public static final BazelJavaSemantics INSTANCE = new BazelJavaSemantics();

  private static final Template STUB_SCRIPT =
      Template.forResource(BazelJavaSemantics.class, "java_stub_template.txt");

  public static final InstrumentationSpec GREEDY_COLLECTION_SPEC = new InstrumentationSpec(
      FileTypeSet.of(FileType.of(".sh"), JavaSemantics.JAVA_SOURCE),
      "srcs", "deps", "data");

  private static final String JAVABUILDER_CLASS_NAME =
      "com.google.devtools.build.buildjar.BazelJavaBuilder";

  private BazelJavaSemantics() {
  }

  private boolean isJavaBinaryOrJavaTest(RuleContext ruleContext) {
    String ruleClass = ruleContext.getRule().getRuleClass();
    return ruleClass.equals("java_binary") || ruleClass.equals("java_test");
  }

  @Override
  public void checkRule(RuleContext ruleContext, JavaCommon javaCommon) {
    if (isJavaBinaryOrJavaTest(ruleContext)) {
      checkMainClass(ruleContext, javaCommon);
    }
  }
  
  private String getMainClassInternal(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("main_class", Type.STRING)
        ? ruleContext.attributes().get("main_class", Type.STRING) : "";
  }

  private void checkMainClass(RuleContext ruleContext, JavaCommon javaCommon) {
    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
    String mainClass = getMainClassInternal(ruleContext);

    if (!createExecutable && !mainClass.isEmpty()) {
      ruleContext.ruleError("main class must not be specified when executable is not created");
    }

    if (createExecutable && mainClass.isEmpty()) {
      if (javaCommon.getSrcsArtifacts().isEmpty()) {
        ruleContext.ruleError(
            "need at least one of 'main_class', 'use_testrunner' or Java source files");
      }
      mainClass = javaCommon.determinePrimaryClass(javaCommon.getSrcsArtifacts());
      if (mainClass == null) {
        ruleContext.ruleError("cannot determine main class for launching "
                  + "(found neither a source file '" + ruleContext.getTarget().getName()
                  + ".java', nor a main_class attribute, and package name "
                  + "doesn't include 'java' or 'javatests')");
      }
    }
  }

  @Override
  public String getMainClass(RuleContext ruleContext, JavaCommon javaCommon) {
    checkMainClass(ruleContext, javaCommon);
    return getMainClassInternal(ruleContext);
  }

  @Override
  public ImmutableList<Artifact> collectResources(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined("resources", Type.LABEL_LIST)) {
      return ImmutableList.of();
    }

    return ruleContext.getPrerequisiteArtifacts("resources", Mode.TARGET).list();
  }

  @Override
  public Artifact createInstrumentationMetadataArtifact(
      RuleContext ruleContext, Artifact outputJar) {
    return null;
  }

  @Override
  public Iterable<Artifact> getInstrumentationJars(RuleContext context) {
    return ImmutableList.of();
  }

  @Override
  public void buildJavaCommandLine(Collection<Artifact> outputs, BuildConfiguration configuration,
      CustomCommandLine.Builder result) {
  }

  @Override
  public void createStubAction(RuleContext ruleContext, final JavaCommon javaCommon,
      List<String> jvmFlags, Artifact executable, String javaStartClass,
      String javaExecutable) {

    Preconditions.checkNotNull(jvmFlags);
    Preconditions.checkNotNull(executable);
    Preconditions.checkNotNull(javaStartClass);
    Preconditions.checkNotNull(javaExecutable);

    List<Substitution> arguments = new ArrayList<>();
    String workspacePrefix = ruleContext.getWorkspaceName();
    if (!workspacePrefix.isEmpty()) {
      workspacePrefix += "/";
    }
    arguments.add(Substitution.of("%workspace_prefix%", workspacePrefix));
    arguments.add(Substitution.of("%javabin%", javaExecutable));
    arguments.add(Substitution.of("%needs_runfiles%",
        ruleContext.getFragment(Jvm.class).getJavaExecutable().isAbsolute() ? "0" : "1"));
    arguments.add(new ComputedSubstitution("%classpath%") {
      @Override
      public String getValue() {
        StringBuilder buffer = new StringBuilder();
        Iterable<Artifact> jars = javaCommon.getRuntimeClasspath();
        appendRunfilesRelativeEntries(buffer, jars, ':');
        return buffer.toString();
      }
    });

    arguments.add(Substitution.of("%java_start_class%",
        ShellEscaper.escapeString(javaStartClass)));
    arguments.add(Substitution.ofSpaceSeparatedList("%jvm_flags%", jvmFlags));

    ruleContext.registerAction(new TemplateExpansionAction(
        ruleContext.getActionOwner(), executable, STUB_SCRIPT, arguments, true));
  }

  /**
   * Builds a class path by concatenating the root relative paths of the artifacts separated by the
   * delimiter. Each relative path entry is prepended with "${RUNPATH}" which will be expanded by
   * the stub script at runtime, to either "${JAVA_RUNFILES}/" or if we are lucky, the empty
   * string.
   *
   * @param buffer the buffer to use for concatenating the entries
   * @param artifacts the entries to concatenate in the buffer
   * @param delimiter the delimiter character to separate the entries
   */
  private static void appendRunfilesRelativeEntries(StringBuilder buffer,
      Iterable<Artifact> artifacts, char delimiter) {
    for (Artifact artifact : artifacts) {
      if (buffer.length() > 0) {
        buffer.append(delimiter);
      }
      buffer.append("${RUNPATH}");
      buffer.append(artifact.getRootRelativePath().getPathString());
    }
  }

  @Override
  public void addRunfilesForBinary(RuleContext ruleContext, Artifact launcher,
      Runfiles.Builder runfilesBuilder) {
  }

  @Override
  public void addRunfilesForLibrary(RuleContext ruleContext, Runfiles.Builder runfilesBuilder) {
  }

  @Override
  public void collectTargetsTreatedAsDeps(
      RuleContext ruleContext, ImmutableList.Builder<TransitiveInfoCollection> builder) {
  }

  @Override
  public InstrumentationSpec getCoverageInstrumentationSpec() {
    return GREEDY_COLLECTION_SPEC.withAttributes("srcs", "deps", "data", "exports", "runtime_deps");
  }

  @Override
  public Iterable<String> getExtraJavacOpts(RuleContext ruleContext) {
    return ImmutableList.<String>of();
  }

  @Override
  public void addProviders(RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact classJar,
      Artifact srcJar,
      Artifact genJar,
      Artifact gensrcJar,
      ImmutableMap<Artifact, Artifact> compilationToRuntimeJarMap,
      JavaCompilationHelper helper,
      NestedSetBuilder<Artifact> filesBuilder,
      RuleConfiguredTargetBuilder ruleBuilder) {
    if (!isJavaBinaryOrJavaTest(ruleContext)) {
      Artifact outputDepsProto = helper.getOutputDepsProtoArtifact();
      if (outputDepsProto != null && helper.getStrictJavaDeps() != StrictDepsMode.OFF) {
        ImmutableList<Dependency> strictDependencies =
            javaCommon.computeStrictDepsFromJavaAttributes(helper.getAttributes());
        ruleBuilder.add(DirectDependencyProvider.class,
            new DirectDependencyProvider(strictDependencies));
      }
    } else {
      boolean createExec = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
      ruleBuilder.add(JavaPrimaryClassProvider.class, 
          new JavaPrimaryClassProvider(createExec ? getMainClassInternal(ruleContext) : null));
    }
  }

  
  @Override
  public Iterable<String> getJvmFlags(RuleContext ruleContext, JavaCommon javaCommon,
      Artifact launcher, List<String> userJvmFlags) {
    return userJvmFlags;
  }

  @Override
  public String addCoverageSupport(JavaCompilationHelper helper,
      JavaTargetAttributes.Builder attributes,
      Artifact executable, Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder, String mainClass) {
    return mainClass;
  }

  @Override
  public boolean useStrictJavaDeps(BuildConfiguration configuration) {
    return true;
  }

  @Override
  public CustomCommandLine buildSingleJarCommandLine(BuildConfiguration configuration,
      Artifact output, String mainClass, ImmutableList<String> manifestLines,
      Iterable<Artifact> buildInfoFiles, ImmutableList<Artifact> resources,
      Iterable<Artifact> classpath, boolean includeBuildData,
      Compression compression, Artifact launcher) {
    return DeployArchiveBuilder.defaultSingleJarCommandLine(output, mainClass, manifestLines,
        buildInfoFiles, resources, classpath, includeBuildData, compression, launcher).build();
  }

  @Override
  public Collection<Artifact> translate(RuleContext ruleContext, JavaConfiguration javaConfig,
      List<Artifact> messages) {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public Artifact getLauncher(RuleContext ruleContext, JavaCommon common,
      DeployArchiveBuilder deployArchiveBuilder, Runfiles.Builder runfilesBuilder,
      List<String> jvmFlags, JavaTargetAttributes.Builder attributesBuilder, boolean shouldStrip) {
    return JavaHelper.launcherArtifactForTarget(this, ruleContext);
  }

  @Override
  public void addDependenciesForRunfiles(RuleContext ruleContext, Runfiles.Builder builder) {
  }

  @Override
  public boolean forceUseJavaLauncherTarget(RuleContext ruleContext) {
    return false;
  }

  @Override
  public void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder,
      Artifact srcArtifact) {
  }

  @Override
  public void commonDependencyProcessing(RuleContext ruleContext,
      JavaTargetAttributes.Builder attributes,
      Collection<? extends TransitiveInfoCollection> deps) {
  }

  @Override
  public PathFragment getJavaResourcePath(PathFragment path) {
    // Look for src/.../resources to match Maven repository structure.
    for (int i = 0; i < path.segmentCount() - 2; ++i) {
      if (path.getSegment(i).equals("src") && path.getSegment(i + 2).equals("resources")) {
        return path.subFragment(i + 3, path.segmentCount());
      }
    }
    PathFragment javaPath = JavaUtil.getJavaPath(path);
    return javaPath == null ? path : javaPath;
  }

  @Override
  public List<String> getExtraArguments(RuleContext ruleContext, JavaCommon javaCommon) {
    if (ruleContext.getRule().getRuleClass().equals("java_test")) {
      if (ruleContext.getConfiguration().getTestArguments().isEmpty()
          && !ruleContext.attributes().isAttributeValueExplicitlySpecified("args")) {
        ImmutableList.Builder<String> builder = ImmutableList.builder();
        for (Artifact artifact : javaCommon.getSrcsArtifacts()) {
          PathFragment path = artifact.getRootRelativePath();
          String className = JavaUtil.getJavaFullClassname(FileSystemUtils.removeExtension(path));
          if (className != null) {
            builder.add(className);
          }
        }
        return builder.build();
      }
    }
    return ImmutableList.<String>of();
  }

  @Override
  public String getJavaBuilderMainClass() {
    return JAVABUILDER_CLASS_NAME;
  }
}
