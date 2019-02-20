// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.StandardSystemProperty.JAVA_SPECIFICATION_VERSION;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaToolchainSkylarkApiProviderApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;

/** Information about the JDK used by the <code>java_*</code> rules. */
@Immutable
@AutoCodec
public class JavaToolchainProvider extends ToolchainInfo
    implements JavaToolchainSkylarkApiProviderApi {

  /** Returns the Java Toolchain associated with the rule being analyzed or {@code null}. */
  public static JavaToolchainProvider from(RuleContext ruleContext) {
    TransitiveInfoCollection prerequisite =
        ruleContext.getPrerequisite(JavaRuleClasses.JAVA_TOOLCHAIN_ATTRIBUTE_NAME, Mode.TARGET);
    return from(prerequisite, ruleContext);
  }

  public static JavaToolchainProvider from(ProviderCollection collection) {
    return from(collection, null);
  }

  private static JavaToolchainProvider from(
      ProviderCollection collection, @Nullable RuleErrorConsumer errorConsumer) {
    ToolchainInfo toolchainInfo = collection.get(ToolchainInfo.PROVIDER);
    if (toolchainInfo instanceof JavaToolchainProvider) {
      return (JavaToolchainProvider) toolchainInfo;
    }
    if (errorConsumer != null) {
      errorConsumer.ruleError("The selected Java toolchain is not a JavaToolchainProvider");
    }
    return null;
  }

  public static JavaToolchainProvider create(
      Label label,
      ImmutableList<String> javacOptions,
      ImmutableList<String> jvmOptions,
      boolean javacSupportsWorkers,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> extclasspath,
      List<String> defaultJavacFlags,
      Artifact javac,
      NestedSet<Artifact> tools,
      FilesToRunProvider javaBuilder,
      @Nullable FilesToRunProvider headerCompiler,
      @Nullable FilesToRunProvider headerCompilerDirect,
      boolean forciblyDisableHeaderCompilation,
      Artifact singleJar,
      @Nullable Artifact oneVersion,
      @Nullable Artifact oneVersionWhitelist,
      Artifact genClass,
      @Nullable Artifact resourceJarBuilder,
      @Nullable Artifact timezoneData,
      FilesToRunProvider ijar,
      ImmutableListMultimap<String, String> compatibleJavacOptions,
      ImmutableList<JavaPackageConfigurationProvider> packageConfiguration,
      JavaSemantics javaSemantics) {
    return new JavaToolchainProvider(
        label,
        bootclasspath,
        extclasspath,
        javac,
        tools,
        javaBuilder,
        headerCompiler,
        headerCompilerDirect,
        forciblyDisableHeaderCompilation,
        singleJar,
        oneVersion,
        oneVersionWhitelist,
        genClass,
        resourceJarBuilder,
        timezoneData,
        ijar,
        compatibleJavacOptions,
        // merges the defaultJavacFlags from
        // {@link JavaConfiguration} with the flags from the {@code java_toolchain} rule.
        ImmutableList.<String>builder().addAll(javacOptions).addAll(defaultJavacFlags).build(),
        jvmOptions,
        javacSupportsWorkers,
        packageConfiguration,
        javaSemantics);
  }

  private final Label label;
  private final NestedSet<Artifact> bootclasspath;
  private final NestedSet<Artifact> extclasspath;
  private final Artifact javac;
  private final NestedSet<Artifact> tools;
  private final FilesToRunProvider javaBuilder;
  @Nullable private final FilesToRunProvider headerCompiler;
  @Nullable private final FilesToRunProvider headerCompilerDirect;
  private final boolean forciblyDisableHeaderCompilation;
  private final Artifact singleJar;
  @Nullable private final Artifact oneVersion;
  @Nullable private final Artifact oneVersionWhitelist;
  private final Artifact genClass;
  @Nullable private final Artifact resourceJarBuilder;
  @Nullable private final Artifact timezoneData;
  private final FilesToRunProvider ijar;
  private final ImmutableListMultimap<String, String> compatibleJavacOptions;
  private final ImmutableList<String> javacOptions;
  private final ImmutableList<String> jvmOptions;
  private final boolean javacSupportsWorkers;
  private final ImmutableList<JavaPackageConfigurationProvider> packageConfiguration;
  private final JavaSemantics javaSemantics;

  @VisibleForSerialization
  JavaToolchainProvider(
      Label label,
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> extclasspath,
      Artifact javac,
      NestedSet<Artifact> tools,
      FilesToRunProvider javaBuilder,
      @Nullable FilesToRunProvider headerCompiler,
      @Nullable FilesToRunProvider headerCompilerDirect,
      boolean forciblyDisableHeaderCompilation,
      Artifact singleJar,
      @Nullable Artifact oneVersion,
      @Nullable Artifact oneVersionWhitelist,
      Artifact genClass,
      @Nullable Artifact resourceJarBuilder,
      @Nullable Artifact timezoneData,
      FilesToRunProvider ijar,
      ImmutableListMultimap<String, String> compatibleJavacOptions,
      ImmutableList<String> javacOptions,
      ImmutableList<String> jvmOptions,
      boolean javacSupportsWorkers,
      ImmutableList<JavaPackageConfigurationProvider> packageConfiguration,
      JavaSemantics javaSemantics) {
    super(ImmutableMap.of(), Location.BUILTIN);

    this.label = label;
    this.bootclasspath = bootclasspath;
    this.extclasspath = extclasspath;
    this.javac = javac;
    this.tools = tools;
    this.javaBuilder = javaBuilder;
    this.headerCompiler = headerCompiler;
    this.headerCompilerDirect = headerCompilerDirect;
    this.forciblyDisableHeaderCompilation = forciblyDisableHeaderCompilation;
    this.singleJar = singleJar;
    this.oneVersion = oneVersion;
    this.oneVersionWhitelist = oneVersionWhitelist;
    this.genClass = genClass;
    this.resourceJarBuilder = resourceJarBuilder;
    this.timezoneData = timezoneData;
    this.ijar = ijar;
    this.compatibleJavacOptions = compatibleJavacOptions;
    this.javacOptions = javacOptions;
    this.jvmOptions = jvmOptions;
    this.javacSupportsWorkers = javacSupportsWorkers;
    this.packageConfiguration = packageConfiguration;
    this.javaSemantics = javaSemantics;
  }

  /** Returns the label for this {@code java_toolchain}. */
  public Label getToolchainLabel() {
    return label;
  }

  /** @return the target Java bootclasspath */
  public NestedSet<Artifact> getBootclasspath() {
    return bootclasspath;
  }

  /** @return the target Java extclasspath */
  public NestedSet<Artifact> getExtclasspath() {
    return extclasspath;
  }

  /** Returns the {@link Artifact} of the javac jar */
  public Artifact getJavac() {
    return javac;
  }

  /** Returns the {@link Artifact}s of compilation tools. */
  public NestedSet<Artifact> getTools() {
    return tools;
  }

  /** Returns the {@link FilesToRunProvider} of JavaBuilder */
  public FilesToRunProvider getJavaBuilder() {
    return javaBuilder;
  }

  /** @return the {@link FilesToRunProvider} of the Header Compiler deploy jar */
  @Nullable
  public FilesToRunProvider getHeaderCompiler() {
    return headerCompiler;
  }

  /**
   * Returns the {@link FilesToRunProvider} of the Header Compiler deploy jar for direct-classpath,
   * non-annotation processing actions.
   */
  @Nullable
  public FilesToRunProvider getHeaderCompilerDirect() {
    return headerCompilerDirect;
  }

  /**
   * Returns {@code true} if header compilation should be forcibly disabled, overriding
   * --java_header_compilation.
   */
  public boolean getForciblyDisableHeaderCompilation() {
    return forciblyDisableHeaderCompilation;
  }

  /** Returns the {@link Artifact} of the SingleJar deploy jar */
  public Artifact getSingleJar() {
    return singleJar;
  }

  /**
   * Return the {@link Artifact} of the binary that enforces one-version compliance of java
   * binaries.
   */
  @Nullable
  public Artifact getOneVersionBinary() {
    return oneVersion;
  }

  /** Return the {@link Artifact} of the whitelist used by the one-version compliance checker. */
  @Nullable
  public Artifact getOneVersionWhitelist() {
    return oneVersionWhitelist;
  }

  /** Returns the {@link Artifact} of the GenClass deploy jar */
  public Artifact getGenClass() {
    return genClass;
  }

  @Nullable
  public Artifact getResourceJarBuilder() {
    return resourceJarBuilder;
  }

  /**
   * Returns the {@link Artifact} of the latest timezone data resource jar that can be loaded by
   * Java 8 binaries.
   */
  @Nullable
  public Artifact getTimezoneData() {
    return timezoneData;
  }

  /** Returns the ijar executable */
  public FilesToRunProvider getIjar() {
    return ijar;
  }

  ImmutableListMultimap<String, String> getCompatibleJavacOptions() {
    return compatibleJavacOptions;
  }

  /** @return the map of target environment-specific javacopts. */
  public ImmutableList<String> getCompatibleJavacOptions(String key) {
    return getCompatibleJavacOptions().get(key);
  }

  /** @return the list of default options for the java compiler */
  public ImmutableList<String> getJavacOptions() {
    return javacOptions;
  }

  /**
   * @return the list of default options for the JVM running the java compiler and associated tools.
   */
  public ImmutableList<String> getJvmOptions() {
    return jvmOptions;
  }

  /** @return whether JavaBuilders supports running as a persistent worker or not */
  public boolean getJavacSupportsWorkers() {
    return javacSupportsWorkers;
  }

  /** Returns the global {@code java_plugin_configuration} data. */
  public ImmutableList<JavaPackageConfigurationProvider> packageConfiguration() {
    return packageConfiguration;
  }

  public JavaSemantics getJavaSemantics() {
    return javaSemantics;
  }

  /** Returns the input Java language level */
  // TODO(cushon): remove this API; it bakes a deprecated detail of the javac API into Bazel
  @Override
  public String getSourceVersion() {
    Iterator<String> it = getJavacOptions().iterator();
    while (it.hasNext()) {
      if (it.next().equals("-source") && it.hasNext()) {
        return it.next();
      }
    }
    return JAVA_SPECIFICATION_VERSION.value();
  }

  /** Returns the target Java language level */
  // TODO(cushon): remove this API; it bakes a deprecated detail of the javac API into Bazel
  @Override
  public String getTargetVersion() {
    Iterator<String> it = getJavacOptions().iterator();
    while (it.hasNext()) {
      if (it.next().equals("-target") && it.hasNext()) {
        return it.next();
      }
    }
    return JAVA_SPECIFICATION_VERSION.value();
  }

  @Override
  public FileApi getJavacJar() {
    return getJavac();
  }

  @Override
  public SkylarkNestedSet getSkylarkBootclasspath() {
    return SkylarkNestedSet.of(Artifact.class, getBootclasspath());
  }

  @Override
  public SkylarkList<String> getSkylarkJvmOptions() {
    return SkylarkList.createImmutable(getJvmOptions());
  }

  @Override
  public SkylarkNestedSet getSkylarkTools() {
    return SkylarkNestedSet.of(Artifact.class, getTools());
  }
}
