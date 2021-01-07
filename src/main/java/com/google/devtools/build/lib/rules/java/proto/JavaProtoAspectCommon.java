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
package com.google.devtools.build.lib.rules.java.proto;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaLibraryHelper;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainProvider;

/** Common logic used by java*_proto aspects (e.g. {@link JavaLiteProtoAspect}). */
public class JavaProtoAspectCommon {

  // The name of an attribute of {@link JavaProtoAspect} used for storing the {@link Label} of
  // the proto toolchain for Java.
  public static final String SPEED_PROTO_TOOLCHAIN_ATTR = ":aspect_java_proto_toolchain";
  // The name of an attribute of {@link JavaLiteProtoLibrary} and {@link JavaLiteProtoAspect} used
  // for storing the {@link Label} of the proto toolchain for Java-lite.
  public static final String LITE_PROTO_TOOLCHAIN_ATTR = ":aspect_proto_toolchain_for_javalite";

  private static final String SPEED_JAR_SUFFIX = "-speed";
  private static final String LITE_JAR_SUFFIX = "-lite";
  private static final String MUTABLE_JAR_SUFFIX = "-new-mutable";

  private final RuleContext ruleContext;
  private final JavaSemantics javaSemantics;
  private final String protoToolchainAttr;
  private final String jarSuffix;
  private final RpcSupport rpcSupport;

  /**
   * Returns a {@link JavaProtoAspectCommon} instance that handles logic for {@code
   * java_proto_library}.
   */
  static JavaProtoAspectCommon getSpeedInstance(
      RuleContext ruleContext, JavaSemantics javaSemantics, RpcSupport rpcSupport) {
    return new JavaProtoAspectCommon(
        ruleContext, javaSemantics, rpcSupport, SPEED_PROTO_TOOLCHAIN_ATTR, SPEED_JAR_SUFFIX);
  }

  /**
   * Returns a {@link JavaProtoAspectCommon} instance that handles logic for {@code
   * java_lite_proto_library}.
   */
  static JavaProtoAspectCommon getLiteInstance(
      RuleContext ruleContext, JavaSemantics javaSemantics) {
    return new JavaProtoAspectCommon(
        ruleContext, javaSemantics, null, LITE_PROTO_TOOLCHAIN_ATTR, LITE_JAR_SUFFIX);
  }

  /**
   * Returns a {@link JavaProtoAspectCommon} instance that handles logic for {@code
   * java_mutable_proto_library}.
   */
  public static JavaProtoAspectCommon getMutableInstance(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      RpcSupport rpcSupport,
      String protoToolchainAttr) {
    return new JavaProtoAspectCommon(
        ruleContext, javaSemantics, rpcSupport, protoToolchainAttr, MUTABLE_JAR_SUFFIX);
  }

  private JavaProtoAspectCommon(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      RpcSupport rpcSupport,
      String protoToolchainAttr,
      String jarSuffix) {
    this.ruleContext = ruleContext;
    this.javaSemantics = javaSemantics;
    this.protoToolchainAttr = protoToolchainAttr;
    this.jarSuffix = jarSuffix;
    this.rpcSupport = rpcSupport;
  }

  /**
   * Registers an action that compiles the given {@code sourceJar} and archives the compiled classes
   * into {@code outputJar}, using {@code dep} as information about the dependencies compilation.
   *
   * @return a {@JavaCompilationArgsProvider} wrapping information about the compilation action that
   *     was registered.
   */
  public JavaCompilationArgsProvider createJavaCompileAction(
      String injectingRuleKind,
      Artifact sourceJar,
      Artifact outputJar,
      JavaCompilationArgsProvider dep)
      throws InterruptedException {
    JavaLibraryHelper helper =
        new JavaLibraryHelper(ruleContext)
            .setInjectingRuleKind(injectingRuleKind)
            .setOutput(outputJar)
            .addSourceJars(sourceJar)
            .setJavacOpts(ProtoJavacOpts.constructJavacOpts(ruleContext))
            .addDep(dep)
            .setCompilationStrictDepsMode(StrictDepsMode.ERROR);
    for (TransitiveInfoCollection t : getProtoRuntimeDeps()) {
      JavaCompilationArgsProvider provider =
          JavaInfo.getProvider(JavaCompilationArgsProvider.class, t);
      if (provider != null) {
        helper.addDep(provider);
      }
    }

    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            JavaToolchainProvider.from(ruleContext),
            JavaRuleOutputJarsProvider.builder(),
            /*createOutputSourceJar*/ false,
            /*outputSourceJar=*/ null);
    return helper.buildCompilationArgsProvider(
        artifacts, /*isReportedAsStrict=*/ true, /*isNeverlink=*/ false);
  }

  /**
   * Returns a list of all the target needed by proto libraries (e.g. {@code java_proto_library}) at
   * runtime.
   */
  public ImmutableList<TransitiveInfoCollection> getProtoRuntimeDeps() {
    ImmutableList.Builder<TransitiveInfoCollection> result = ImmutableList.builder();
    TransitiveInfoCollection runtime = getProtoToolchainProvider().runtime();
    if (runtime != null) {
      result.add(runtime);
    }
    if (rpcSupport != null) {
      result.addAll(rpcSupport.getRuntimes(ruleContext));
    }
    return result.build();
  }

  /** Returns the toolchain that specifies how to generate code from {@code .proto} files. */
  public ProtoLangToolchainProvider getProtoToolchainProvider() {
    return checkNotNull(
        ruleContext.getPrerequisite(protoToolchainAttr, ProtoLangToolchainProvider.class));
  }

  /**
   * Returns the toolchain that specifies how to generate Java-lite code from {@code .proto} files.
   */
  static ProtoLangToolchainProvider getLiteProtoToolchainProvider(RuleContext ruleContext) {
    return ruleContext.getPrerequisite(LITE_PROTO_TOOLCHAIN_ATTR, ProtoLangToolchainProvider.class);
  }

  /**
   * Returns an {@link Artifact} corresponding to a source jar. Its name is computed from the label
   * name and the library type of the current instance. For example, if the instance is created with
   * {@link getLiteInstance} the name of the jar will be "<label>-lite-src.jar".
   *
   * <p>The {@link Artifact} will be created in the bazel-genfiles directory.
   */
  public Artifact getSourceJarArtifact() {
    return ruleContext.getGenfilesArtifact(
        ruleContext.getLabel().getName() + jarSuffix + "-src.jar");
  }

  /**
   * Returns an {@link Artifact} corresponding to an output jar. Its name is computed from the label
   * name and the library type of the current instance. For example, if the instance is created with
   * {@link getLiteInstance} the name of the jar will be "lib<label>-lite.jar".
   *
   * <p>The {@link Artifact} will be created in the bazel-bin directory.
   */
  public Artifact getOutputJarArtifact() {
    return ruleContext.getBinArtifact(
        "lib" + ruleContext.getLabel().getName() + jarSuffix + ".jar");
  }
}
