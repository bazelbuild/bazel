// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;

/**
 * Aspect to {@link DexArchiveProvider build .dex Archives} from Jars.
 */
public final class DexArchiveAspect extends NativeAspectClass implements ConfiguredAspectFactory {
  public static final String NAME = "DexArchiveAspect";
  private static final String ASPECT_DEXBUILDER_PREREQ = "$dex_archive_dexbuilder";
  private static final ImmutableList<String> TRANSITIVE_ATTRIBUTES =
      ImmutableList.of("deps", "exports", "runtime_deps");
  private final String toolsRepository;

  public DexArchiveAspect(String toolsRepository) {
    this.toolsRepository = toolsRepository;
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters params) {
    AspectDefinition.Builder result = new AspectDefinition.Builder(NAME)
        // Actually we care about JavaRuntimeJarProvider, but rules don't advertise that provider.
        .requireProvider(JavaCompilationArgsProvider.class)
        .add(attr(ASPECT_DEXBUILDER_PREREQ, LABEL).cfg(HOST).exec()
        // Parse label here since we don't have RuleDefinitionEnvironment.getLabel like in a rule
            .value(Label.parseAbsoluteUnchecked(toolsRepository + "//tools/android:dexbuilder")))
        .requiresConfigurationFragments(AndroidConfiguration.class);
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      result.attributeAspect(attr, this);
    }
    return result.build();
  }

  @Override
  public ConfiguredAspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters params) throws InterruptedException {
    checkState(base.getProvider(DexArchiveProvider.class) == null,
        "dex archive natively generated: %s", ruleContext.getLabel());

    if (JavaCommon.isNeverLink(ruleContext)) {
      return new ConfiguredAspect.Builder(NAME, ruleContext)
          .addProvider(DexArchiveProvider.class, DexArchiveProvider.NEVERLINK)
          .build();
    }

    DexArchiveProvider.Builder result = createArchiveProviderBuilderFromDeps(ruleContext);
    JavaRuntimeJarProvider jarProvider = base.getProvider(JavaRuntimeJarProvider.class);
    if (jarProvider != null) {
      for (Artifact jar : jarProvider.getRuntimeJars()) {
        Artifact dexArchive = createDexArchiveAction(ruleContext, jar);
        result.addDexArchive(dexArchive, jar);
      }
    }
    return new ConfiguredAspect.Builder(NAME, ruleContext)
        .addProvider(DexArchiveProvider.class, result.build())
        .build();
  }

  private static DexArchiveProvider.Builder createArchiveProviderBuilderFromDeps(
      RuleContext ruleContext) {
    DexArchiveProvider.Builder result = new DexArchiveProvider.Builder();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (ruleContext.getRule().getRuleClassObject().hasAttr(attr, LABEL_LIST)) {
        result.addTransitiveProviders(
            ruleContext.getPrerequisites(attr, Mode.TARGET, DexArchiveProvider.class));
      }
    }
    return result;
  }

  private static Artifact createDexArchiveAction(RuleContext ruleContext, Artifact jar) {
    Artifact result = AndroidBinary.getDxArtifact(ruleContext, jar.getFilename() + ".dex.zip");
    // Aspect must use attribute name for dexbuilder prereq that's different from the prerequisite
    // declared on AndroidBinaryBaseRule because the two prereq's can otherwise name-clash when
    // android_binary targets are built as part of an android_test: building android_test causes
    // the aspect to apply to the android_binary target, but android_binary itself also declares
    // a $dexbuilder prerequisite, so if the aspect also used $dexbuilder then
    // RuleContext.getExecutablePrerequisite would fail with "$dexbuilder produces multiple prereqs"
    // (note they both resolve to the same artifact but that doesn't seem to prevent the exception
    // from being thrown).
    createDexArchiveAction(ruleContext, ASPECT_DEXBUILDER_PREREQ, jar, result);
    return result;
  }

  /**
   * Creates a dex archive using an executable prerequisite called {@code "$dexbuilder"}.  Rules
   * calling this method must declare the appropriate prerequisite, similar to how
   * {@link #getDefinition} does it for {@link DexArchiveAspect} under a different name.
   */
  // Package-private method for use in AndroidBinary
  static void createDexArchiveAction(RuleContext ruleContext, Artifact jar, Artifact dexArchive) {
    createDexArchiveAction(ruleContext, "$dexbuilder", jar, dexArchive);
  }

  private static void createDexArchiveAction(RuleContext ruleContext, String dexbuilderPrereq,
      Artifact jar, Artifact dexArchive) {
    // Write command line arguments into a params file for compatibility with WorkerSpawnStrategy
    CustomCommandLine.Builder args = new CustomCommandLine.Builder()
        .addExecPath("--input_jar", jar)
        .addExecPath("--output_zip", dexArchive);
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      // Match what we do in AndroidCommon.createDexAction
      args.add("--nolocals"); // TODO(bazel-team): Still needed? See createDexAction
    }
    Artifact paramFile =
        ruleContext.getDerivedArtifact(
            ParameterFile.derivePath(dexArchive.getRootRelativePath()), dexArchive.getRoot());
    ruleContext.registerAction(
        new ParameterFileWriteAction(
            ruleContext.getActionOwner(),
            paramFile,
            args.build(),
            ParameterFile.ParameterFileType.UNQUOTED,
            ISO_8859_1));
    SpawnAction.Builder dexbuilder = new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite(dexbuilderPrereq, Mode.HOST))
        // WorkerSpawnStrategy expects the last argument to be @paramfile
        .addArgument("@" + paramFile.getExecPathString())
        .addInput(jar)
        .addInput(paramFile)
        .addOutput(dexArchive)
        .setMnemonic("DexBuilder")
        .setProgressMessage("Dexing " + jar.prettyPrint());
    ruleContext.registerAction(dexbuilder.build(ruleContext));
  }
}
