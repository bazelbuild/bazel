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

package com.google.devtools.build.lib.ideinfo;

import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.ArtifactLocation;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.JavaRuleIdeInfo;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.LibraryArtifact;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;

/**
 * Generates ide-build information for Android Studio.
 */
public class AndroidStudioInfoAspect implements ConfiguredAspectFactory {
  public static final String NAME = "AndroidStudioInfoAspect";
  public static final String IDE_RESOLVE = "ide-resolve";
  public static final String IDE_BUILD = "ide-build";
  public static final String IDE_BUILD_SUFFIX = ".aswb-build";

  @Override
  public AspectDefinition getDefinition() {
    return new AspectDefinition.Builder(NAME)
        .requireProvider(JavaSourceInfoProvider.class)
        .attributeAspect("deps", AndroidStudioInfoAspect.class)
        .build();
  }

  @Override
  public Aspect create(ConfiguredTarget base, RuleContext ruleContext,
      AspectParameters parameters) {
    NestedSetBuilder<Artifact> ideBuildFilesBuilder = NestedSetBuilder.stableOrder();
    Iterable<AndroidStudioInfoFilesProvider> deps =
        ruleContext.getPrerequisites("deps", Mode.TARGET, AndroidStudioInfoFilesProvider.class);
    for (AndroidStudioInfoFilesProvider dep : deps) {
      ideBuildFilesBuilder.addTransitive(dep.getIdeBuildFiles());
    }

    RuleIdeInfo.Kind ruleKind = getRuleKind(ruleContext.getRule());
    if (ruleKind != RuleIdeInfo.Kind.UNRECOGNIZED) {
      Artifact ideBuildFile = createIdeBuildArtifact(base, ruleContext, ruleKind);
      ideBuildFilesBuilder.add(ideBuildFile);
    }

    NestedSet<Artifact> ideBuildFiles = ideBuildFilesBuilder.build();
    return new Aspect.Builder(NAME)
        .addOutputGroup(IDE_BUILD, ideBuildFiles)
        .addProvider(
            AndroidStudioInfoFilesProvider.class, new AndroidStudioInfoFilesProvider(ideBuildFiles))
        .build();
  }

  private Artifact createIdeBuildArtifact(
      ConfiguredTarget base, RuleContext ruleContext, RuleIdeInfo.Kind ruleKind) {
    PathFragment ideBuildFilePath = getOutputFilePath(base, ruleContext);
    Root genfilesDirectory = ruleContext.getConfiguration().getGenfilesDirectory();
    Artifact ideBuildFile =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(ideBuildFilePath, genfilesDirectory);

    RuleIdeInfo.Builder outputBuilder = RuleIdeInfo.newBuilder();

    outputBuilder.setLabel(base.getLabel().toString());

    outputBuilder.setBuildFile(
        ruleContext
            .getRule()
            .getPackage()
            .getBuildFile()
            .getPath()
            .toString());

    outputBuilder.setKind(ruleKind);

    // Calculate source files.
    JavaSourceInfoProvider sourceInfoProvider = base.getProvider(JavaSourceInfoProvider.class);
    Collection<Artifact> sourceFiles =
        sourceInfoProvider != null
            ? sourceInfoProvider.getSourceFiles()
            : ImmutableList.<Artifact>of();
    for (Artifact sourceFile : sourceFiles) {
      outputBuilder.addSources(makeArtifactLocation(sourceFile));
    }

    outputBuilder.setJavaRuleIdeInfo(makeJavaRuleIdeInfo(base));

    final RuleIdeInfo ruleIdeInfo = outputBuilder.build();
    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(),
            ideBuildFile,
            new ByteSource() {
              @Override
              public InputStream openStream() throws IOException {
                return ruleIdeInfo.toByteString().newInput();
              }
            },
            /*makeExecutable =*/ false));
    return ideBuildFile;
  }

  private static ArtifactLocation makeArtifactLocation(Artifact artifact) {
    return ArtifactLocation.newBuilder()
        .setRootPath(artifact.getRoot().getPath().toString())
        .setRelativePath(artifact.getRootRelativePathString())
        .build();
  }

  private static JavaRuleIdeInfo makeJavaRuleIdeInfo(ConfiguredTarget base) {
    JavaRuleIdeInfo.Builder builder = JavaRuleIdeInfo.newBuilder();
    JavaRuleOutputJarsProvider outputJarsProvider =
        base.getProvider(JavaRuleOutputJarsProvider.class);
    if (outputJarsProvider != null) {
      {
        LibraryArtifact.Builder jarsBuilder = LibraryArtifact.newBuilder();
        Artifact classJar = outputJarsProvider.getClassJar();
        if (classJar != null) {
          jarsBuilder.setJar(makeArtifactLocation(classJar));
        }
        Artifact srcJar = outputJarsProvider.getSrcJar();
        if (srcJar != null) {
          jarsBuilder.setSourceJar(makeArtifactLocation(srcJar));
        }
        if (jarsBuilder.hasJar() || jarsBuilder.hasSourceJar()) {
          builder.addJars(jarsBuilder.build());
        }
      }

      {
        LibraryArtifact.Builder genjarsBuilder = LibraryArtifact.newBuilder();

        Artifact genClassJar = outputJarsProvider.getGenClassJar();
        if (genClassJar != null) {
          genjarsBuilder.setJar(makeArtifactLocation(genClassJar));
        }
        Artifact gensrcJar = outputJarsProvider.getGensrcJar();
        if (gensrcJar != null) {
          genjarsBuilder.setSourceJar(makeArtifactLocation(gensrcJar));
        }
        if (genjarsBuilder.hasJar() || genjarsBuilder.hasSourceJar()) {
          builder.addJars(genjarsBuilder.build());
        }
      }
    }

    return builder.build();
  }

  private PathFragment getOutputFilePath(ConfiguredTarget base, RuleContext ruleContext) {
    PathFragment packagePathFragment =
        ruleContext.getLabel().getPackageIdentifier().getPathFragment();
    String name = base.getLabel().getName();
    return new PathFragment(packagePathFragment, new PathFragment(name + IDE_BUILD_SUFFIX));
  }

  private RuleIdeInfo.Kind getRuleKind(Rule rule) {
    RuleIdeInfo.Kind kind;
    if ("java_library".equals(rule.getRuleClassObject().getName())) {
      kind = RuleIdeInfo.Kind.JAVA_LIBRARY;
    } else {
      kind = RuleIdeInfo.Kind.UNRECOGNIZED;
    }
    return kind;
  }
}
