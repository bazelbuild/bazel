// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarProvider.Aar;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaNeverlinkInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * An implementation for the "android_library" rule.
 */
public abstract class AndroidLibrary implements RuleConfiguredTargetFactory {

  protected abstract JavaSemantics createJavaSemantics();
  protected abstract AndroidSemantics createAndroidSemantics();

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) {
    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    if (!AndroidSdkProvider.verifyPresence(ruleContext)) {
      return null;
    }
    List<? extends TransitiveInfoCollection> deps =
        ruleContext.getPrerequisites("deps", Mode.TARGET);
    checkResourceInlining(ruleContext);
    checkIdlRootImport(ruleContext);
    NestedSet<AndroidResourcesProvider.ResourceContainer> transitiveResources =
        collectTransitiveResources(ruleContext);
    NestedSetBuilder<Aar> transitiveAars = collectTransitiveAars(ruleContext);
    NestedSet<LinkerInput> transitiveNativeLibraries =
        AndroidCommon.collectTransitiveNativeLibraries(deps);
    NestedSet<Artifact> transitiveProguardConfigs =
        collectTransitiveProguardConfigs(ruleContext);
    AndroidIdlProvider transitiveIdlImportData = collectTransitiveIdlImports(ruleContext);
    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())) {
      try {
        if (!LocalResourceContainer.validateRuleContext(ruleContext)) {
          throw new RuleConfigurationException();
        }
        JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
        AndroidCommon androidCommon = new AndroidCommon(ruleContext, javaCommon);

        ApplicationManifest applicationManifest = androidSemantics.getManifestForRule(ruleContext);
        ResourceApk resourceApk = applicationManifest.packWithDataAndResources(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
            ruleContext, transitiveResources,
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_SYMBOLS_TXT),
            ImmutableList.<String>of(), /* configurationFilters */
            ImmutableList.<String>of(), /* uncompressedExtensions */
            ImmutableList.<String>of(), /* densities */
            null /* applicationId */,
            null /* versionCode */,
            null /* versionName */,
            false,
            null /* proguardCfgOut */);

        JavaTargetAttributes javaTargetAttributes = androidCommon.init(
            javaSemantics,
            androidSemantics,
            resourceApk,
            transitiveIdlImportData,
            false /* addCoverageSupport */,
            true /* collectJavaCompilationArgs */,
            AndroidRuleClasses.ANDROID_LIBRARY_GEN_JAR); 
        if (javaTargetAttributes == null) {
          return null;
        }

        Artifact classesJar = mergeJarsFromSrcs(ruleContext,
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR));

        Artifact aarOut = ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.ANDROID_LIBRARY_AAR);

        new AarGeneratorBuilder(ruleContext)
                .withPrimary(resourceApk.getPrimaryResource())
                .withManifest(resourceApk.getPrimaryResource().getManifest())
                .withRtxt(resourceApk.getPrimaryResource().getRTxt())
                .withClasses(classesJar)
                .strictResourceMerging()
                .setAAROut(aarOut)
                .build(ruleContext);

        Aar aar = new Aar(aarOut, applicationManifest.getManifest());

        RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
        androidCommon.addTransitiveInfoProviders(builder);
        androidSemantics.addTransitiveInfoProviders(
            builder, ruleContext, javaCommon, androidCommon,
            null, resourceApk, null, ImmutableList.<Artifact>of());

        return builder
            .add(AndroidNativeLibraryProvider.class,
                new AndroidNativeLibraryProvider(transitiveNativeLibraries))
            .add(JavaSourceJarsProvider.class, new JavaSourceJarsProvider(
                androidCommon.getTransitiveSourceJars(),
                androidCommon.getTopLevelSourceJars()))
            .add(JavaNeverlinkInfoProvider.class,
                new JavaNeverlinkInfoProvider(androidCommon.isNeverLink()))
            .add(AndroidCcLinkParamsProvider.class,
                new AndroidCcLinkParamsProvider(androidCommon.getCcLinkParamsStore()))
            .add(ProguardSpecProvider.class, new ProguardSpecProvider(transitiveProguardConfigs))
            .add(AndroidLibraryAarProvider.class, new AndroidLibraryAarProvider(aar,
                transitiveAars.add(aar).build()))
            .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, transitiveProguardConfigs)
            .build();
      } catch (RuleConfigurationException e) {
        // RuleConfigurations exceptions will only be thrown after the RuleContext is updated.
        // So, exit.
        return null;
      }
    } else {
      JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
      AndroidCommon androidCommon = new AndroidCommon(ruleContext, javaCommon);
      ResourceApk resourceApk = ResourceApk.fromTransitiveResources(transitiveResources);

      JavaTargetAttributes javaTargetAttributes = androidCommon.init(
          javaSemantics,
          androidSemantics,
          resourceApk,
          transitiveIdlImportData,
          false /* addCoverageSupport */,
          true /* collectJavaCompilationArgs */,
          AndroidRuleClasses.ANDROID_LIBRARY_GEN_JAR); 
      if (javaTargetAttributes == null) {
        return null;
      }

      RuleConfiguredTargetBuilder targetBuilder = androidCommon.addTransitiveInfoProviders(
          new RuleConfiguredTargetBuilder(ruleContext));

      androidSemantics.addTransitiveInfoProviders(
          targetBuilder, ruleContext, javaCommon, androidCommon,
          null, null, null, ImmutableList.<Artifact>of());
      targetBuilder
          .add(AndroidNativeLibraryProvider.class,
              new AndroidNativeLibraryProvider(transitiveNativeLibraries))
          .add(JavaSourceJarsProvider.class, androidCommon.getJavaSourceJarsProvider())
          .add(AndroidCcLinkParamsProvider.class,
              new AndroidCcLinkParamsProvider(androidCommon.getCcLinkParamsStore()))
          .add(JavaNeverlinkInfoProvider.class,
              new JavaNeverlinkInfoProvider(androidCommon.isNeverLink()))
          .add(ProguardSpecProvider.class, new ProguardSpecProvider(transitiveProguardConfigs))
          .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, transitiveProguardConfigs);

      Artifact aarOut = ruleContext.getImplicitOutputArtifact(
          AndroidRuleClasses.ANDROID_LIBRARY_AAR);

      Artifact classesJar = mergeJarsFromSrcs(ruleContext,
          ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LIBRARY_CLASS_JAR));

      ResourceContainer primaryResources;

      if (AndroidCommon.getAndroidResources(ruleContext) != null) {
        primaryResources = Iterables.getOnlyElement(
            AndroidCommon.getAndroidResources(ruleContext).getTransitiveAndroidResources());

        Aar aar = new Aar(aarOut, primaryResources.getManifest());
        targetBuilder.add(AndroidLibraryAarProvider.class, new AndroidLibraryAarProvider(
            aar, transitiveAars.add(aar).build()));
      } else {
        // there are no local resources and resources attribute was not specified either
        ApplicationManifest applicationManifest =
            ApplicationManifest.generatedManifest(ruleContext);

        Artifact apk = ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.ANDROID_RESOURCES_APK);

        String javaPackage;
        if (apk.getExecPath().getFirstSegment(ImmutableSet.of("java", "javatests"))
            != PathFragment.INVALID_SEGMENT) {
          javaPackage = JavaUtil.getJavaPackageName(apk.getExecPath());
        } else {
          // This is a workaround for libraries that don't follow the standard Bazel package format
          javaPackage = apk.getRootRelativePath().getPathString().replace('/', '.');
        }
        if (ruleContext.attributes().isAttributeValueExplicitlySpecified("custom_package")) {
          javaPackage = ruleContext.attributes().get("custom_package", Type.STRING);
        }

        primaryResources = new ResourceContainer(ruleContext.getLabel(),
            javaPackage, null /* renameManifestPackage */, false /* inlinedConstants */,
            apk, applicationManifest.getManifest(),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_JAVA_SOURCE_JAR),
            ImmutableList.<Artifact>of(), ImmutableList.<Artifact>of(),
            ImmutableList.<PathFragment>of(), ImmutableList.<PathFragment>of(),
            ruleContext.attributes().get("exports_manifest", Type.BOOLEAN),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT), null);

        primaryResources = new AndroidResourcesProcessorBuilder(ruleContext)
                .setApkOut(apk)
                .setRTxtOut(primaryResources.getRTxt())
                .setSourceJarOut(primaryResources.getJavaSourceJar())
                .setJavaPackage(primaryResources.getJavaPackage())
                .withPrimary(primaryResources)
                .withDependencies(transitiveResources)
                .setDebug(
                    ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT)
                .setWorkingDirectory(ruleContext.getUniqueDirectory("_resources"))
                .build(ruleContext);

        targetBuilder.add(AndroidLibraryAarProvider.class, new AndroidLibraryAarProvider(
            null, transitiveAars.build()));
      }

      new AarGeneratorBuilder(ruleContext)
          .withPrimary(primaryResources)
          .withManifest(primaryResources.getManifest())
          .withRtxt(primaryResources.getRTxt())
          .withClasses(classesJar)
          .setAAROut(aarOut)
          .build(ruleContext);

      return targetBuilder.build();
    }
  }

  private static Artifact mergeJarsFromSrcs(RuleContext ruleContext, Artifact inputJar) {
    ImmutableList<Artifact> jarSources = ruleContext
        .getPrerequisiteArtifacts("srcs", Mode.TARGET).filter(JavaSemantics.JAR).list();
    if (jarSources.isEmpty()) {
      return inputJar;
    }
    Artifact mergedJar = ruleContext.getImplicitOutputArtifact(
        AndroidRuleClasses.ANDROID_LIBRARY_AAR_CLASSES_JAR);
    new SingleJarBuilder(ruleContext)
        .setOutputJar(mergedJar)
        .addInputJar(inputJar)
        .addInputJars(jarSources)
        .build();
    return mergedJar;
  }

  private AndroidIdlProvider collectTransitiveIdlImports(RuleContext ruleContext) {
    NestedSetBuilder<String> rootsBuilder = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Artifact> importsBuilder = NestedSetBuilder.naiveLinkOrder();

    for (AndroidIdlProvider dep : ruleContext.getPrerequisites(
        "deps", Mode.TARGET, AndroidIdlProvider.class)) {
      rootsBuilder.addTransitive(dep.getTransitiveIdlImportRoots());
      importsBuilder.addTransitive(dep.getTransitiveIdlImports());
    }

    Collection<Artifact> idlImports = getIdlImports(ruleContext);
    if (!hasExplicitlySpecifiedIdlImportRoot(ruleContext)) {
      for (Artifact idlImport : idlImports) {
        PathFragment javaRoot = JavaUtil.getJavaRoot(idlImport.getExecPath());
        if (javaRoot == null) {
          ruleContext.ruleError("Cannot determine java/javatests root for import "
              + idlImport.getExecPathString());
        } else {
          rootsBuilder.add(javaRoot.toString());
        }
      }
    } else {
      PathFragment pkgFragment = ruleContext.getLabel().getPackageFragment();
      Set<PathFragment> idlImportRoots = new HashSet<>();
      for (Artifact idlImport : idlImports) {
        idlImportRoots.add(idlImport.getRoot().getExecPath()
            .getRelative(pkgFragment)
            .getRelative(getIdlImportRoot(ruleContext)));
      }
      for (PathFragment idlImportRoot : idlImportRoots) {
        rootsBuilder.add(idlImportRoot.toString());
      }
    }
    importsBuilder.addAll(idlImports);

    return new AndroidIdlProvider(rootsBuilder.build(), importsBuilder.build());
  }

  private void checkIdlRootImport(RuleContext ruleContext) {
    if (hasExplicitlySpecifiedIdlImportRoot(ruleContext)
        && !hasExplicitlySpecifiedIdlSrcsOrParcelables(ruleContext)) {
      ruleContext.attributeError("idl_import_root",
          "Neither idl_srcs nor idl_parcelables were specified, "
          + "but 'idl_import_root' attribute was set");
    }
  }

  private void checkResourceInlining(RuleContext ruleContext) {
    AndroidResourcesProvider resources = AndroidCommon.getAndroidResources(ruleContext);
    if (resources == null) {
      return;
    }

    ResourceContainer container = Iterables.getOnlyElement(
        resources.getTransitiveAndroidResources());

    if (container.getConstantsInlined()
        && !container.getArtifacts(ResourceType.RESOURCES).isEmpty()) {
      ruleContext.ruleError("This android library has some resources assigned, so the target '"
          + resources.getLabel() + "' should have the attribute inline_constants set to 0");
    }
  }

  private NestedSet<ResourceContainer> collectTransitiveResources(RuleContext ruleContext) {
    NestedSetBuilder<ResourceContainer> builder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidResourcesProvider resource : Iterables.concat(
        ruleContext.getPrerequisites("resources", Mode.TARGET, AndroidResourcesProvider.class),
        ruleContext.getPrerequisites("deps", Mode.TARGET, AndroidResourcesProvider.class))) {
      builder.addTransitive(resource.getTransitiveAndroidResources());
    }

    return builder.build();
  }

  private NestedSetBuilder<Aar> collectTransitiveAars(RuleContext ruleContext) {
    NestedSetBuilder<Aar> builder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidLibraryAarProvider library :
        ruleContext.getPrerequisites("deps", Mode.TARGET, AndroidLibraryAarProvider.class)) {
      builder.addTransitive(library.getTransitiveAars());
    }
    return builder;
  }

  private boolean hasExplicitlySpecifiedIdlImportRoot(RuleContext ruleContext) {
    return ruleContext.getRule().isAttributeValueExplicitlySpecified("idl_import_root");
  }

  private boolean hasExplicitlySpecifiedIdlSrcsOrParcelables(RuleContext ruleContext) {
    return ruleContext.getRule().isAttributeValueExplicitlySpecified("idl_srcs")
        || ruleContext.getRule().isAttributeValueExplicitlySpecified("idl_parcelables");
  }

  private String getIdlImportRoot(RuleContext ruleContext) {
    return ruleContext.attributes().get("idl_import_root", Type.STRING);
  }

  /**
   * Returns the union of "idl_srcs" and "idl_parcelables", i.e. all .aidl files
   * provided by this library that contribute to .aidl --> .java compilation.
   */
  private static Collection<Artifact> getIdlImports(RuleContext ruleContext) {
    return ImmutableList.<Artifact>builder()
        .addAll(AndroidCommon.getIdlParcelables(ruleContext))
        .addAll(AndroidCommon.getIdlSrcs(ruleContext))
        .build();
  }

  private NestedSet<Artifact> collectTransitiveProguardConfigs(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> specsBuilder = NestedSetBuilder.naiveLinkOrder();

    for (ProguardSpecProvider dep : ruleContext.getPrerequisites(
        "deps", Mode.TARGET, ProguardSpecProvider.class)) {
      specsBuilder.addTransitive(dep.getTransitiveProguardSpecs());
    }

    // Pass our local proguard configs through the validator, which checks a whitelist.
    if (!getProguardConfigs(ruleContext).isEmpty()) {
      FilesToRunProvider proguardWhitelister = ruleContext
        .getExecutablePrerequisite("$proguard_whitelister", Mode.HOST);
      for (Artifact specToValidate : getProguardConfigs(ruleContext)) {
        //If we're validating j/a/b/testapp/proguard.cfg, the output will be:
        //j/a/b/testapp/proguard.cfg_valid
        Artifact output = ruleContext.getUniqueDirectoryArtifact(
            "validated_proguard",
            specToValidate.getRootRelativePath().replaceName(
                specToValidate.getFilename() + "_valid"),
            ruleContext.getBinOrGenfilesDirectory());
        ruleContext.registerAction(new SpawnAction.Builder()
            .addInput(specToValidate)
            .setExecutable(proguardWhitelister)
            .setProgressMessage("Validating proguard configuration")
            .setMnemonic("ValidateProguard")
            .addArgument("--path")
            .addArgument(specToValidate.getExecPathString())
            .addArgument("--output")
            .addArgument(output.getExecPathString())
            .addOutput(output)
            .build(ruleContext));
        specsBuilder.add(output);
      }
    }
    return specsBuilder.build();
  }

  private Collection<Artifact> getProguardConfigs(RuleContext ruleContext) {
    return ruleContext.getPrerequisiteArtifacts("proguard_specs", Mode.TARGET).list();
  }
}
