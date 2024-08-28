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
// Copyright 2017 The Bazel Authors. All rights reserved.
package com.google.devtools.build.android;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.XmlNode;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.ProtoXmlUtils;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Performs resource validation and static linking for compiled android resources. */
@Parameters(separators = "= ")
public final class ValidateAndLinkResourcesAction {

  /** Action configuration options. */
  public static class Options {
    /**
     * TODO(b/64570523): Still used by blaze. Will be removed as part of the command line cleanup.
     *
     * @deprecated Use --resources.
     */
    @Parameter(
        names = "--compiled",
        converter = Converters.CompatExistingPathConverter.class,
        description = "Compiled resources to link.")
    @Deprecated
    public Path compiled;

    @Parameter(
        names = "--compiledDep",
        listConverter = Converters.CompatPathListConverter.class,
        description = "Compiled resource dependencies to link.")
    public List<Path> compiledDeps = ImmutableList.of();

    /**
     * TODO(b/64570523): Still used by blaze. Will be removed as part of the command line cleanup.
     *
     * @deprecated Use --resources.
     */
    @Parameter(
        names = "--manifest",
        converter = Converters.CompatExistingPathConverter.class,
        description = "Manifest for the library.")
    @Deprecated
    public Path manifest;

    @Parameter(
        names = "--resources",
        converter = Converters.CompatCompiledResourcesConverter.class,
        description = "Compiled resources to link.")
    public CompiledResources resources;

    // TODO(b/64570523): remove this flag when it is no longer used.
    @Parameter(
        names = "--libraries",
        listConverter = Converters.CompatStaticLibraryListConverter.class,
        description = "Static libraries to link against. Deprecated, use --library")
    public List<StaticLibrary> deprecatedLibraries;

    @Parameter(
        names = "--library",
        converter = Converters.CompatStaticLibraryConverter.class,
        description = "Static libraries to link against.")
    public List<StaticLibrary> libraries = ImmutableList.of();

    @Parameter(names = "--packageForR", description = "Package for the resources.")
    public String packageForR;

    @Parameter(
        names = "--staticLibraryOut",
        converter = Converters.CompatPathConverter.class,
        description = "Static library produced.")
    public Path staticLibraryOut;

    @Parameter(
        names = "--rTxtOut",
        converter = Converters.CompatPathConverter.class,
        description = "R.txt out.")
    public Path rTxtOut;

    @Parameter(
        names = "--sourceJarOut",
        converter = Converters.CompatPathConverter.class,
        description = "Generated java classes from the resources.")
    public Path sourceJarOut;

    @Parameter(
        names = "--resourceApks",
        listConverter = Converters.CompatPathListConverter.class,
        description = "List of reource only APK files to link against.")
    public List<Path> resourceApks = ImmutableList.of();
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    Aapt2ConfigOptions aapt2Options = new Aapt2ConfigOptions();
    Object[] allOptions =
        new Object[] {options, aapt2Options, new ResourceProcessorCommonOptions()};
    JCommander jc = new JCommander(allOptions);
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(jc, args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(allOptions, preprocessedArgs);
    jc.parse(normalizedArgs);

    final Profiler profiler = LoggingProfiler.createAndStart("manifest");

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      CompiledResources resources =
          // TODO(b/64570523): Remove when the flags are standardized.
          Optional.ofNullable(options.resources)
              .orElseGet(
                  () ->
                      CompiledResources.from(
                          Preconditions.checkNotNull(options.compiled),
                          Preconditions.checkNotNull(options.manifest)))
              // We need to make the manifest aapt safe (w.r.t., placeholders). For now, just stub
              // it out.
              .processManifest(
                  manifest ->
                      AndroidManifest.parseFrom(manifest)
                          .writeDummyManifestForAapt(
                              scopedTmp.getPath().resolve("manifest-aapt-dummy"),
                              options.packageForR));
      ImmutableList<CompiledResources> includes =
          options.compiledDeps.stream().map(CompiledResources::from).collect(toImmutableList());
      profiler.recordEndOf("manifest").startTask("validate");

      // TODO(b/146663858): distinguish direct/transitive deps for "strict deps".
      // TODO(b/128711690): validate AndroidManifest.xml
      checkVisibilityOfResourceReferences(
          /* androidManifest= */ XmlNode.getDefaultInstance(), resources, includes);

      ImmutableList<StaticLibrary> resourceApks = ImmutableList.of();
      if (options.resourceApks != null) {
        resourceApks =
            options.resourceApks.stream().map(StaticLibrary::from).collect(toImmutableList());
      }

      profiler.recordEndOf("validate").startTask("link");
      ResourceLinker.create(aapt2Options.aapt2, executorService, scopedTmp.getPath())
          .profileUsing(profiler)
          // NB: these names are really confusing.
          //   .dependencies is meant for linking in android.jar
          //   .include is meant for regular dependencies
          //   .resourceApks is meant for linking runtime resource only apks
          .dependencies(Optional.ofNullable(options.deprecatedLibraries).orElse(options.libraries))
          .include(includes)
          .resourceApks(resourceApks)
          .buildVersion(aapt2Options.buildToolsVersion)
          .outputAsProto(aapt2Options.resourceTableAsProto)
          .linkStatically(resources)
          .copyLibraryTo(options.staticLibraryOut)
          .copySourceJarTo(options.sourceJarOut)
          .copyRTxtTo(options.rTxtOut);
      profiler.recordEndOf("link");
    }
  }

  /**
   * Validates that resources referenced from AndroidManifest.xml and res/ are visible.
   *
   * @param androidManifest AndroidManifest in protobuf format; while {@link CompiledResources} also
   *     contains a manifest, aapt2 requires it to be regular XML and only converts it to protobuf
   *     after "linking".
   * @param compiled resources of a compilation unit (i.e. an android_library rule)
   * @param deps resources from the transitive closure of the rule's "deps" attribute
   */
  static void checkVisibilityOfResourceReferences(
      XmlNode androidManifest, CompiledResources compiled, List<CompiledResources> deps) {
    ImmutableSet<String> privateResourceNames =
        deps.stream()
            .flatMap(
                cr ->
                    AndroidCompiledDataDeserializer.create(
                        /* includeFileContentsForValidation= */ false)
                        .read(DependencyInfo.UNKNOWN, cr.getZip())
                        .entrySet()
                        .stream())
            .filter(entry -> entry.getValue().getVisibility() == Visibility.PRIVATE)
            .map(entry -> ((FullyQualifiedName) entry.getKey()).asQualifiedReference())
            .collect(toImmutableSet());

    StringBuilder errorMessage = new StringBuilder();
    {
      ImmutableList<String> referencedPrivateResources =
          ProtoXmlUtils.getAllResourceReferences(androidManifest).stream()
              .map(Reference::getName)
              .filter(privateResourceNames::contains)
              .collect(toImmutableList());
      if (!referencedPrivateResources.isEmpty()) {
        errorMessage
            .append("AndroidManifest.xml references external private resources ")
            .append(referencedPrivateResources)
            .append('\n');
      }
    }

    for (Map.Entry<DataKey, DataResource> resource :
        AndroidCompiledDataDeserializer.create(/*includeFileContentsForValidation=*/ true)
            .read(DependencyInfo.UNKNOWN, compiled.getZip())
            .entrySet()) {
      ImmutableList<String> referencedPrivateResources =
          resource.getValue().getReferencedResources().stream()
              .map(Reference::getName)
              .filter(privateResourceNames::contains)
              .collect(toImmutableList());

      if (!referencedPrivateResources.isEmpty()) {
        errorMessage
            .append(resource.getKey().toPrettyString())
            .append(" (defined in ")
            .append(resource.getValue().source().getPath())
            .append(") references external private resources ")
            .append(referencedPrivateResources)
            .append('\n');
      }
    }

    if (errorMessage.length() != 0) {
      throw new UserException(errorMessage.toString());
    }
  }

  private ValidateAndLinkResourcesAction() {}
}
