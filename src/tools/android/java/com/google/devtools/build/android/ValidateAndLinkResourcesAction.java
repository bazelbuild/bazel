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

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.XmlNode;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.ProtoXmlUtils;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Performs resource validation and static linking for compiled android resources. */
public class ValidateAndLinkResourcesAction {

  /** Action configuration options. */
  public static class Options extends OptionsBase {
    @Option(
        name = "compiled",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.ExistingPathConverter.class,
        category = "input",
        help = "Compiled resources to link.",
        deprecationWarning = "Use --resources.")
    // TODO(b/64570523): Still used by blaze. Will be removed as part of the command line cleanup.
    @Deprecated
    public Path compiled;

    @Option(
        name = "compiledDep",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "",
        converter = Converters.PathListConverter.class,
        category = "input",
        allowMultiple = true,
        help = "Compiled resource dependencies to link.")
    public List<Path> compiledDeps;

    @Option(
        name = "manifest",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.ExistingPathConverter.class,
        category = "input",
        help = "Manifest for the library.",
        deprecationWarning = "Use --resources.")
    // TODO(b/64570523): Still used by blaze. Will be removed as part of the command line cleanup.
    @Deprecated
    public Path manifest;

    @Option(
        name = "resources",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.CompiledResourcesConverter.class,
        category = "input",
        help = "Compiled resources to link.")
    public CompiledResources resources;

    // TODO(b/64570523): remove this flag when it is no longer used.
    @Option(
        name = "libraries",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.StaticLibraryListConverter.class,
        category = "input",
        help = "Static libraries to link against. Deprecated, use --library")
    public List<StaticLibrary> deprecatedLibraries;

    @Option(
        name = "library",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.StaticLibraryConverter.class,
        category = "input",
        allowMultiple = true,
        help = "Static libraries to link against.")
    public List<StaticLibrary> libraries;

    @Option(
        name = "packageForR",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        category = "input",
        help = "Package for the resources.")
    public String packageForR;

    @Option(
        name = "staticLibraryOut",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.PathConverter.class,
        category = "output",
        help = "Static library produced.")
    public Path staticLibraryOut;

    @Option(
        name = "rTxtOut",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        defaultValue = "null",
        converter = Converters.PathConverter.class,
        category = "output",
        help = "R.txt out.")
    public Path rTxtOut;

    @Option(
        name = "sourceJarOut",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = Converters.PathConverter.class,
        defaultValue = "null",
        category = "output",
        help = "Generated java classes from the resources.")
    public Path sourceJarOut;
  }

  public static void main(String[] args) throws Exception {
    OptionsParser optionsParser =
        OptionsParser.builder()
            .optionsClasses(Options.class, Aapt2ConfigOptions.class)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    optionsParser.parse(args);

    Options options = optionsParser.getOptions(Options.class);
    final Aapt2ConfigOptions aapt2Options = optionsParser.getOptions(Aapt2ConfigOptions.class);
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
      List<CompiledResources> deps =
          options.compiledDeps.stream()
              .map(CompiledResources::from)
              .collect(ImmutableList.toImmutableList());
      profiler.recordEndOf("manifest").startTask("validate");

      // TODO(b/146663858): distinguish direct/transitive deps for "strict deps".
      // TODO(b/128711690): validate AndroidManifest.xml
      checkVisibilityOfResourceReferences(
          /*androidManifest=*/ XmlNode.getDefaultInstance(), resources, deps);

      profiler.recordEndOf("validate").startTask("link");
      ResourceLinker.create(aapt2Options.aapt2, executorService, scopedTmp.getPath())
          .profileUsing(profiler)
          // NB: these names are really confusing.
          //   .dependencies is meant for linking in android.jar
          //   .include is meant for regular dependencies
          .dependencies(Optional.ofNullable(options.deprecatedLibraries).orElse(options.libraries))
          .include(deps)
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
                        /*includeFileContentsForValidation=*/ false)
                        .read(DependencyInfo.UNKNOWN, cr.getZip())
                        .entrySet()
                        .stream())
            .filter(entry -> entry.getValue().getVisibility() == Visibility.PRIVATE)
            .map(entry -> ((FullyQualifiedName) entry.getKey()).asQualifiedReference())
            .collect(ImmutableSet.toImmutableSet());

    StringBuilder errorMessage = new StringBuilder();
    {
      List<String> referencedPrivateResources =
          ProtoXmlUtils.getAllResourceReferences(androidManifest).stream()
              .map(Reference::getName)
              .filter(privateResourceNames::contains)
              .collect(ImmutableList.toImmutableList());
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
      List<String> referencedPrivateResources =
          resource.getValue().getReferencedResources().stream()
              .map(Reference::getName)
              .filter(privateResourceNames::contains)
              .collect(ImmutableList.toImmutableList());

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
}
