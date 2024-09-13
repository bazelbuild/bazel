// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkOptions;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.SymbolGenerator;

/**
 * A helper class for android rules.
 *
 * <p>Helps create the java compilation as well as handling the exporting of the java compilation
 * artifacts to the other rules.
 */
public class AndroidCommon {

  private static final ImmutableSet<String> TRANSITIVE_ATTRIBUTES =
      ImmutableSet.of("application_resources", "deps", "exports");

  static <T extends Info> Iterable<T> getTransitivePrerequisites(
      RuleContext ruleContext, BuiltinProvider<T> key) {
    ImmutableList.Builder<List<T>> builder = ImmutableList.builder();
    AttributeMap attributes = ruleContext.attributes();
    for (String attr : TRANSITIVE_ATTRIBUTES) {
      if (attributes.has(attr, BuildType.LABEL_LIST)) {
        List<T> prereqs = ruleContext.getPrerequisites(attr, key);
        if (!prereqs.isEmpty()) {
          builder.add(prereqs);
        }
      }
    }
    return Iterables.concat(builder.build());
  }

  private NestedSet<Artifact> transitiveNeverlinkLibraries =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);


  /**
   * Collects the transitive neverlink dependencies.
   *
   * @param ruleContext the context of the rule neverlink deps are to be computed for
   * @param deps the targets to be treated as dependencies
   * @param runtimeJars the runtime jars produced by the rule (non-transitive)
   * @return a nested set of the neverlink deps.
   */
  public static NestedSet<Artifact> collectTransitiveNeverlinkLibraries(
      RuleContext ruleContext,
      Iterable<? extends TransitiveInfoCollection> deps,
      NestedSet<Artifact> runtimeJars)
      throws RuleErrorException {
    NestedSetBuilder<Artifact> neverlinkedRuntimeJars = NestedSetBuilder.naiveLinkOrder();
    for (AndroidNeverLinkLibrariesProvider provider :
        AnalysisUtils.getProviders(deps, AndroidNeverLinkLibrariesProvider.PROVIDER)) {
      neverlinkedRuntimeJars.addTransitive(provider.getTransitiveNeverLinkLibraries());
    }

    if (JavaCommon.isNeverLink(ruleContext)) {
      neverlinkedRuntimeJars.addTransitive(runtimeJars);
      for (TransitiveInfoCollection dep : deps) {
        neverlinkedRuntimeJars.addTransitive(JavaInfo.transitiveRuntimeJars(dep));
      }
    }
    return neverlinkedRuntimeJars.build();
  }

  /**
   * Gets the Java package for the current target.
   *
   * @deprecated If no custom_package is specified, this method will derive the Java package from
   *     the package path, even if that path is not a valid Java path. Use {@code
   *     AndroidManifest#getAndroidPackage(RuleContext)}} instead.
   */
  @Deprecated
  public static String getJavaPackage(RuleContext ruleContext) {
    AttributeMap attributes = ruleContext.attributes();
    if (attributes.isAttributeValueExplicitlySpecified("custom_package")) {
      return attributes.get("custom_package", Type.STRING);
    }
    return getDefaultJavaPackage(ruleContext.getRule());
  }

  private static String getDefaultJavaPackage(Rule rule) {
    PathFragment nameFragment = rule.getPackage().getNameFragment();
    String packageName = JavaUtil.getJavaFullClassname(nameFragment);
    if (packageName != null) {
      return packageName;
    } else {
      // This is a workaround for libraries that don't follow the standard Bazel package format
      return nameFragment.getPathString().replace('/', '.');
    }
  }

  @Nullable
  static PathFragment getSourceDirectoryRelativePathFromResource(Artifact resource) {
    PathFragment resourceDir = AndroidResources.findResourceDir(resource);
    if (resourceDir == null) {
      return null;
    }
    return trimTo(resource.getRootRelativePath(), resourceDir);
  }

  /**
   * Finds the rightmost occurrence of the needle and returns subfragment of the haystack from left
   * to the end of the occurrence inclusive of the needle.
   *
   * <pre>
   * `Example:
   *   Given the haystack:
   *     res/research/handwriting/res/values/strings.xml
   *   And the needle:
   *     res
   *   Returns:
   *     res/research/handwriting/res
   * </pre>
   */
  static PathFragment trimTo(PathFragment haystack, PathFragment needle) {
    if (needle.equals(PathFragment.EMPTY_FRAGMENT)) {
      return haystack;
    }
    List<String> needleSegments = needle.splitToListOfSegments();
    // Compute the overlap offset for duplicated parts of the needle.
    int[] overlap = new int[needleSegments.size() + 1];
    // Start overlap at -1, as it will cancel out the increment in the search.
    // See http://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm for the
    // details.
    overlap[0] = -1;
    for (int i = 0, j = -1; i < needleSegments.size(); j++, i++, overlap[i] = j) {
      while (j >= 0 && !needleSegments.get(i).equals(needleSegments.get(j))) {
        // Walk the overlap until the bound is found.
        j = overlap[j];
      }
    }
    // TODO(corysmith): reverse the search algorithm.
    // Keep the index of the found so that the rightmost index is taken.
    List<String> haystackSegments = haystack.splitToListOfSegments();
    int found = -1;
    for (int i = 0, j = 0; i < haystackSegments.size(); i++) {

      while (j >= 0 && !haystackSegments.get(i).equals(needleSegments.get(j))) {
        // Not matching, walk the needle index to attempt another match.
        j = overlap[j];
      }
      j++;
      // Needle index is exhausted, so the needle must match.
      if (j == needleSegments.size()) {
        // Record the found index + 1 to be inclusive of the end index.
        found = i + 1;
        // Subtract one from the needle index to restart the search process
        j = j - 1;
      }
    }
    if (found != -1) {
      // Return the subsection of the haystack.
      return haystack.subFragment(0, found);
    }
    throw new IllegalArgumentException(String.format("%s was not found in %s", needle, haystack));
  }

  public static NestedSetBuilder<Artifact> collectTransitiveNativeLibs(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> transitiveNativeLibs = NestedSetBuilder.naiveLinkOrder();
    Iterable<AndroidNativeLibsInfo> infos =
        getTransitivePrerequisites(ruleContext, AndroidNativeLibsInfo.PROVIDER);
    for (AndroidNativeLibsInfo nativeLibsZipsInfo : infos) {
      transitiveNativeLibs.addTransitive(nativeLibsZipsInfo.getNativeLibs());
    }
    return transitiveNativeLibs;
  }

  static boolean getExportsManifest(RuleContext ruleContext) {
    // AndroidLibraryBaseRule has exports_manifest but AndroidBinaryBaseRule does not.
    // ResourceContainers are built for both, so we must check if exports_manifest is present.
    if (!ruleContext.attributes().has("exports_manifest", BuildType.TRISTATE)) {
      return false;
    }
    TriState attributeValue = ruleContext.attributes().get("exports_manifest", BuildType.TRISTATE);

    // If the rule does not have the Android configuration fragment, we default to false.
    boolean exportsManifestDefault =
        ruleContext.isLegalFragment(AndroidConfiguration.class)
            && ruleContext.getFragment(AndroidConfiguration.class).getExportsManifestDefault();
    return attributeValue == TriState.YES
        || (attributeValue == TriState.AUTO && exportsManifestDefault);
  }

  /** Returns the artifact for the debug key for signing the APK. */
  static ImmutableList<Artifact> getApkDebugSigningKeys(RuleContext ruleContext) {
    ImmutableList<Artifact> keys =
        ruleContext.getPrerequisiteArtifacts("debug_signing_keys").list();
    if (!keys.isEmpty()) {
      return keys;
    }
    return ImmutableList.of(ruleContext.getPrerequisiteArtifact("debug_key"));
  }

  public NestedSet<Artifact> getTransitiveNeverLinkLibraries() {
    return transitiveNeverlinkLibraries;
  }

  static CcInfo getCcInfo(
      final Collection<? extends TransitiveInfoCollection> deps,
      final ImmutableList<String> linkOpts,
      Label label,
      SymbolGenerator<?> symbolGenerator)
      throws RuleErrorException {

    CcLinkingContext ccLinkingContext =
        CcLinkingContext.builder()
            .setOwner(label)
            .addUserLinkFlags(
                ImmutableList.of(LinkOptions.of(linkOpts, symbolGenerator.generate())))
            .build();

    CcInfo linkoptsCcInfo = CcInfo.builder().setCcLinkingContext(ccLinkingContext).build();

    ImmutableList<CcInfo> ccInfos =
        Streams.concat(
                Stream.of(linkoptsCcInfo),
                JavaInfo.ccInfos(deps).stream(),
                AnalysisUtils.getProviders(deps, AndroidCcLinkParamsProvider.PROVIDER).stream()
                    .map(AndroidCcLinkParamsProvider::getLinkParams),
                AnalysisUtils.getProviders(deps, CcInfo.PROVIDER).stream())
            .collect(toImmutableList());

    return CcInfo.merge(ccInfos);
  }

  /** Returns {@link AndroidConfiguration} in given context. */
  public static AndroidConfiguration getAndroidConfig(RuleContext context) {
    return context.getConfiguration().getFragment(AndroidConfiguration.class);
  }

  /**
   * Gets the transitive support APKs required by this rule through the {@code support_apks}
   * attribute.
   */
  static NestedSet<Artifact> getSupportApks(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> supportApks = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("support_apks")) {
      ApkInfo apkProvider = dep.get(ApkInfo.PROVIDER);
      FileProvider fileProvider = dep.getProvider(FileProvider.class);
      // If ApkInfo is present, do not check FileProvider for .apk files. For example,
      // android_binary creates a FileProvider containing both the signed and unsigned APKs.
      if (apkProvider != null) {
        supportApks.add(apkProvider.getApk());
      } else if (fileProvider != null) {
        // The rule definition should enforce that only .apk files are allowed, however, it can't
        // hurt to double check.
        supportApks.addAll(
            FileType.filter(fileProvider.getFilesToBuild().toList(), AndroidRuleClasses.APK));
      }
    }
    return supportApks.build();
  }

  private static class FlagMatcher implements Predicate<String> {
    private final ImmutableList<String> matching;

    FlagMatcher(ImmutableList<String> matching) {
      this.matching = matching;
    }

    @Override
    public boolean apply(String input) {
      for (String match : matching) {
        if (input.contains(match)) {
          return true;
        }
      }
      return false;
    }
  }

  private enum FlagConverter implements Function<String, String> {
    DX_TO_DEXBUILDER;

    @Override
    public String apply(String input) {
      return input.replace("--no-", "--no");
    }
  }

  private static ImmutableSet<String> normalizeDexopts(Iterable<String> tokenizedDexopts) {
    // Sort and use ImmutableSet to drop duplicates and get fixed (sorted) order.  Fixed order is
    // important so we generate one dex archive per set of flag in create() method, regardless of
    // how those flags are listed in all the top-level targets being built.
    return Streams.stream(tokenizedDexopts)
        .map(FlagConverter.DX_TO_DEXBUILDER)
        .sorted()
        .collect(ImmutableSet.toImmutableSet()); // collector with dedupe
  }

  /**
   * Derives options to use in DexFileMerger actions from the given context and dx flags, where the
   * latter typically come from a {@code dexopts} attribute on a top-level target.
   */
  public static ImmutableSet<String> mergerDexopts(
      RuleContext ruleContext, Iterable<String> tokenizedDexopts) {
    // We don't need an ordered set but might as well.  Note we don't need to worry about coverage
    // builds since the merger doesn't use --no-locals.
    return normalizeDexopts(
        Iterables.filter(
            tokenizedDexopts,
            new FlagMatcher(getAndroidConfig(ruleContext).getDexoptsSupportedInDexMerger())));
  }
}
