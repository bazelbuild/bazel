// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.hash;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bazel.repository.RepoDefinition;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.DeclaredExecGroup;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.LabelVisitationUtils;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RepositoryFetchException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.query.output.SyntheticAttributeHashCalculator;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil;
import com.google.devtools.build.lib.skyframe.TargetLoadingUtil.TargetAndErrorIfAny;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Computes the direct hash of an unconfigured target for {@code query --generate_hashes}: a hash of
 * everything about the target itself, but not of its dependencies, plus the labels of those
 * dependencies.
 *
 * <p>The direct hash covers the target's label and kind, its attribute values and rule definition
 * environment (for rules), and its content (for source files). Dependencies include those of
 * aspects, conservatively.
 *
 * <p>This function intentionally does not request the values of its dependencies: the unconfigured
 * target graph may contain cycles (which configuration transitions break in a real build), and
 * Skyframe does not allow cyclic dependencies. {@link TransitiveHashes} combines the direct hashes
 * along the dependency graph, handling cycles, outside of Skyframe.
 */
public final class DirectTargetHashFunction implements SkyFunction {
  public static final SkyFunctionName NAME =
      SkyFunctionName.createHermetic("QUERY_DIRECT_TARGET_HASH");

  private final BlazeDirectories directories;

  public DirectTargetHashFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  public static Key key(Label label) {
    return Key.create(label);
  }

  /** Key for {@link DirectTargetHashFunction}. */
  public static final class Key extends AbstractSkyKey<Label> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(Label arg) {
      super(arg);
    }

    private static Key create(Label arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return NAME;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /** The direct hash of a target and the labels of its dependencies, sorted. */
  public static final class Value implements SkyValue {
    private final byte[] directHash;
    private final ImmutableList<Label> deps;
    private final boolean toolchainType;

    @VisibleForTesting
    Value(byte[] directHash, ImmutableList<Label> deps) {
      this(directHash, deps, /* toolchainType= */ false);
    }

    @VisibleForTesting
    Value(byte[] directHash, ImmutableList<Label> deps, boolean toolchainType) {
      this.directHash = directHash;
      this.deps = deps;
      this.toolchainType = toolchainType;
    }

    public byte[] getDirectHash() {
      return directHash;
    }

    public ImmutableList<Label> getDeps() {
      return deps;
    }

    /**
     * Whether the target is a {@code toolchain_type}. {@link TransitiveHashes} adds edges from it
     * to the toolchains registered for it; they are not part of {@link #getDeps}, so that callers
     * that don't follow them don't need to load every registered toolchain.
     */
    public boolean isToolchainType() {
      return toolchainType;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof Value other
          && Arrays.equals(directHash, other.directHash)
          && deps.equals(other.deps)
          && toolchainType == other.toolchainType;
    }

    @Override
    public int hashCode() {
      return 31 * (31 * Arrays.hashCode(directHash) + deps.hashCode())
          + Boolean.hashCode(toolchainType);
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws DirectTargetHashFunctionException, InterruptedException {
    Label label = (Label) skyKey.argument();
    Target target;
    Package pkg;
    try {
      Object loaded = TargetLoadingUtil.loadTarget(env, label);
      if (!(loaded instanceof TargetAndErrorIfAny targetAndErrorIfAny)) {
        return null;
      }
      target = targetAndErrorIfAny.getTarget();
      pkg = targetAndErrorIfAny.getPackage();
    } catch (NoSuchThingException e) {
      throw new DirectTargetHashFunctionException(e);
    }

    // Sorted so the hash does not depend on attribute or visitation order.
    TreeSet<Label> deps = new TreeSet<>();
    LabelVisitationUtils.visitTarget(
        target,
        DependencyFilter.NO_NODEP_ATTRIBUTES,
        (fromTarget, attribute, toLabel) -> deps.add(toLabel));
    // The implementations of a toolchain() and the digests of the definitions of the repos they
    // come from.
    TreeMap<Label, byte[]> toolchainImplementationDigests = new TreeMap<>();
    if (target instanceof Rule rule && rule.getRuleClass().equals("toolchain")) {
      // The implementation of a toolchain() is a nodep label, because only the toolchain that
      // resolution picks is analyzed. Follow it: toolchain types depend on all their toolchains.
      AggregatingAttributeMapper attributes = AggregatingAttributeMapper.of(rule);
      ImmutableSet<Label> implementations =
          attributes.getReachableLabels("toolchain", /* includeSelectKeys= */ false);
      for (Label selectKey :
          attributes.getReachableLabels("toolchain", /* includeSelectKeys= */ true)) {
        if (!implementations.contains(selectKey)) {
          deps.add(selectKey);
        }
      }
      for (Label implementation : implementations) {
        // Toolchain implementations usually live in external repos with very large file sets
        // (compilers, sysroots), for every platform. Loading their packages would require fetching
        // all of those repos. Instead, hash the definitions of the implementation's repo and of
        // every repo it can see.
        byte[] digest;
        try {
          digest = repoClosureDefinitionDigest(env, implementation.getRepository());
        } catch (IOException e) {
          throw new DirectTargetHashFunctionException(e);
        }
        if (env.valuesMissing()) {
          return null;
        }
        if (digest != null) {
          toolchainImplementationDigests.put(implementation, digest);
        } else {
          deps.add(implementation);
        }
      }
    }
    if (target instanceof Rule rule) {
      // LabelVisitationUtils only visits the toolchain types of the default exec group.
      for (DeclaredExecGroup execGroup :
          rule.getRuleClassObject().getDeclaredExecGroups().values()) {
        for (ToolchainTypeRequirement toolchainType : execGroup.toolchainTypes()) {
          deps.add(toolchainType.toolchainType());
        }
      }
    }
    if (target instanceof Rule rule && rule.hasAspects()) {
      // Conservatively include the deps of every aspect that may apply, like
      // --aspect_deps=conservative.
      for (Attribute attribute :
          rule.getTransitions(DependencyFilter.NO_NODEP_ATTRIBUTES).keySet()) {
        for (Aspect aspect : attribute.getAspects(rule)) {
          AspectDefinition.forEachLabelDepFromAllAttributesOfAspect(
              aspect,
              DependencyFilter.ALL_DEPS,
              (aspectAttribute, aspectLabel) -> deps.add(aspectLabel));
          AspectDefinition definition = aspect.getDefinition();
          for (ToolchainTypeRequirement toolchainType : definition.getToolchainTypes()) {
            deps.add(toolchainType.toolchainType());
          }
          for (DeclaredExecGroup execGroup : definition.execGroups().values()) {
            for (ToolchainTypeRequirement toolchainType : execGroup.toolchainTypes()) {
              deps.add(toolchainType.toolchainType());
            }
          }
        }
      }
    }
    deps.remove(label);

    // Source files of external repos that only change when refetched are hashed by the digest of
    // the whole repo, rather than tracking each file in Skyframe.
    RepoContentDigestFunction.Value repoDigest = null;
    if (!label.getRepository().isMain()
        && (target instanceof InputFile || target instanceof Rule)) {
      try {
        repoDigest =
            (RepoContentDigestFunction.Value)
                env.getValueOrThrow(
                    RepoContentDigestFunction.key(label.getRepository()), IOException.class);
      } catch (IOException e) {
        throw new DirectTargetHashFunctionException(e);
      }
      if (repoDigest == null) {
        return null;
      }
      if (repoDigest.usesFileDigests()) {
        repoDigest = null;
      }
    }
    boolean coversPackageFiles = false;
    if (repoDigest != null && target instanceof Rule) {
      // The repo digest covers the rule's source files in this package, so they don't need to be
      // visited. This keeps the walk from visiting every file of large toolchain repos.
      coversPackageFiles =
          deps.removeIf(
              dep ->
                  dep.getPackageIdentifier().equals(label.getPackageIdentifier())
                      && pkg.getTargetOrNull(dep.getName()) instanceof InputFile);
    }

    SourceFileDigestFunction.Value fileDigest = null;
    if (target instanceof InputFile inputFile && repoDigest == null) {
      try {
        fileDigest =
            (SourceFileDigestFunction.Value)
                env.getValueOrThrow(
                    SourceFileDigestFunction.key(
                        RootedPath.toRootedPath(
                            inputFile.getPackageMetadata().sourceRoot(), inputFile.getPath())),
                    IOException.class);
      } catch (IOException e) {
        throw new DirectTargetHashFunctionException(e);
      }
      if (fileDigest == null) {
        return null;
      }
    }

    Fingerprint fp = new Fingerprint();
    fp.addString(label.getUnambiguousCanonicalForm());
    fp.addString(target.getTargetKind());
    if (target instanceof Rule rule) {
      fp.addString(
          SyntheticAttributeHashCalculator.compute(
              rule,
              /* serializedAttributes= */ pathIndependentAttributes(rule),
              /* extraDataForAttrHash= */ "",
              Hashing.sha256(),
              /* includeAttributeSourceAspects= */ false,
              /* includeStarlarkRuleEnv= */ true));
    } else if (target instanceof PackageGroup packageGroup) {
      fp.addStrings(packageGroup.getContainedPackages(/* includeDoubleSlash= */ true));
    }
    for (var implementation : toolchainImplementationDigests.entrySet()) {
      fp.addString("<toolchain implementation>");
      fp.addString(implementation.getKey().getUnambiguousCanonicalForm());
      fp.addBytes(implementation.getValue());
    }
    if (repoDigest != null && (target instanceof InputFile || coversPackageFiles)) {
      fp.addString("<repo contents>");
      fp.addBytes(repoDigest.getDigest());
    }
    if (fileDigest != null) {
      fp.addBytes(fileDigest.getDigest());
      if (fileDigest.isExecutable()) {
        // Only added when set, so that hashes of non-executable files are unaffected.
        fp.addString("<executable>");
      }
    }
    return new Value(
        fp.digestAndReset(),
        ImmutableList.copyOf(deps),
        target instanceof Rule rule && rule.getRuleClass().equals("toolchain_type"));
  }

  /**
   * Returns the attributes of {@code rule} whose string values contain the output base or the
   * workspace path, serialized with those paths replaced by placeholders.
   *
   * <p>Repo rules write such paths into BUILD files they generate, for instance into the error
   * message of a toolchain that can't be configured on the host. The hash shouldn't depend on where
   * the output base or the workspace is.
   */
  private ImmutableMap<Attribute, Build.Attribute> pathIndependentAttributes(Rule rule) {
    ImmutableMap.Builder<Attribute, Build.Attribute> result = ImmutableMap.builder();
    RawAttributeMapper attributes = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getAttributes()) {
      Object value = attributes.getRawAttributeValue(rule, attribute);
      Object replaced;
      if (value instanceof String string && containsAbsolutePath(string)) {
        replaced = replaceAbsolutePaths(string);
      } else if (value instanceof List<?> list
          && list.stream()
              .anyMatch(e -> e instanceof String string && containsAbsolutePath(string))) {
        replaced =
            list.stream()
                .map(e -> e instanceof String string ? replaceAbsolutePaths(string) : e)
                .collect(toImmutableList());
      } else {
        continue;
      }
      result.put(
          attribute,
          AttributeFormatter.getAttributeProto(
              attribute,
              replaced,
              /* explicitlySpecified= */ false,
              /* encodeBooleanAndTriStateAsIntegerAndString= */ false,
              /* sourceAspect= */ null,
              /* includeAttributeSourceAspects= */ false,
              LabelPrinter.legacy()));
    }
    return result.buildOrThrow();
  }

  private boolean containsAbsolutePath(String string) {
    return string.contains(directories.getOutputBase().getPathString())
        || string.contains(directories.getWorkspace().getPathString());
  }

  private String replaceAbsolutePaths(String string) {
    String outputBase = directories.getOutputBase().getPathString();
    String workspace = directories.getWorkspace().getPathString();
    // Replace the longer path first, in case one contains the other.
    if (outputBase.length() >= workspace.length()) {
      return string.replace(outputBase, "<output_base>").replace(workspace, "<workspace>");
    }
    return string.replace(workspace, "<workspace>").replace(outputBase, "<output_base>");
  }

  /**
   * Returns a digest of the definitions of {@code repo} and of every repo visible from it, or null
   * if the contents of {@code repo} can change without its definition changing, in which case the
   * caller follows the dependency instead. Also returns null if a Skyframe value is missing.
   *
   * <p>A repo's definition is its repo rule (including the transitive digest of the {@code .bzl}
   * files defining it), the attribute values it is called with and the contents of the files its
   * label attributes point to, such as patches and BUILD files. Repos those labels point to are
   * included like visible repos. Unlike hashing the repo's contents, this doesn't require fetching
   * the repo, which matters because toolchains are registered for every platform and their repos
   * are usually large downloads. Changes to a repo's contents that don't come with a change to its
   * definition are not seen; for repo rules that download checksummed archives, there are none.
   *
   * <p>Visible repos whose contents can change without a change to their definition (the main repo,
   * {@code local} and {@code configure} repos, and overridden repos, such as {@code @bazel_tools})
   * are skipped: changes in them don't reach the toolchain through this digest, except through
   * files that label attributes point to.
   */
  @Nullable
  private static byte[] repoClosureDefinitionDigest(Environment env, RepositoryName repo)
      throws InterruptedException, IOException {
    if (repo.isMain()) {
      return null;
    }
    RepositoryMappingValue mapping;
    try {
      mapping =
          (RepositoryMappingValue)
              env.getValueOrThrow(RepositoryMappingValue.key(repo), NoSuchThingException.class);
    } catch (NoSuchThingException e) {
      // The caller follows the label, which then reports the error.
      return null;
    }
    if (mapping == null || mapping.repositoryMapping() == null) {
      // A repo that doesn't exist: the caller follows the label, which then hashes as missing.
      return null;
    }
    Set<RepositoryName> seen = new HashSet<>();
    seen.add(repo);
    for (RepositoryName visible : mapping.repositoryMapping().entries().values()) {
      if (!visible.isMain()) {
        seen.add(visible);
      }
    }
    TreeMap<String, RepoDefinition> hashedDefinitions = new TreeMap<>();
    Set<RepositoryName> unhashedRepos = new HashSet<>();
    TreeSet<Label> labels = new TreeSet<>();
    List<RepositoryName> pending = new ArrayList<>(seen);
    while (!pending.isEmpty()) {
      ImmutableList<SkyKey> keys =
          pending.stream().map(RepoDefinitionValue::key).collect(toImmutableList());
      SkyframeLookupResult definitions = env.getValuesAndExceptions(keys);
      if (env.valuesMissing()) {
        return null;
      }
      pending = new ArrayList<>();
      for (SkyKey key : keys) {
        var name = (RepositoryName) key.argument();
        RepoDefinitionValue definition;
        try {
          definition =
              (RepoDefinitionValue)
                  definitions.getOrThrow(
                      key, BzlLoadFailedException.class, ExternalDepsException.class);
        } catch (BzlLoadFailedException | ExternalDepsException e) {
          // A repo that failed to be defined is treated like one whose contents can change without
          // its definition changing.
          definition = null;
        }
        if (!(definition instanceof RepoDefinitionValue.Found(RepoDefinition repoDefinition))
            || repoDefinition.repoRule().local()
            || repoDefinition.repoRule().configure()) {
          // Also covers repos that failed to be defined.
          if (name.equals(repo)) {
            return null;
          }
          unhashedRepos.add(name);
          continue;
        }
        hashedDefinitions.put(name.getName(), repoDefinition);
        // Includes default values, which the .bzl digest covers, but not the files they point to.
        for (String attribute : repoDefinition.getFieldNames()) {
          addLabels(repoDefinition.getValue(attribute), labels);
        }
      }
      for (Label label : labels) {
        RepositoryName labelRepo = label.getRepository();
        if (!labelRepo.isMain() && seen.add(labelRepo)) {
          pending.add(labelRepo);
        }
      }
    }

    // Files in repos hashed by their definition are covered by it. Hash the others by content.
    ImmutableList<Label> files =
        labels.stream()
            .filter(
                label ->
                    label.getRepository().isMain() || unhashedRepos.contains(label.getRepository()))
            .collect(toImmutableList());
    ImmutableList<SkyKey> packageKeys =
        files.stream()
            .<SkyKey>map(label -> PackageLookupValue.key(label.getPackageIdentifier()))
            .collect(toImmutableList());
    SkyframeLookupResult packages = env.getValuesAndExceptions(packageKeys);
    if (env.valuesMissing()) {
      return null;
    }
    List<SkyKey> fileKeys = new ArrayList<>();
    for (int i = 0; i < files.size(); i++) {
      PackageLookupValue packageLookup;
      try {
        packageLookup =
            (PackageLookupValue)
                packages.getOrThrow(
                    packageKeys.get(i),
                    BuildFileNotFoundException.class,
                    RepositoryFetchException.class,
                    InconsistentFilesystemException.class);
      } catch (BuildFileNotFoundException e) {
        packageLookup = null;
      } catch (RepositoryFetchException | InconsistentFilesystemException e) {
        throw new IOException(e.getMessage(), e);
      }
      if (packageLookup != null && packageLookup.packageExists()) {
        fileKeys.add(
            SourceFileDigestFunction.key(
                RootedPath.toRootedPath(packageLookup.getRoot(), files.get(i).toPathFragment())));
      } else {
        fileKeys.add(null);
      }
    }
    SkyframeLookupResult fileDigests =
        env.getValuesAndExceptions(fileKeys.stream().filter(k -> k != null).toList());

    Fingerprint fp = new Fingerprint();
    for (var entry : hashedDefinitions.entrySet()) {
      RepoRule repoRule = entry.getValue().repoRule();
      fp.addString(entry.getKey());
      fp.addString(repoRule.id().bzlFileLabel().getUnambiguousCanonicalForm());
      fp.addString(repoRule.id().ruleName());
      fp.addBytes(repoRule.transitiveBzlDigest());
      Dict<String, Object> attributes = entry.getValue().attrValues().attributes();
      fp.addInt(attributes.size());
      for (String attribute : new TreeSet<>(attributes.keySet())) {
        fp.addString(attribute);
        fp.addString(Starlark.repr(attributes.get(attribute), StarlarkSemantics.DEFAULT));
      }
    }
    fp.addInt(files.size());
    for (int i = 0; i < files.size(); i++) {
      fp.addString(files.get(i).getUnambiguousCanonicalForm());
      if (fileKeys.get(i) == null) {
        // Repo rules fail on such labels, so their outputs don't depend on anything else.
        fp.addString("<no package>");
        continue;
      }
      var digest =
          (SourceFileDigestFunction.Value)
              fileDigests.getOrThrow(fileKeys.get(i), IOException.class);
      if (digest == null) {
        return null;
      }
      fp.addBytes(digest.getDigest());
    }
    return fp.digestAndReset();
  }

  /** Adds the labels in a repo rule attribute value, which may be a list or dict of them. */
  private static void addLabels(Object value, Set<Label> labels) {
    if (value instanceof Label label) {
      labels.add(label);
    } else if (value instanceof Map<?, ?> map) {
      for (Map.Entry<?, ?> entry : map.entrySet()) {
        addLabels(entry.getKey(), labels);
        addLabels(entry.getValue(), labels);
      }
    } else if (value instanceof Iterable<?> iterable) {
      for (Object element : iterable) {
        addLabels(element, labels);
      }
    }
  }

  private static final class DirectTargetHashFunctionException extends SkyFunctionException {
    DirectTargetHashFunctionException(NoSuchThingException e) {
      super(e, Transience.PERSISTENT);
    }

    DirectTargetHashFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
