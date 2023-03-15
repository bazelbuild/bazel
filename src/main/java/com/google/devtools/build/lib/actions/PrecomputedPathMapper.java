package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.PathStripper.PathMapper;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * An implementation of {@link PathMapper} that replaces the configuration segment of paths of
 * generated files with synthetic values, backed by a fixed mapping for the input files of an
 * action.
 */
public final class PrecomputedPathMapper implements PathMapper {

  // 25 * log_2(32) = 125 bits ought to be enough to prevent collisions.
  private static final int MAX_DIGEST_CHARS = 25;

  private final ImmutableMap<PathFragment, PathFragment> execPathMapping;
  private final PathFragment primaryOutputRoot;
  private final PathFragment mappedPrimaryOutputRoot;
  private final PathFragment outputDir;

  private PrecomputedPathMapper(ImmutableMap<PathFragment, PathFragment> execPathMapping,
      Artifact primaryOutput) {
    this.execPathMapping = execPathMapping;
    // See strip(PathFragment execPath) for an explanation of the output path mapping.
    this.primaryOutputRoot = primaryOutput.getRoot().getExecPath();
    this.mappedPrimaryOutputRoot = execPathWithSyntheticConfig(primaryOutputRoot, "out");
    this.outputDir = primaryOutputRoot.subFragment(0, 1);
  }

  public static PathMapper noop() {
    return NOOP;
  }

  /**
   * Creates a {@link PathMapper} that replaces the config segment in the exec paths of all
   * generated artifacts with a synthetic value derived from the artifact's contents.
   *
   * <p>As a result, if the inputs to a spawn using this path mapping are the same, then the spawn's
   * command line will be identical regardless of the inputs' configurations, which improves remote
   * caching.
   *
   * @param artifactExpander an {@link ArtifactExpander} used to expand middleman artifacts
   * @param metadataHandler a {@link MetadataHandler} used to obtain digests for inputs
   * @param inputs the set of input artifacts to the action
   * @param primaryOutput the primary output of the action
   * @return a {@link PathMapper}
   * @throws CommandLineExpansionException if getting the digest of an input artifact fails
   */
  public static PathMapper createContentBased(
      ArtifactExpander artifactExpander,
      MetadataHandler metadataHandler,
      NestedSet<Artifact> inputs,
      Artifact primaryOutput) throws CommandLineExpansionException {
    Optional<List<DerivedArtifact>> artifactsToMap = computeArtifactsToMap(artifactExpander,
        inputs);
    if (artifactsToMap.isEmpty()) {
      return NOOP;
    }
    ImmutableMap.Builder<PathFragment, PathFragment> execPathMapping = new Builder<>();
    for (Artifact input : artifactsToMap.get()) {
      byte[] digest;
      try {
        digest = metadataHandler.getMetadata(input).getDigest();
      } catch (IOException e) {
        throw new CommandLineExpansionException(
            String.format("Failed to get metadata for %s", input), e);
      }
      if (digest == null) {
        return NOOP;
      }
      execPathMapping.put(input.getExecPath(),
          execPathWithSyntheticConfig(input.getExecPath(), prettifyDigest(digest)));
    }
    return new PrecomputedPathMapper(execPathMapping.build(), primaryOutput);
  }

  private static Optional<List<DerivedArtifact>> computeArtifactsToMap(
      ArtifactExpander artifactExpander, NestedSet<Artifact> inputs) {
    List<DerivedArtifact> inputsToMap = new ArrayList<>();
    for (Artifact input : inputs.toList()) {
      if (!(input instanceof DerivedArtifact)) {
        // Source file paths have an empty root and in particular contain no config part. They thus
        // don't require path mapping.
        continue;
      }
      if (input.isFileset()) {
        // Filesets are not supported.
        return Optional.empty();
      }
      // We only expand middleman artifacts as TreeArtifactFiles are mapped based on their parent's
      // hash and exec path.
      if (input.isMiddlemanArtifact()) {
        List<Artifact> expandedArtifacts = new ArrayList<>();
        artifactExpander.expand(input, expandedArtifacts);
        for (Artifact artifact : expandedArtifacts) {
          if (artifact instanceof DerivedArtifact) {
            inputsToMap.add((DerivedArtifact) artifact);
          }
        }
      } else {
        inputsToMap.add((DerivedArtifact) input);
      }
    }
    return Optional.of(inputsToMap);
  }

  private static PathFragment execPathWithSyntheticConfig(PathFragment execPath, String config) {
    // TODO: Support experimental_sibling_repository_layout, which uses output roots such as
    //  __main__/bazel-out/k8-fastbuild/bin instead of keeping the repository name out of the root.
    return execPath.subFragment(0, 1)
        .getRelative(config)
        .getRelative(execPath.subFragment(2));
  }

  private static String prettifyDigest(byte[] digest) {
    String fullDigest = BaseEncoding.base32().lowerCase().omitPadding().encode(digest);
    return fullDigest.substring(0, Math.min(fullDigest.length(), MAX_DIGEST_CHARS));
  }

  @Override
  public PathFragment strip(PathFragment execPath) {
    if (!execPath.startsWith(outputDir)) {
      // execPath belongs to a source file and doesn't contain a config part - no need to map
      // anything.
      return execPath;
    }
    PathFragment remappedPath = execPathMapping.get(execPath);
    if (remappedPath != null) {
      // execPath belongs to a generated input file.
      return remappedPath;
    }
    // execPath does not belong to an input artifact and thus must be an output-like artifact, i.e.
    // an honest action output or a derived path such as e.g. a param file. All such artifacts live
    // in a single configuration: the configuration of the target that registered the action. We can
    // thus replace the config part in the root with a fixed string such as "out".
    return mappedPrimaryOutputRoot.getRelative(execPath.relativeTo(primaryOutputRoot));
  }

  @Override
  public PathFragment unstrip(PathFragment execPath) {
    return primaryOutputRoot.getRelative(execPath.relativeTo(mappedPrimaryOutputRoot));
  }

  @Override
  public String getExecPathString(ActionInput artifact) {
    if (!(artifact instanceof DerivedArtifact)) {
      return artifact.getExecPathString();
    }
    return strip((DerivedArtifact) artifact);
  }


  @Override
  public String strip(DerivedArtifact artifact) {
    return strip(artifact.getExecPath()).getPathString();
  }

  @Override
  public List<String> stripCustomStarlarkArgs(List<String> args) {
    return args;
  }
}
