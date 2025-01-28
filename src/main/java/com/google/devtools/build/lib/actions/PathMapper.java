// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.CommandLineItem.ExceptionlessMapFn;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CheckReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Support for mapping config parts of exec paths in an action's command line as well as when
 * staging its inputs and outputs for execution, with the aim of making the resulting {@link Spawn}
 * more cacheable.
 *
 * <p>Actions that want to support path mapping should use {@link
 * com.google.devtools.build.lib.analysis.actions.PathMappers}.
 *
 * <p>An example of an implementing class is {@link
 * com.google.devtools.build.lib.analysis.actions.StrippingPathMapper}, which removes the config
 * part (e.g. "k8-fastbuild") from exec paths to allow for cross-configuration cache hits.
 */
public interface PathMapper {
  /**
   * Retrieve the {@link PathMapper} instance stored in the given {@link StarlarkSemantics} via
   * {@link #storeIn(StarlarkSemantics)}.
   */
  static PathMapper loadFrom(StarlarkSemantics semantics) {
    return semantics.get(PathMapperConstants.SEMANTICS_KEY);
  }

  /**
   * Creates a new {@link StarlarkSemantics} instance which causes all Starlark threads using it to
   * automatically apply this {@link PathMapper} to all struct fields of {@link
   * com.google.devtools.build.lib.starlarkbuildapi.FileApi}.
   *
   * <p>This is meant to be used when evaluating user-defined callbacks to Starlark variants of
   * custom command lines that are evaluated during the execution phase.
   *
   * <p>Since any unmapped path appearing in a command line will prevent cross-configuration cache
   * hits, this mapping is applied automatically instead of requiring users to explicitly map all
   * paths themselves. As an added benefit, this allows actions to opt into path mapping without
   * actual changes to their command line code.
   */
  @CheckReturnValue
  default StarlarkSemantics storeIn(StarlarkSemantics semantics) {
    // This in particular covers the case where the semantics do not have a path mapper yet and this
    // is NOOP.
    if (semantics.get(PathMapperConstants.SEMANTICS_KEY) == this) {
      return semantics;
    }
    return new StarlarkSemantics(
        semantics.toBuilder().set(PathMapperConstants.SEMANTICS_KEY, this).build()) {
      // The path mapper doesn't affect which fields or methods are available on any given Starlark
      // object; it just affects the behavior of certain methods on Artifact. We thus preserve the
      // original semantics as a cache key. Otherwise, even if PathMapper#equals returned true for
      // each two non-NOOP instances, cache lookups in CallUtils would result in frequent
      // comparisons of equal but not reference equal semantics maps, which regresses CPU (~7% on
      // a benchmark with ~10 semantics options).
      @Override
      public StarlarkSemantics getStarlarkClassDescriptorCacheKey() {
        return semantics;
      }
    };
  }

  /** Returns the instance to use during action key computation. */
  static PathMapper forActionKey(CoreOptions.OutputPathsMode outputPathsMode) {
    return outputPathsMode == CoreOptions.OutputPathsMode.OFF
        ? NOOP
        : PathMapperConstants.FOR_FINGERPRINTING;
  }

  /**
   * Returns the exec path with the path mapping applied.
   *
   * <p>Path mappers may return paths with different roots for two paths that have the same root
   * (e.g., they may map an artifact at {@code bazel-out/k8-fastbuild/bin/pkg/foo} to {@code
   * bazel-out/<hash of the file>/bin/pkg/foo}). Paths of artifacts that should share the same
   * parent directory, such as runfiles or tree artifact files, should thus be derived from the
   * mapped path of their parent.
   */
  PathFragment map(PathFragment execPath);

  /** Returns the exec path of the input with the path mapping applied. */
  default String getMappedExecPathString(ActionInput artifact) {
    return map(artifact.getExecPath()).getPathString();
  }

  /**
   * We don't yet have a Starlark API for mapping paths in command lines. Simple Starlark calls like
   * {@code args.add(arg_name, file_path} are automatically handled. But calls that involve custom
   * Starlark code require deeper API support that remains a TODO.
   *
   * <p>This method allows implementations to hard-code support for specific command line entries
   * for specific Starlark actions.
   */
  default ArgChunk mapCustomStarlarkArgs(ArgChunk chunk) {
    return chunk;
  }

  /**
   * Returns the {@link MapFn} to apply to a vector argument with the given previous String argument
   * in a {@link com.google.devtools.build.lib.analysis.actions.CustomCommandLine}.
   *
   * <p>For example, if the previous argument is {@code "--foo"}, this method should return a {@link
   * MapFn} that maps the next arguments to the correct path, potentially mapping them if "--foo"
   * requires it.
   *
   * <p>This is used to map paths obtained via location expansion in native rules, which returns a
   * list of strings rather than a structured command line.
   *
   * <p>By default, this method returns {@link MapFn#DEFAULT}.
   */
  default ExceptionlessMapFn<Object> getMapFn(@Nullable String previousFlag) {
    return MapFn.DEFAULT;
  }

  /** Heuristically maps all path-like strings in the given argument. */
  default String mapHeuristically(String arg) {
    return arg;
  }

  /**
   * Returns a {@link FileRootApi} representing the new root of the given artifact after mapping.
   *
   * <p>All objects returned by this method must be {@link Comparable} among each other.
   */
  default FileRootApi mapRoot(Artifact artifact) {
    ArtifactRoot root = artifact.getRoot();
    if (root.isSourceRoot()) {
      // Source roots' paths are never mapped, but we still need to wrap them in a
      // MappedArtifactRoot to ensure correct Starlark comparison behavior.
      return PathMapperConstants.mappedSourceRoots.get(root);
    }
    // It would *not* be correct to just apply #map to the exec path of the root: The root part of
    // the mapped exec path of this artifact may depend on its complete exec path as well as on e.g.
    // the digest of the artifact.
    PathFragment execPath = artifact.getExecPath();
    PathFragment mappedExecPath = map(execPath);
    // map never changes the root-relative part of the exec path, so we can remove that suffix to
    // get the mapped root part.
    int rootRelativeSegmentCount = execPath.segmentCount() - root.getExecPath().segmentCount();
    PathFragment mappedRootExecPath =
        mappedExecPath.subFragment(0, mappedExecPath.segmentCount() - rootRelativeSegmentCount);
    return new MappedArtifactRoot(mappedRootExecPath);
  }

  /**
   * Returns {@code true} if the mapper is known to map all paths identically.
   *
   * <p>Can be used by actions to skip additional work that isn't needed if path mapping is not
   * enabled.
   */
  default boolean isNoop() {
    return this == NOOP;
  }

  /**
   * Returns an opaque object whose equality class should encode all information that goes into the
   * behavior of the {@link #map(PathFragment)} function of this path mapper. This is used as a key
   * for in-memory caches.
   *
   * <p>The default implementation returns the {@link Class} of the path mapper.
   */
  default Object cacheKey() {
    return this.getClass();
  }

  /** A {@link PathMapper} that doesn't change paths. */
  PathMapper NOOP =
      new PathMapper() {
        @Override
        public PathFragment map(PathFragment execPath) {
          return execPath;
        }

        @Override
        public FileRootApi mapRoot(Artifact artifact) {
          return artifact.getRoot();
        }
      };

  /** A {@link FileRootApi} returned by {@link PathMapper#mapRoot(Artifact)}. */
  @StarlarkBuiltin(
      name = "mapped_root",
      category = DocCategory.BUILTIN,
      doc = "A root for files that have been subject to path mapping")
  final class MappedArtifactRoot implements FileRootApi, Comparable<MappedArtifactRoot> {
    private final PathFragment mappedRootExecPath;

    public MappedArtifactRoot(PathFragment mappedRootExecPath) {
      this.mappedRootExecPath = mappedRootExecPath;
    }

    @Override
    public String getExecPathString() {
      return mappedRootExecPath.getPathString();
    }

    @Override
    public int compareTo(MappedArtifactRoot otherRoot) {
      return mappedRootExecPath.compareTo(otherRoot.mappedRootExecPath);
    }

    @Override
    public boolean equals(Object obj) {
      // Per the contract of PathMapper#map, mapped roots never have exec paths that are equal to
      // exec paths of non-mapped roots, that is, of instances of ArtifactRoot. Thus, it is correct
      // for both equals implementations to return false if the other object is not an instance of
      // the respective class.
      if (!(obj instanceof MappedArtifactRoot other)) {
        return false;
      }
      return mappedRootExecPath.equals(other.mappedRootExecPath);
    }

    @Override
    public int hashCode() {
      return mappedRootExecPath.hashCode();
    }

    @Override
    public String toString() {
      return mappedRootExecPath + " [mapped]";
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<mapped root>");
    }

    @Override
    public boolean isImmutable() {
      return true;
    }
  }
}

/** Holder class for symbols used by the PathMapper interface that shouldn't be public. */
final class PathMapperConstants {

  public static final StarlarkSemantics.Key<PathMapper> SEMANTICS_KEY =
      new StarlarkSemantics.Key<>("path_mapper", PathMapper.NOOP);
  public static final LoadingCache<ArtifactRoot, PathMapper.MappedArtifactRoot> mappedSourceRoots =
      Caffeine.newBuilder()
          .weakKeys()
          .build(sourceRoot -> new PathMapper.MappedArtifactRoot(sourceRoot.getExecPath()));

  private static final PathFragment BAZEL_OUT = PathFragment.create("bazel-out/");
  private static final PathFragment BLAZE_OUT = PathFragment.create("blaze-out/");

  /**
   * A special instance for use in {@link AbstractAction#computeKey} when path mapping is generally
   * enabled for an action.
   *
   * <p>When computing an action key, the following approaches to taking path mapping into account
   * do <b>not</b> work:
   *
   * <ul>
   *   <li>Using the actual path mapper is prohibitive since constructing it requires checking for
   *       collisions among the action input's paths when computing the action key, which flattens
   *       the input depsets of all actions that opt into path mapping and also increases CPU usage.
   *   <li>Unconditionally using {@link
   *       com.google.devtools.build.lib.analysis.actions.StrippingPathMapper} can result in stale
   *       action keys when an action is opted out of path mapping at execution time due to input
   *       path collisions after stripping. See path_mapping_test for an example.
   *   <li>Using {@link PathMapper#NOOP} does not distinguish between map_each results built from
   *       strings and those built from {@link
   *       com.google.devtools.build.lib.starlarkbuildapi.FileApi#getExecPathStringForStarlark}.
   *       While the latter will be mapped at execution time, the former won't, resulting in the
   *       same digest for actions that behave differently at execution time. This is covered by
   *       tests in StarlarkRuleImplementationFunctionsTest.
   * </ul>
   *
   * <p>Instead, we use a special path mapping instance that preserves the equality relations
   * between the original config segments, but prepends a fixed string to distinguish hard-coded
   * path strings from mapped paths. This relies on actions using path mapping to be "root
   * agnostic": they must not contain logic that depends on any particular (output) root path.
   */
  static final PathMapper FOR_FINGERPRINTING =
      execPath -> {
        if (!execPath.startsWith(BAZEL_OUT) && !execPath.startsWith(BLAZE_OUT)) {
          // This is not an output path.
          return execPath;
        }
        String execPathString = execPath.getPathString();
        int startOfConfigSegment = execPathString.indexOf('/') + 1;
        return PathFragment.createAlreadyNormalized(
            execPathString.substring(0, startOfConfigSegment)
                + "pm-"
                + execPathString.substring(startOfConfigSegment));
      };

  private PathMapperConstants() {}
}
