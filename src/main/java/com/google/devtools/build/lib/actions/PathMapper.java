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

import com.google.devtools.build.lib.actions.CommandLine.ArgChunk;
import com.google.devtools.build.lib.actions.CommandLineItem.ExceptionlessMapFn;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.CheckReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Support for mapping config parts of exec paths in an action's command line as well as when
 * staging its inputs and outputs for execution, with the aim of making the resulting {@link Spawn}
 * more cacheable.
 *
 * <p>Implementations must be pure functions of the set of inputs (including paths and potentially
 * content) of a given action.
 *
 * <p>Actions that want to support path mapping should use {@link
 * com.google.devtools.build.lib.analysis.actions.PathMappers}.
 *
 * <p>An example of an implementing class is {@link
 * com.google.devtools.build.lib.analysis.actions.StrippingPathMapper}, which removes the config
 * part (e.g. "k8-fastbuild") from exec paths to allow for cross-configuration cache hits.
 */
public abstract class PathMapper {
  /**
   * Returns the exec path with the path mapping applied.
   *
   * <p>An exec path typically consists of a root part (such as <code>"bazel-out/k8-opt/bin"</code>
   * or <code>""</code>) and a root-relative part (such as <code>"path/to/pkg/file"</code>).
   *
   * <p>Overrides must satisfy the following properties:
   *
   * <ul>
   *   <li>The root-relative part of the path must not be modified.
   *   <li>If the path is modified, the new root part must be different from any possible valid root
   *       part of an unmapped path.
   * </ul>
   *
   * <p>Path mappers may return paths with different roots for two paths that have the same root
   * (e.g., they may map an artifact at {@code bazel-out/k8-fastbuild/bin/pkg/foo} to {@code
   * bazel-out/<hash of the file>/bin/pkg/foo}). Paths of artifacts that should share the same
   * parent directory, such as runfiles or tree artifact files, should thus be derived from the
   * mapped path of their parent.
   */
  public abstract PathFragment map(PathFragment execPath);

  /** Returns the exec path of the input with the path mapping applied. */
  public String getMappedExecPathString(ActionInput artifact) {
    return map(artifact.getExecPath()).getPathString();
  }

  /** A {@link PathMapper} that doesn't change paths. */
  public static final PathMapper NOOP =
      new PathMapper() {
        @Override
        public PathFragment map(PathFragment execPath) {
          return execPath;
        }
      };

  private static final StarlarkSemantics.Key<PathMapper> SEMANTICS_KEY =
      new StarlarkSemantics.Key<>("path_mapper", PathMapper.NOOP);

  /**
   * Retrieve the {@link PathMapper} instance stored in the given {@link StarlarkSemantics} via
   * {@link #storeIn(StarlarkSemantics)}.
   */
  public static PathMapper loadFrom(StarlarkSemantics semantics) {
    return semantics.get(SEMANTICS_KEY);
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
  public StarlarkSemantics storeIn(StarlarkSemantics semantics) {
    // Since PathMapper#equals returns true for all instances of the same class, every non-noop
    // instance is different from the default value for the key and thus persisted. Furthermore,
    // since this causes the resulting semantics to compare equal if they PathMapper class
    // agrees, there is only a single cache entry for it in Starlark's CallUtils cache for Starlark
    // methods.
    return semantics.toBuilder().set(SEMANTICS_KEY, this).build();
  }

  /**
   * We don't yet have a Starlark API for mapping paths in command lines. Simple Starlark calls like
   * {@code args.add(arg_name, file_path} are automatically handled. But calls that involve custom
   * Starlark code require deeper API support that remains a TODO.
   *
   * <p>This method allows implementations to hard-code support for specific command line entries
   * for specific Starlark actions.
   */
  @CanIgnoreReturnValue
  public ArgChunk mapCustomStarlarkArgs(ArgChunk chunk) {
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
  public ExceptionlessMapFn<Object> getMapFn(@Nullable String previousFlag) {
    return MapFn.DEFAULT;
  }

  /**
   * Returns {@code true} if the mapper is known to map all paths identically.
   *
   * <p>Can be used by actions to skip additional work that isn't needed if path mapping is not
   * enabled.
   */
  public boolean isNoop() {
    return this == NOOP;
  }

  /**
   * Returns an opaque object whose equality class should encode all information that goes into the
   * behavior of the {@link #map(PathFragment)} function of this path mapper. This is used as a key
   * for in-memory caches.
   *
   * <p>The default implementation returns the {@link Class} of the path mapper.
   */
  public Object cacheKey() {
    return this.getClass();
  }

  @Override
  public final boolean equals(Object obj) {
    if (!(obj instanceof PathMapper)) {
      return false;
    }
    // We store PathMapper instances in StarlarkSemantics to allow mapping of File struct fields
    // such as "path" in map_each callbacks of Args. Since this code is only executed during the
    // execution phase, we do not want two different non-noop PathMapper instances to result in
    // distinct StarlarkSemantics instances for caching purposes.
    return isNoop() == ((PathMapper) obj).isNoop();
  }

  @Override
  public final int hashCode() {
    return Boolean.hashCode(isNoop());
  }
}
