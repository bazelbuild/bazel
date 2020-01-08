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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * HeaderDiscovery checks whether all header files that a compile action uses are actually declared
 * as inputs.
 *
 * <p>Tree artifacts: a tree artifact with path P causes any header file prefixed by P to be
 * accepted. Testing whether a used header file is prefixed by any tree artifact is linear search,
 * but the result is cached. If all files in a tree artifact are at the root of the artifact, the
 * entire check is performed by hash lookups.
 */
public class HeaderDiscovery {

  /** Indicates if a compile should perform dotd pruning. */
  public enum DotdPruningMode {
    USE,
    DO_NOT_USE
  }

  private final Action action;
  private final Artifact sourceFile;

  private final boolean shouldValidateInclusions;

  private final Collection<Path> dependencies;
  private final List<Path> permittedSystemIncludePrefixes;

  /**
   * allowedDerivedInputsMap maps paths of derived artifacts to the artifacts. These only include
   * file artifacts.
   */
  private final ImmutableMap<PathFragment, Artifact> allowedDerivedInputsMap;

  /**
   * {@link #treeArtifacts} contains tree artifacts given as input to the action.
   *
   * <p>Header files whose prefix is in this set are considered as included, and will not trigger a
   * header inclusion error.
   */
  private final ImmutableSet<Artifact> treeArtifacts;

  /**
   * {@link #pathToOwningTreeArtifact} caches answers to "does a fragment have a prefix in {@link
   * #treeArtifacts}".
   *
   * <p>It is initialized to (a.path, a) for each a in {@link #treeArtifacts}, in order to speed up
   * the typical case of headers coming from a flat tree artifact.
   */
  private final HashMap<PathFragment, Artifact> pathToOwningTreeArtifact;

  /**
   * Creates a HeaderDiscover instance
   *
   * @param action the action instance requiring header discovery
   * @param sourceFile the source file for the compile
   * @param shouldValidateInclusions true if include validation should be performed
   * @param allowedDerivedInputsMap see javadoc for {@link #allowedDerivedInputsMap}
   * @param treeArtifacts see javadoc for {@link #treeArtifacts}
   */
  private HeaderDiscovery(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      Collection<Path> dependencies,
      List<Path> permittedSystemIncludePrefixes,
      ImmutableMap<PathFragment, Artifact> allowedDerivedInputsMap,
      ImmutableSet<Artifact> treeArtifacts) {
    this.action = Preconditions.checkNotNull(action);
    this.sourceFile = Preconditions.checkNotNull(sourceFile);
    this.shouldValidateInclusions = shouldValidateInclusions;
    this.dependencies = dependencies;
    this.permittedSystemIncludePrefixes = permittedSystemIncludePrefixes;
    this.allowedDerivedInputsMap = allowedDerivedInputsMap;
    this.treeArtifacts = treeArtifacts;

    this.pathToOwningTreeArtifact = new HashMap<>();
    for (Artifact a : treeArtifacts) {
      pathToOwningTreeArtifact.put(a.getExecPath(), a);
    }
  }

  /**
   * Returns a collection with additional input artifacts relevant to the action by reading the
   * dynamically-discovered dependency information from the parsed dependency set after the action
   * has run.
   *
   * <p>Artifacts are considered inputs but not "mandatory" inputs.
   *
   * @throws ActionExecutionException iff the .d is missing (when required), malformed, or has
   *     unresolvable included artifacts.
   */
  @ThreadCompatible
  NestedSet<Artifact> discoverInputsFromDependencies(
      Path execRoot, ArtifactResolver artifactResolver) throws ActionExecutionException {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    if (dependencies == null) {
      return inputs.build();
    }

    // Check inclusions.
    IncludeProblems problems = new IncludeProblems();
    for (Path execPath : dependencies) {
      PathFragment execPathFragment = execPath.asFragment();
      if (execPathFragment.isAbsolute()) {
        // Absolute includes from system paths are ignored.
        if (FileSystemUtils.startsWithAny(execPath, permittedSystemIncludePrefixes)) {
          continue;
        }
        // Since gcc is given only relative paths on the command line,
        // non-system include paths here should never be absolute. If they
        // are, it's probably due to a non-hermetic #include, & we should stop
        // the build with an error.
        if (execPath.startsWith(execRoot)) {
          execPathFragment = execPath.relativeTo(execRoot); // funky but tolerable path
        } else {
          problems.add(execPathFragment.getPathString());
          continue;
        }
      }
      Artifact artifact = allowedDerivedInputsMap.get(execPathFragment);
      if (artifact == null) {
        try {
          RepositoryName repository =
              PackageIdentifier.discoverFromExecPath(execPathFragment, false).getRepository();
          artifact = artifactResolver.resolveSourceArtifact(execPathFragment, repository);
        } catch (LabelSyntaxException e) {
          throw new ActionExecutionException(
              String.format(
                  "Could not find the external repository for %s: " + e.getMessage(),
                  execPathFragment),
              e,
              action,
              false);
        }
      }
      if (artifact != null) {
        // We don't need to add the sourceFile itself as it is a mandatory input.
        if (!artifact.equals(sourceFile)) {
          inputs.add(artifact);
        }
        continue;
      }

      Artifact treeArtifact = findOwningTreeArtifact(execPathFragment);
      if (treeArtifact != null) {
        inputs.add(treeArtifact);
        continue;
      }

      // Abort if we see files that we can't resolve, likely caused by
      // undeclared includes or illegal include constructs.
      problems.add(execPathFragment.getPathString());
    }
    if (shouldValidateInclusions) {
      problems.assertProblemFree(action, sourceFile);
    }
    return inputs.build();
  }

  private Artifact findOwningTreeArtifact(PathFragment execPathFragment) {
    PathFragment dir = execPathFragment.getParentDirectory();
    Artifact artifact = pathToOwningTreeArtifact.get(dir);
    if (artifact != null) {
      return artifact;
    }
    return treeArtifacts.stream()
        .filter(a -> dir.startsWith(a.getExecPath()))
        .findAny()
        .orElse(null);
  }

  /** A Builder for HeaderDiscovery */
  public static class Builder {
    private Action action;
    private Artifact sourceFile;
    private boolean shouldValidateInclusions = false;

    private Collection<Path> dependencies;
    private List<Path> permittedSystemIncludePrefixes;
    private NestedSet<Artifact> allowedDerivedInputs;

    /** Sets the action for which to discover inputs. */
    public Builder setAction(Action action) {
      this.action = action;
      return this;
    }

    /** Sets the source file for which to discover inputs. */
    public Builder setSourceFile(Artifact sourceFile) {
      this.sourceFile = sourceFile;
      return this;
    }

    /** Sets that this compile should validate inclusions against the dotd file. */
    public Builder shouldValidateInclusions() {
      this.shouldValidateInclusions = true;
      return this;
    }

    /** Sets the dependencies capturing used headers by this compile. */
    public Builder setDependencies(Collection<Path> dependencies) {
      this.dependencies = dependencies;
      return this;
    }

    /** Sets prefixes of allowed absolute inclusions */
    public Builder setPermittedSystemIncludePrefixes(List<Path> systemIncludePrefixes) {
      this.permittedSystemIncludePrefixes = systemIncludePrefixes;
      return this;
    }

    /** Sets permitted inputs to the build */
    public Builder setAllowedDerivedInputs(NestedSet<Artifact> allowedDerivedInputs) {
      this.allowedDerivedInputs = allowedDerivedInputs;
      return this;
    }

    /** Creates a CppHeaderDiscovery instance. */
    public HeaderDiscovery build() {
      Map<PathFragment, Artifact> allowedDerivedInputsMap = new HashMap<>();
      ImmutableSet.Builder<Artifact> treeArtifacts = ImmutableSet.builder();
      for (Artifact a : allowedDerivedInputs.toList()) {
        if (a.isSourceArtifact()) {
          continue;
        }
        if (a.isTreeArtifact()) {
          treeArtifacts.add(a);
        }
        // We may encounter duplicate keys in the derived inputs if two artifacts have different
        // owners. Just use the first one. The two artifacts must be generated by equivalent
        // (shareable) actions in order to have not generated a conflict in Bazel. If on an
        // incremental build one changes without the other one changing, then if their paths remain
        // the same, that will trigger an action conflict and fail the build. If one path changes,
        // then this action will be re-analyzed, and will execute in Skyframe. It can legitimately
        // get an action cache hit in that case, since even if it previously depended on the
        // artifact whose path changed, that is not taken into account by the action cache, and it
        // will get an action cache hit using the remaining un-renamed artifact.
        allowedDerivedInputsMap.putIfAbsent(a.getExecPath(), a);
      }

      return new HeaderDiscovery(
          action,
          sourceFile,
          shouldValidateInclusions,
          dependencies,
          permittedSystemIncludePrefixes,
          ImmutableMap.copyOf(allowedDerivedInputsMap),
          treeArtifacts.build());
    }
  }
}
