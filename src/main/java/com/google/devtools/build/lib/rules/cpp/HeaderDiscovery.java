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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
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
import java.util.List;
import java.util.Map;

/**
 * Manages the process of obtaining inputs used in a compilation from a dependency set parsed from
 * either .d files or /showIncludes output.
 */
public class HeaderDiscovery {

  /** Indicates if a compile should perform dotd pruning. */
  public static enum DotdPruningMode {
    USE,
    DO_NOT_USE
  }
  
  private final Action action;
  private final Artifact sourceFile;

  private final boolean shouldValidateInclusions;

  private final Collection<Path> dependencies;
  private final List<Path> permittedSystemIncludePrefixes;
  private final Map<PathFragment, Artifact> allowedDerivedInputsMap;
  
  /**
   * Creates a HeaderDiscover instance
   *
   * @param action the action instance requiring header discovery
   * @param sourceFile the source file for the compile
   * @param shouldValidateInclusions true if include validation should be performed
   */
  public HeaderDiscovery(
      Action action,
      Artifact sourceFile,
      boolean shouldValidateInclusions,
      Collection<Path> dependencies,
      List<Path> permittedSystemIncludePrefixes,
      Map<PathFragment, Artifact> allowedDerivedInputsMap) {
    this.action = Preconditions.checkNotNull(action);
    this.sourceFile = Preconditions.checkNotNull(sourceFile);
    this.shouldValidateInclusions = shouldValidateInclusions;
    this.dependencies = dependencies;
    this.permittedSystemIncludePrefixes = permittedSystemIncludePrefixes;
    this.allowedDerivedInputsMap = allowedDerivedInputsMap;
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
  @VisibleForTesting
  @ThreadCompatible
  public NestedSet<Artifact> discoverInputsFromDependencies(
      Path execRoot, ArtifactResolver artifactResolver) throws ActionExecutionException {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    if (dependencies == null) {
      return inputs.build();
    }
    List<Path> systemIncludePrefixes = permittedSystemIncludePrefixes;

    // Check inclusions.
    IncludeProblems problems = new IncludeProblems();
    for (Path execPath : dependencies) {
      PathFragment execPathFragment = execPath.asFragment();
      if (execPathFragment.isAbsolute()) {
        // Absolute includes from system paths are ignored.
        if (FileSystemUtils.startsWithAny(execPath, systemIncludePrefixes)) {
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
              String.format("Could not find the external repository for %s", execPathFragment),
              e, action, false);
        }
      }
      if (artifact != null) {
        inputs.add(artifact);
      } else {
        // Abort if we see files that we can't resolve, likely caused by
        // undeclared includes or illegal include constructs.
        problems.add(execPathFragment.getPathString());
      }
    }
    if (shouldValidateInclusions) {
      problems.assertProblemFree(action, sourceFile);
    }
    return inputs.build();
  }

  /** A Builder for HeaderDiscovery */
  public static class Builder {
    private Action action;
    private Artifact sourceFile;
    private boolean shouldValidateInclusions = false;

    private Collection<Path> dependencies;
    private List<Path> permittedSystemIncludePrefixes;
    private Map<PathFragment, Artifact> allowedDerivedInputsMap;

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
    public Builder setAllowedDerivedinputsMap(Map<PathFragment, Artifact> allowedDerivedInputsMap) {
      this.allowedDerivedInputsMap = allowedDerivedInputsMap;
      return this;
    }

    /** Creates a CppHeaderDiscovery instance. */
    public HeaderDiscovery build() {
      return new HeaderDiscovery(
          action,
          sourceFile,
          shouldValidateInclusions,
          dependencies,
          permittedSystemIncludePrefixes,
          allowedDerivedInputsMap);
    }
  }
}
