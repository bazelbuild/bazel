// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Implements the loading phase; responsible for:
 * <ul>
 *   <li>target pattern evaluation
 *   <li>test suite expansion
 *   <li>loading the labels needed to construct the build configuration
 *   <li>loading the labels needed for the analysis with the build configuration
 *   <li>loading the transitive closure of the targets and the configuration labels
 * </ul>
 *
 * <p>In order to ensure correctness of incremental loading and of full cache hits, this class is
 * very restrictive about access to its internal state and to its collaborators. In particular, none
 * of the collaborators of this class may change in incompatible ways, such as changing the relative
 * working directory for the target pattern parser, without notifying this class.
 *
 * <p>For full caching, this class tracks the exact values of all inputs to the loading phase. To
 * maximize caching, it is vital that these change as rarely as possible.
 */
public abstract class LoadingPhaseRunner {
  /**
   * Performs target pattern evaluation, test suite expansion (if requested), and loads the
   * transitive closure of the resulting targets as well as of the targets needed to use the
   * given build configuration provider.
   */
  public abstract LoadingResult execute(EventHandler eventHandler, EventBus eventBus,
      List<String> targetPatterns, PathFragment relativeWorkingDirectory, LoadingOptions options,
      ListMultimap<String, Label> labelsToLoadUnconditionally, boolean keepGoing,
      boolean enableLoading, boolean determineTests, @Nullable LoadingCallback callback)
          throws TargetParsingException, LoadingFailedException, InterruptedException;

  /**
   * Returns a map of collected package names to root paths.
   */
  public static ImmutableMap<PackageIdentifier, Path> collectPackageRoots(
      Collection<Package> packages) {
    // Make a map of the package names to their root paths.
    ImmutableMap.Builder<PackageIdentifier, Path> packageRoots = ImmutableMap.builder();
    for (Package pkg : packages) {
      packageRoots.put(pkg.getPackageIdentifier(), pkg.getSourceRoot());
    }
    return packageRoots.build();
  }

  /**
   * Emit a warning when a deprecated target is mentioned on the command line.
   *
   * <p>Note that this does not stop us from emitting "target X depends on deprecated target Y"
   * style warnings for the same target and it is a good thing; <i>depending</i> on a target and
   * <i>wanting</i> to build it are different things.
   */
  // Public for use by skyframe.TargetPatternPhaseFunction until this class goes away.
  public static void maybeReportDeprecation(EventHandler eventHandler, Collection<Target> targets) {
    for (Rule rule : Iterables.filter(targets, Rule.class)) {
      if (rule.isAttributeValueExplicitlySpecified("deprecation")) {
        eventHandler.handle(Event.warn(rule.getLocation(), String.format(
            "target '%s' is deprecated: %s", rule.getLabel(),
            NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING))));
      }
    }
  }
}
