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
package com.google.devtools.build.lib.query2.output;

import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;

import java.util.Collection;
import java.util.Set;

/** Utility class that determines additional dependencies of a target from its aspects. */
public interface AspectResolver {

  /**
   * How to resolve aspect dependencies in 'blaze query'.
   */
  public enum Mode {
    // Do not report aspect dependencies
    OFF {
      @Override
      public AspectResolver createResolver(PackageProvider provider, EventHandler eventHandler) {
        return new NullAspectResolver();
      }
    },

    // Do not load dependent packages; report deps assuming all aspects defined on a rule are
    // triggered
    CONSERVATIVE {
      @Override
      public AspectResolver createResolver(PackageProvider provider, EventHandler eventHandler) {
        return new ConservativeAspectResolver();
      }
    },

    // Load direct dependencies and report aspects that can be triggered based on their types.
    PRECISE {
      @Override
      public AspectResolver createResolver(PackageProvider provider, EventHandler eventHandler) {
        return new PreciseAspectResolver(provider, eventHandler);
      }
    };

    public abstract AspectResolver createResolver(
        PackageProvider provider, EventHandler eventHandler);
  }

  /** The way aspect dependencies for a BUILD file are calculated. */
  enum BuildFileDependencyMode  {

    /** Return all the subincluded files that may affect the package. */
    SUBINCLUDE {
      @Override
      protected Collection<Label> getDependencies(Package pkg) {
        return pkg.getSubincludeLabels();
      }
    },

    /** Return all Skylark files that may affect the package. */
    SKYLARK {
      @Override
      protected Collection<Label> getDependencies(Package pkg) {
        return pkg.getSkylarkFileDependencies();
      }
    };

    protected abstract Collection<Label> getDependencies(Package pkg);
  }

  /**
   * Compute additional dependencies of target from aspects. This method may load the direct deps
   * of target to determine their types. Returns map of attributes and corresponding label values.
   */
  ImmutableMultimap<Attribute, Label> computeAspectDependencies(Target target,
      DependencyFilter dependencyFilter)
      throws InterruptedException;

  /**
   * Compute the labels of the BUILD and subinclude and Skylark files on which the results of the
   * other two methods depend for a target in the given package.
   */
  Set<Label> computeBuildFileDependencies(Package pkg, BuildFileDependencyMode mode)
      throws InterruptedException;
}
