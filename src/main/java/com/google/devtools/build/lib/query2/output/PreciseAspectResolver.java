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
package com.google.devtools.build.lib.query2.output;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;

import java.util.LinkedHashSet;
import java.util.Map.Entry;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * An aspect resolver that returns only those aspects that are possibly active given the rule
 * classes of direct dependencies.
 *
 * <p>Needs to load the packages that contain dependencies through attributes with aspects.
 */
public class PreciseAspectResolver implements AspectResolver {
  private final PackageProvider packageProvider;
  private final EventHandler eventHandler;

  public PreciseAspectResolver(PackageProvider packageProvider, EventHandler eventHandler) {
    this.packageProvider = packageProvider;
    this.eventHandler = eventHandler;
  }

  @Override
  public ImmutableMultimap<Attribute, Label> computeAspectDependencies(Target target)
      throws InterruptedException {
    Multimap<Attribute, Label> result = LinkedListMultimap.create();
    if (target instanceof Rule) {
      Multimap<Attribute, Label> transitions =
          ((Rule) target).getTransitions(DependencyFilter.NO_NODEP_ATTRIBUTES);
      for (Entry<Attribute, Label> entry : transitions.entries()) {
        Target toTarget;
        try {
          toTarget = packageProvider.getTarget(eventHandler, entry.getValue());
          result.putAll(AspectDefinition.visitAspectsIfRequired(target, entry.getKey(), toTarget));
        } catch (NoSuchThingException e) {
          // Do nothing. One of target direct deps has an error. The dependency on the BUILD file
          // (or one of the files included in it) will be reported in the query result of :BUILD.
        }
      }
    }
    return ImmutableMultimap.copyOf(result);
  }

  @Override
  public Set<Label> computeBuildFileDependencies(Package pkg, BuildFileDependencyMode mode)
      throws InterruptedException {
    Set<Label> result = new LinkedHashSet<>();
    result.addAll(mode.getDependencies(pkg));

    Set<PackageIdentifier> dependentPackages = new LinkedHashSet<>();
    // First compute with packages can possibly affect the aspect attributes of this package:
    // Iterate over all rules...
    for (Target target : pkg.getTargets()) {

      if (!(target instanceof Rule)) {
        continue;
      }

      // ...figure out which direct dependencies can possibly have aspects attached to them...
      Multimap<Attribute, Label> depsWithPossibleAspects =
          ((Rule) target)
              .getTransitions(
                  new DependencyFilter() {
                    @Override
                    public boolean apply(@Nullable Rule rule, Attribute attribute) {
                      for (Aspect aspectWithParameters : attribute.getAspects(rule)) {
                        if (!aspectWithParameters.getDefinition().getAttributes().isEmpty()) {
                          return true;
                        }
                      }

                      return false;
                    }
                  });

      // ...and add the package of the aspect.
      for (Label depLabel : depsWithPossibleAspects.values()) {
        dependentPackages.add(depLabel.getPackageIdentifier());
      }
    }

    // Then add all the subinclude labels of the packages thus found to the result.
    for (PackageIdentifier packageIdentifier : dependentPackages) {
      try {
        result.add(Label.create(packageIdentifier, "BUILD"));
        Package dependentPackage = packageProvider.getPackage(eventHandler, packageIdentifier);
        result.addAll(mode.getDependencies(dependentPackage));
      } catch (NoSuchPackageException e) {
        // If the package is not found, just add its BUILD file, which is already done above.
        // Hopefully this error is not raised when there is a syntax error in a subincluded file
        // or something.
      } catch (LabelSyntaxException e) {
        throw new IllegalStateException(e);
      }
    }

    return result;
  }
}
