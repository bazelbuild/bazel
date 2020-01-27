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
package com.google.devtools.build.lib.query2.query.aspectresolvers;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
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
import java.util.Set;

/**
 * An aspect resolver that returns only those aspects that are possibly active given the rule
 * classes of direct dependencies.
 *
 * <p>Needs to load the packages that contain dependencies through attributes with aspects.
 */
public class PreciseAspectResolver implements AspectResolver {
  private final PackageProvider packageProvider;
  private final ExtendedEventHandler eventHandler;

  public PreciseAspectResolver(PackageProvider packageProvider, ExtendedEventHandler eventHandler) {
    this.packageProvider = packageProvider;
    this.eventHandler = eventHandler;
  }

  @Override
  public ImmutableMultimap<Attribute, Label> computeAspectDependencies(
      Target target, DependencyFilter dependencyFilter) throws InterruptedException {
    if (!(target instanceof Rule)) {
      return ImmutableMultimap.of();
    }
    Rule rule = (Rule) target;
    if (!rule.hasAspects()) {
      return ImmutableMultimap.of();
    }
    Multimap<Attribute, Label> result = LinkedListMultimap.create();
    Multimap<Attribute, Label> transitions =
        rule.getTransitions(DependencyFilter.NO_NODEP_ATTRIBUTES);
    for (Attribute attribute : transitions.keySet()) {
      for (Aspect aspect : attribute.getAspects(rule)) {
        if (hasDepThatSatisfies(aspect, transitions.get(attribute))) {
          AspectDefinition.forEachLabelDepFromAllAttributesOfAspect(
              rule, aspect, dependencyFilter, result::put);
        }
      }
    }
    return ImmutableMultimap.copyOf(result);
  }

  private boolean hasDepThatSatisfies(Aspect aspect, Iterable<Label> labelDeps)
      throws InterruptedException {
    for (Label toLabel : labelDeps) {
      Target toTarget;
      try {
        toTarget = packageProvider.getTarget(eventHandler, toLabel);
      } catch (NoSuchThingException e) {
        // Do nothing interesting. One of target direct deps has an error. The dependency on the
        // BUILD file (or one of the files included in it) will be reported in the query result of
        // :BUILD.
        continue;
      }
      if (!(toTarget instanceof Rule)) {
        continue;
      }
      if (AspectDefinition.satisfies(
          aspect, ((Rule) toTarget).getRuleClassObject().getAdvertisedProviders())) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Set<Label> computeBuildFileDependencies(Package pkg) throws InterruptedException {
    Set<Label> result = new LinkedHashSet<>();
    result.addAll(pkg.getSkylarkFileDependencies());

    Set<PackageIdentifier> dependentPackages = new LinkedHashSet<>();
    // First compute with packages can possibly affect the aspect attributes of this package:
    // Iterate over all rules...
    for (Target target : pkg.getTargets().values()) {

      if (!(target instanceof Rule)) {
        continue;
      }

      // ...figure out which direct dependencies can possibly have aspects attached to them...
      Multimap<Attribute, Label> depsWithPossibleAspects =
          ((Rule) target)
              .getTransitions(
                  (Rule rule, Attribute attribute) -> {
                    for (Aspect aspectWithParameters : attribute.getAspects(rule)) {
                      if (!aspectWithParameters.getDefinition().getAttributes().isEmpty()) {
                        return true;
                      }
                    }

                    return false;
                  });

      // ...and add the package of the aspect.
      for (Label depLabel : depsWithPossibleAspects.values()) {
        dependentPackages.add(depLabel.getPackageIdentifier());
      }
    }

    // Then add all the labels of all the bzl files loaded by the packages found.
    for (PackageIdentifier packageIdentifier : dependentPackages) {
      try {
        result.add(Label.create(packageIdentifier, "BUILD"));
        Package dependentPackage = packageProvider.getPackage(eventHandler, packageIdentifier);
        result.addAll(dependentPackage.getSkylarkFileDependencies());
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
