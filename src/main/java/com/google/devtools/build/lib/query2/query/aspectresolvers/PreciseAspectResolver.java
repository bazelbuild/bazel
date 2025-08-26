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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSetMultimap;
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
import java.util.LinkedHashMap;
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
  public ImmutableMap<Aspect, ImmutableMultimap<Attribute, Label>> computeAspectDependencies(
      Target target, DependencyFilter dependencyFilter) throws InterruptedException {
    if (!(target instanceof Rule rule)) {
      return ImmutableMap.of();
    }
    if (!rule.hasAspects()) {
      return ImmutableMap.of();
    }

    LinkedHashMap<Aspect, ImmutableMultimap<Attribute, Label>> results = new LinkedHashMap<>();
    Multimap<Attribute, Label> transitions =
        rule.getTransitions(DependencyFilter.NO_NODEP_ATTRIBUTES);
    for (Attribute attribute : transitions.keySet()) {
      for (Aspect aspect : attribute.getAspects(rule)) {
        if (hasDepThatSatisfies(aspect, transitions.get(attribute))) {
          ImmutableSetMultimap.Builder<Attribute, Label> attributeLabelsBuilder =
              ImmutableSetMultimap.builder();
          AspectDefinition.forEachLabelDepFromAllAttributesOfAspect(
              aspect, dependencyFilter, attributeLabelsBuilder::put);
          ImmutableSetMultimap<Attribute, Label> attributeLabels = attributeLabelsBuilder.build();
          if (!attributeLabels.isEmpty()) {
            results.put(aspect, attributeLabels);
          }
        }
      }
    }
    return ImmutableMap.copyOf(results);
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

  private ImmutableCollection<Target> getSiblingTargets(Target buildFile)
      throws InterruptedException {
    Package pkg = buildFile.getPackage();
    if (pkg == null) {
      // Lazy macro expansion is enabled; try to expand the full package.
      try {
        pkg =
            packageProvider.getPackage(
                eventHandler, buildFile.getPackageMetadata().packageIdentifier());
      } catch (NoSuchPackageException e) {
        // If we fail to expand the full package (e.g. because a package piece for a symbolic macro
        // is in error), fall back to iterating only over the targets in the BUILD file's package
        // piece. The error encountered will be reported in the eventHandler.
        return buildFile.getPackageoid().getTargets().values();
      }
    }
    return pkg.getTargets().values();
  }

  @Override
  public Set<Label> computeBuildFileDependencies(Target buildFile) throws InterruptedException {
    Set<Label> result = new LinkedHashSet<>();
    buildFile.getPackageDeclarations().visitLoadGraph(result::add);

    Set<PackageIdentifier> dependentPackages = new LinkedHashSet<>();
    // First compute what packages can possibly affect the aspect attributes of this package:
    // Iterate over all rules...
    for (Target target : getSiblingTargets(buildFile)) {

      if (!(target instanceof Rule rule)) {
        continue;
      }

      // ...figure out which direct dependencies can possibly have aspects attached to them...
      Multimap<Attribute, Label> depsWithPossibleAspects =
          rule.getTransitions(
              (infoProvider, attribute) -> {
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
        dependentPackage.getDeclarations().visitLoadGraph(result::add);
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
