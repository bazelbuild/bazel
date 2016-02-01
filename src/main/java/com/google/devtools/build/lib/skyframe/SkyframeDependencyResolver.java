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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;

import javax.annotation.Nullable;

/**
 * A dependency resolver for use within Skyframe. Loads packages lazily when possible.
 */
public final class SkyframeDependencyResolver extends DependencyResolver {

  private final Environment env;

  public SkyframeDependencyResolver(Environment env) {
    this.env = env;
  }

  @Override
  protected void invalidVisibilityReferenceHook(TargetAndConfiguration value, Label label) {
    env.getListener().handle(
        Event.error(TargetUtils.getLocationMaybe(value.getTarget()), String.format(
            "Label '%s' in visibility attribute does not refer to a package group", label)));
  }

  @Override
  protected void invalidPackageGroupReferenceHook(TargetAndConfiguration value, Label label) {
    env.getListener().handle(
        Event.error(TargetUtils.getLocationMaybe(value.getTarget()), String.format(
            "label '%s' does not refer to a package group", label)));
  }

  @Override
  protected void missingEdgeHook(Target from, Label to, NoSuchThingException e) {
    if (e instanceof NoSuchTargetException) {
      NoSuchTargetException nste = (NoSuchTargetException) e;
      if (to.equals(nste.getLabel())) {
        env.getListener().handle(
            Event.error(
                TargetUtils.getLocationMaybe(from),
                TargetUtils.formatMissingEdge(from, to, e)));
      }
    } else if (e instanceof NoSuchPackageException) {
      NoSuchPackageException nspe = (NoSuchPackageException) e;
      if (nspe.getPackageId().equals(to.getPackageIdentifier())) {
        env.getListener().handle(
            Event.error(
                TargetUtils.getLocationMaybe(from),
                TargetUtils.formatMissingEdge(from, to, e)));
      }
    }
  }

  @Nullable
  @Override
  protected Target getTarget(Target from, Label label, NestedSetBuilder<Label> rootCauses) {
    SkyKey key = PackageValue.key(label.getPackageIdentifier());
    PackageValue packageValue;
    try {
      packageValue = (PackageValue) env.getValueOrThrow(key, NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      rootCauses.add(label);
      missingEdgeHook(from, label, e);
      return null;
    }
    if (packageValue == null) {
      return null;
    }
    Package pkg = packageValue.getPackage();
    try {
      Target target = pkg.getTarget(label.getName());
      if (pkg.containsErrors()) {
        NoSuchPackageException e =
            new BuildFileContainsErrorsException(label.getPackageIdentifier());
        missingEdgeHook(from, label, e);
        if (target != null) {
          rootCauses.add(label);
          return target;
        } else {
          return null;
        }
      }
      return target;
    } catch (NoSuchTargetException e) {
      rootCauses.add(label);
      missingEdgeHook(from, label, e);
      return null;
    }
  }
}
