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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
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
  protected void missingEdgeHook(Target from, Label to, NoSuchThingException e)
      throws InterruptedException {
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
  protected Map<Label, Target> getTargets(
      Iterable<Label> labels,
      Target fromTarget,
      NestedSetBuilder<Cause> rootCauses,
      int labelsSizeHint)
      throws InterruptedException {
    Map<SkyKey, ValueOrException<NoSuchPackageException>> packages =
        env.getValuesOrThrow(
            Iterables.transform(labels, label -> PackageValue.key(label.getPackageIdentifier())),
            NoSuchPackageException.class);
    if (env.valuesMissing()) {
      return null;
    }
    if (labels instanceof Collection) {
      labelsSizeHint = ((Collection<Label>) labels).size();
    } else if (labelsSizeHint <= 0) {
      labelsSizeHint = 2 * packages.size();
    }
    // Duplicates can occur, so we can't use ImmutableMap.
    HashMap<Label, Target> result = Maps.newHashMapWithExpectedSize(labelsSizeHint);
    for (Label label : labels) {
      PackageValue packageValue;
      try {
        packageValue =
            Preconditions.checkNotNull(
                (PackageValue) packages.get(PackageValue.key(label.getPackageIdentifier())).get(),
                label);
      } catch (NoSuchPackageException e) {
        rootCauses.add(new LoadingFailedCause(label, e.getMessage()));
        missingEdgeHook(fromTarget, label, e);
        continue;
      }
      Package pkg = packageValue.getPackage();
      try {
        Target target = pkg.getTarget(label.getName());
        if (pkg.containsErrors()) {
          NoSuchTargetException e = new NoSuchTargetException(target);
          missingEdgeHook(fromTarget, label, e);
          rootCauses.add(new LoadingFailedCause(label, e.getMessage()));
        }
        result.put(label, target);
      } catch (NoSuchTargetException e) {
        rootCauses.add(new LoadingFailedCause(label, e.getMessage()));
        missingEdgeHook(fromTarget, label, e);
      }
    }
    return result;
  }
}
