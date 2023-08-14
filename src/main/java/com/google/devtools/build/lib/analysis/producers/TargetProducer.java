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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.ValueOrExceptionSink;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * Looks up a {@link Target} from its {@link Package} using its {@link Label}.
 *
 * <p>This is used by the {@link TargetAndConfigurationProducer}. In contrast to the target lookup
 * in {@link ConfiguredTargetAndDataProducer}, this one does not assume that target presence has
 * already been verified.
 */
final class TargetProducer implements StateMachine, ValueOrExceptionSink<NoSuchPackageException> {
  interface ResultSink {
    void acceptTarget(Target target);

    void acceptTargetError(NoSuchPackageException error);

    void acceptTargetError(NoSuchTargetException error, Location location);
  }

  // -------------------- Input --------------------
  private final Label label;
  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  private Package pkg;

  TargetProducer(
      Label label,
      TransitiveDependencyState transitiveState,
      ResultSink sink,
      StateMachine runAfter) {
    this.label = label;
    this.transitiveState = transitiveState;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(
        label.getPackageIdentifier(),
        NoSuchPackageException.class,
        (ValueOrExceptionSink<NoSuchPackageException>) this);
    return this::unwrapTarget;
  }

  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable NoSuchPackageException error) {
    if (value != null) {
      this.pkg = ((PackageValue) value).getPackage();
      return;
    }

    sink.acceptTargetError(error);
  }

  private StateMachine unwrapTarget(Tasks tasks) {
    if (pkg == null) {
      return DONE; // An error occurred.
    }

    Target target;
    try {
      target = pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      transitiveState.addTransitiveCause(new LoadingFailedCause(label, e.getDetailedExitCode()));
      sink.acceptTargetError(e, pkg.getBuildFile().getLocation());
      return runAfter;
    }

    if (pkg.containsErrors()) {
      FailureDetail failureDetail = pkg.contextualizeFailureDetailForTarget(target);
      // The target can be loaded but may have associated errors, for example, a missing required
      // attribute. In these cases, instead of failing fast, it's possible to perform dependency
      // resolution using the target-in-error to uncover any other errors that could be present in
      // its dependencies. This error is turned into an exception when the transitive causes are
      // examined after dependency resolution.
      transitiveState.addTransitiveCause(
          new LoadingFailedCause(label, DetailedExitCode.of(failureDetail)));
    }
    transitiveState.updateTransitivePackages(pkg);
    sink.acceptTarget(target);
    return runAfter;
  }
}
