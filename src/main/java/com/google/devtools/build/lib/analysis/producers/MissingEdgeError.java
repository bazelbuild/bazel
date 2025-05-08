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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationIdMessage;
import static com.google.devtools.build.lib.cmdline.LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER;

import com.google.devtools.build.lib.analysis.AnalysisRootCauseEvent;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RepositoryFetchException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import javax.annotation.Nullable;

/**
 * A dependency error caused by a missing {@link Package} or {@link Target}.
 *
 * <p>This class is structured this way to relay details of the error all the way out to the top
 * level which has both the base {@link TargetAndConfiguration} and the {@link ExtendedEventHandler}
 * references needed to construct the causes and events.
 */
public final class MissingEdgeError {
  private final DependencyKind kind;
  private final Label label;
  private final NoSuchThingException cause;

  MissingEdgeError(DependencyKind kind, Label label, NoSuchThingException cause) {
    this.kind = kind;
    this.label = label;
    this.cause = cause;
  }

  /** Emits the causes and events associated with this error. */
  public void emitCausesAndEvents(
      TargetAndConfiguration fromNode,
      TransitiveDependencyState transitiveState,
      ExtendedEventHandler listener) {
    Target from = fromNode.getTarget();

    if (cause instanceof RepositoryFetchException) {
      Label repositoryLabel;
      try {
        repositoryLabel =
            Label.create(EXTERNAL_PACKAGE_IDENTIFIER, label.getRepository().getName());
      } catch (LabelSyntaxException lse) {
        // We're taking the repository name from something that was already part of a label, so it
        // should be valid. If we really get into this strange we situation, better not try to be
        // smart and report the original label.
        repositoryLabel = label;
      }
      transitiveState.addTransitiveCause(
          new LoadingFailedCause(repositoryLabel, cause.getDetailedExitCode()));
      listener.handle(
          Event.error(
              TargetUtils.getLocationMaybe(from),
              String.format(
                  "%s depends on %s in repository %s which failed to fetch. %s",
                  from.getLabel(), label, label.getRepository(), cause.getMessage())));
      return;
    }

    if (cause instanceof NoSuchPackageException) {
      // Blames the rule for specifying an unavailable package.
      @Nullable BuildConfigurationValue configuration = fromNode.getConfiguration();
      listener.post(
          AnalysisRootCauseEvent.withConfigurationValue(configuration, label, cause.getMessage()));
      transitiveState.addTransitiveCause(
          new AnalysisFailedCause(
              label, configurationIdMessage(configuration), cause.getDetailedExitCode()));
    } else if (cause instanceof NoSuchTargetException) {
      // If the child target was present, it already has an associated LoadingFailedCause.
      if (!((NoSuchTargetException) cause).hasTarget()) {
        transitiveState.addTransitiveCause(
            new LoadingFailedCause(label, cause.getDetailedExitCode()));
      }
    }

    String message;
    if (DependencyKind.isToolchain(kind)) {
      message =
          String.format(
              "Target '%s' depends on toolchain '%s', which cannot be found: %s'",
              from.getLabel(), label, cause.getMessage());
    } else {
      message = TargetUtils.formatMissingEdge(from, label, cause, kind.getAttribute());
    }
    listener.handle(Event.error(TargetUtils.getLocationMaybe(from), message));
  }

  @Override
  public String toString() {
    return toStringHelper(this)
        .add("kind", kind)
        .add("label", label)
        .add("cause", cause)
        .toString();
  }
}
