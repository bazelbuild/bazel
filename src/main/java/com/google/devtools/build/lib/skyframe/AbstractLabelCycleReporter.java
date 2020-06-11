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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageProvider;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.Callable;
import java.util.function.Predicate;

/** Reports cycles between skyframe values whose keys contains {@link Label}s. */
abstract class AbstractLabelCycleReporter implements CyclesReporter.SingleCycleReporter {

  private final PackageProvider packageProvider;

  AbstractLabelCycleReporter(PackageProvider packageProvider) {
    this.packageProvider = packageProvider;
  }

  /** Returns the associated Label of the SkyKey. */
  protected abstract Label getLabel(SkyKey key);

  protected abstract boolean canReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo);

  /** Returns the String representation of the {@code SkyKey}. */
  protected String prettyPrint(SkyKey key) {
    return getLabel(key).toString();
  }

  /** Can be used to skip individual keys on the path to the cycle. */
  protected boolean shouldSkipOnPathToCycle(SkyKey key) {
    return false;
  }

  /** Can be used to skip intermediate keys on the cycle itself. */
  protected boolean shouldSkipIntermediateKeyOnCycle(SkyKey key) {
    return false;
  }

  /**
   * Can be used to report an additional message about the cycle.
   *
   * @param eventHandler
   * @param topLevelKey
   * @param cycleInfo
   */
  protected String getAdditionalMessageAboutCycle(
      ExtendedEventHandler eventHandler, SkyKey topLevelKey, CycleInfo cycleInfo) {
    return "";
  }

  @Override
  public boolean maybeReportCycle(
      SkyKey topLevelKey,
      CycleInfo cycleInfo,
      boolean alreadyReported,
      ExtendedEventHandler eventHandler) {
    Preconditions.checkNotNull(eventHandler);
    if (!canReportCycle(topLevelKey, cycleInfo)) {
      return false;
    }

    if (alreadyReported) {
      if (!shouldSkipOnPathToCycle(topLevelKey)) {
        Label label = getLabel(topLevelKey);
        Target target = getTargetForLabel(eventHandler, label);
        eventHandler.handle(
            Event.error(
                target.getLocation(),
                "in "
                    + target.getTargetKind()
                    + " "
                    + label
                    + ": cycle in dependency graph: target depends on an already-reported cycle"));
      }
    } else {
      StringBuilder cycleMessage = new StringBuilder("cycle in dependency graph:");
      ImmutableList<SkyKey> pathToCycle = cycleInfo.getPathToCycle();
      ImmutableList<SkyKey> cycle = cycleInfo.getCycle();
      for (SkyKey value : pathToCycle) {
        if (shouldSkipOnPathToCycle(value)) {
          continue;
        }
        cycleMessage.append("\n    ");
        cycleMessage.append(prettyPrint(value));
      }

      SkyKey cycleValue =
          printCycle(
              cycle, cycleMessage, this::prettyPrint, this::shouldSkipIntermediateKeyOnCycle);

      cycleMessage.append(getAdditionalMessageAboutCycle(eventHandler, topLevelKey, cycleInfo));

      Label label = getLabel(cycleValue);
      Target target = getTargetForLabel(eventHandler, label);
      eventHandler.handle(Event.error(
          target.getLocation(),
          "in " + target.getTargetKind() + " " + label + ": " + cycleMessage));
    }

    return true;
  }

  /** Prints the SkyKey-s in cycle into cycleMessage using the print function. */
  static SkyKey printCycle(
      ImmutableList<SkyKey> cycle,
      StringBuilder cycleMessage,
      Function<SkyKey, String> printFunction) {
    return printCycle(cycle, cycleMessage, printFunction, Predicates.alwaysFalse());
  }

  private static SkyKey printCycle(
      ImmutableList<SkyKey> cycle,
      StringBuilder cycleMessage,
      Function<SkyKey, String> printFunction,
      Predicate<SkyKey> shouldSkipIntermediateKey) {
    Preconditions.checkArgument(!cycle.isEmpty());
    SkyKey cycleValue = null;
    int valuesPrinted = 0;
    for (SkyKey value : Iterables.concat(cycle, ImmutableList.of(cycle.get(0)))) {
      if (cycleValue == null) { // first item
        cycleValue = value;
        cycleMessage.append("\n.-> ");
      } else if (value == cycleValue) { // last item of the cycle
        if (valuesPrinted == 1) {
          cycleMessage.append(" [self-edge]");
          cycleMessage.append("\n`--");
          break;
        } else {
          cycleMessage.append("\n`-- ");
        }
      } else if (shouldSkipIntermediateKey.test(value)) {
        continue;
      } else {
        cycleMessage.append("\n|   ");
      }
      cycleMessage.append(printFunction.apply(value));
      valuesPrinted++;
    }

    return cycleValue;
  }

  protected final Target getTargetForLabel(
      final ExtendedEventHandler eventHandler, final Label label) {
    try {
      return Uninterruptibles.callUninterruptibly(new Callable<Target>() {
        @Override
        public Target call()
            throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
          return packageProvider.getTarget(eventHandler, label);
        }
      });
    } catch (NoSuchThingException e) {
      // This method is used for getting the target from a label in a circular dependency.
      // If we have a cycle that means that we need to have accessed the target (to get its
      // dependencies). So all the labels in a dependency cycle need to exist.
      throw new IllegalStateException(e);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }
}
