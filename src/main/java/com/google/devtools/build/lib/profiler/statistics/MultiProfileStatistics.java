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
package com.google.devtools.build.lib.profiler.statistics;

import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.InfoListener;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Builds and aggregates statistics for multiple profile files.
 */
public final class MultiProfileStatistics implements Iterable<Path> {
  private final PhaseSummaryStatistics summaryStatistics;
  private final EnumMap<ProfilePhase, PhaseStatistics> summaryPhaseStatistics;
  private final Map<Path, EnumMap<ProfilePhase, PhaseStatistics>> filePhaseStatistics;
  private final SkylarkStatistics skylarkStatistics;

  private int missingActionsCount;
  private boolean generateVfsStatistics;

  public MultiProfileStatistics(
      Path workingDirectory,
      String workSpaceName,
      List<String> files,
      InfoListener listener,
      boolean generateVfsStatistics) {
    summaryStatistics = new PhaseSummaryStatistics();
    summaryPhaseStatistics = new EnumMap<>(ProfilePhase.class);
    filePhaseStatistics = new HashMap<>();
    skylarkStatistics = new SkylarkStatistics();
    this.generateVfsStatistics = generateVfsStatistics;
    for (String file : files) {
      loadProfileFile(workingDirectory, workSpaceName, file, listener);
    }
  }

  public PhaseSummaryStatistics getSummaryStatistics() {
    return summaryStatistics;
  }

  public EnumMap<ProfilePhase, PhaseStatistics> getSummaryPhaseStatistics() {
    return summaryPhaseStatistics;
  }

  public PhaseStatistics getSummaryPhaseStatistics(ProfilePhase phase) {
    return summaryPhaseStatistics.get(phase);
  }

  public SkylarkStatistics getSkylarkStatistics() {
    return skylarkStatistics;
  }

  public int getMissingActionsCount() {
    return missingActionsCount;
  }

  public EnumMap<ProfilePhase, PhaseStatistics> getPhaseStatistics(Path file) {
    return filePhaseStatistics.get(file);
  }

  @Override
  public Iterator<Path> iterator() {
    return filePhaseStatistics.keySet().iterator();
  }

  /**
   * Loads a single profile file and adds the statistics to the previously collected ones.
   */
  private void loadProfileFile(
      Path workingDirectory, String workSpaceName, String file, InfoListener listener) {
    ProfileInfo info;
    Path profileFile = workingDirectory.getRelative(file);
    try {
      info = ProfileInfo.loadProfileVerbosely(profileFile, listener);
      ProfileInfo.aggregateProfile(info, listener);
    } catch (IOException e) {
      listener.warn("Ignoring file " + file + " - cannot load: " + e.getMessage());
      return;
    }

    summaryStatistics.addProfileInfo(info);

    EnumMap<ProfilePhase, PhaseStatistics> fileStatistics = new EnumMap<>(ProfilePhase.class);
    filePhaseStatistics.put(profileFile, fileStatistics);

    for (ProfilePhase phase : ProfilePhase.values()) {
      PhaseStatistics filePhaseStat =
          new PhaseStatistics(phase, info, workSpaceName, generateVfsStatistics);
      fileStatistics.put(phase, filePhaseStat);

      PhaseStatistics summaryPhaseStats;
      if (summaryPhaseStatistics.containsKey(phase)) {
        summaryPhaseStats = summaryPhaseStatistics.get(phase);
      } else {
        summaryPhaseStats = new PhaseStatistics(phase, generateVfsStatistics);
        summaryPhaseStatistics.put(phase, summaryPhaseStats);
      }
      summaryPhaseStats.add(filePhaseStat);
    }

    skylarkStatistics.addProfileInfo(info);

    missingActionsCount += info.getMissingActionsCount();
  }
}
