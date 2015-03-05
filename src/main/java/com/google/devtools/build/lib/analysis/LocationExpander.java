// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Expands $(location) tags inside target attributes.
 * You can specify something like this in the BUILD file:
 *
 * somerule(name='some name',
 *          someopt = [ '$(location //mypackage:myhelper)' ],
 *          ...)
 *
 * and location will be substituted with //mypackage:myhelper executable output.
 * Note that //mypackage:myhelper should have just one output.
 */
public class LocationExpander {
  private static final int MAX_PATHS_SHOWN = 5;
  private static final String LOCATION = "$(location";
  private final RuleContext ruleContext;
  private Map<Label, Collection<Artifact>> locationMap;
  private boolean allowDataAttributeEntriesInLabel = false;

  /**
   * Creates location expander helper bound to specific target and with default
   * location map.
   *
   * @param ruleContext BUILD rule
   */
  public LocationExpander(RuleContext ruleContext) {
    this(ruleContext, false);
  }

  public LocationExpander(RuleContext ruleContext,
      boolean allowDataAttributeEntriesInLabel) {
    this.ruleContext = ruleContext;
    this.allowDataAttributeEntriesInLabel = allowDataAttributeEntriesInLabel;
  }

  public Map<Label, Collection<Artifact>> getLocationMap() {
    if (locationMap == null) {
      locationMap = buildLocationMap(ruleContext, allowDataAttributeEntriesInLabel);
    }
    return locationMap;
  }

  /**
   * Expands attribute's location and locations tags based on the target and
   * location map.
   *
   * @param attrName  name of the attribute
   * @param attrValue initial value of the attribute
   * @return attribute value with expanded location tags or original value in
   *         case of errors
   */
  public String expand(String attrName, String attrValue) {
    int restart = 0;

    int attrLength = attrValue.length();
    StringBuilder result = new StringBuilder(attrValue.length());

    while (true) {
      // (1) find '$(location ' or '$(locations '
      String message = "$(location)";
      boolean multiple = false;
      int start = attrValue.indexOf(LOCATION, restart);
      int scannedLength = LOCATION.length();
      if (start == -1 || start + scannedLength == attrLength) {
        result.append(attrValue.substring(restart));
        break;
      }

      if (attrValue.charAt(start + scannedLength) == 's') {
        scannedLength++;
        if (start + scannedLength == attrLength) {
          result.append(attrValue.substring(restart));
          break;
        }
        message = "$(locations)";
        multiple = true;
      }

      if (attrValue.charAt(start + scannedLength) != ' ') {
        result.append(attrValue, restart, start + scannedLength);
        restart = start + scannedLength;
        continue;
      }
      scannedLength++;

      int end = attrValue.indexOf(')', start + scannedLength);
      if (end == -1) {
        ruleContext.attributeError(attrName, "unterminated " + message + " expression");
        return attrValue;
      }

      // (2) parse label
      String labelText = attrValue.substring(start + scannedLength, end);
      Label label;
      try {
        label = ruleContext.getLabel().getRelative(labelText);
      } catch (Label.SyntaxException e) {
        ruleContext.attributeError(attrName,
                              "invalid label in " + message + " expression: " + e.getMessage());
        return attrValue;
      }

      // (3) replace with singleton artifact, iff unique.
      Collection<Artifact> artifacts = getLocationMap().get(label);
      if (artifacts == null) {
        ruleContext.attributeError(attrName,
                              "label '" + label + "' in " + message + " expression is not a "
                              + "declared prerequisite of this rule");
        return attrValue;
      }
      List<String> paths = getPaths(artifacts);
      if (paths.isEmpty()) {
        ruleContext.attributeError(attrName,
                              "label '" + label + "' in " + message + " expression expands to no "
                              + "files");
        return attrValue;
      }

      result.append(attrValue, restart, start);
      if (multiple) {
        Collections.sort(paths);
        Joiner.on(' ').appendTo(result, paths);
      } else {
        if (paths.size() > 1) {
          ruleContext.attributeError(attrName,
              String.format(
                  "label '%s' in %s expression expands to more than one file, "
                      + "please use $(locations %s) instead.  Files (at most %d shown) are: %s",
                  label, message, label,
                  MAX_PATHS_SHOWN, Iterables.limit(paths, MAX_PATHS_SHOWN)));
          return attrValue;
        }
        result.append(Iterables.getOnlyElement(paths));
      }
      restart = end + 1;
    }
    return result.toString();
  }

  /**
   * Extracts all possible target locations from target specification.
   *
   * @param ruleContext BUILD target object
   * @return map of all possible target locations
   */
  private static Map<Label, Collection<Artifact>> buildLocationMap(RuleContext ruleContext,
      boolean allowDataAttributeEntriesInLabel) {
    Map<Label, Collection<Artifact>> locationMap = new HashMap<>();

    // Add all destination locations.
    for (OutputFile out : ruleContext.getRule().getOutputFiles()) {
      mapGet(locationMap, out.getLabel()).add(ruleContext.createOutputArtifact(out));
    }

    if (ruleContext.getRule().isAttrDefined("srcs", Type.LABEL_LIST)) {
      for (FileProvider src : ruleContext
          .getPrerequisites("srcs", Mode.TARGET, FileProvider.class)) {
        Iterables.addAll(mapGet(locationMap, src.getLabel()), src.getFilesToBuild());
      }
    }

    // Add all locations associated with dependencies and tools
    List<FilesToRunProvider> depsDataAndTools = new ArrayList<>();
    if (ruleContext.getRule().isAttrDefined("deps", Type.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisites("deps", Mode.DONT_CHECK, FilesToRunProvider.class));
    }
    if (allowDataAttributeEntriesInLabel
        && ruleContext.getRule().isAttrDefined("data", Type.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisites("data", Mode.DATA, FilesToRunProvider.class));
    }
    if (ruleContext.getRule().isAttrDefined("tools", Type.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisites("tools", Mode.HOST, FilesToRunProvider.class));
    }

    for (FilesToRunProvider dep : depsDataAndTools) {
      Label label = dep.getLabel();
      Artifact executableArtifact = dep.getExecutable();

      // If the label has an executable artifact add that to the multimaps.
      if (executableArtifact != null) {
        mapGet(locationMap, label).add(executableArtifact);
      } else {
        mapGet(locationMap, label).addAll(dep.getFilesToRun());
      }
    }
    return locationMap;
  }

  /**
   * Extracts list of all executables associated with given collection of label
   * artifacts.
   *
   * @param artifacts to get the paths of
   * @return all associated executable paths
   */
  private static List<String> getPaths(Collection<Artifact> artifacts) {
    List<String> paths = Lists.newArrayListWithCapacity(artifacts.size());
    for (Artifact artifact : artifacts) {
      PathFragment execPath = artifact.getExecPath();
      if (execPath != null) {  // omit middlemen etc
        paths.add(execPath.getPathString());
      }
    }
    return paths;
  }

  /**
   * Returns the value in the specified map corresponding to 'key', creating and
   * inserting an empty container if absent. We use Map not Multimap because
   * we need to distinguish the cases of "empty value" and "absent key".
   *
   * @return the value in the specified map corresponding to 'key'
   */
  private static <K, V> Collection<V> mapGet(Map<K, Collection<V>> map, K key) {
    Collection<V> values = map.get(key);
    if (values == null) {
      // We use sets not lists, because it's conceivable that the same label
      // could appear twice, in "srcs" and "deps".
      values = Sets.newHashSet();
      map.put(key, values);
    }
    return values;
  }
}
