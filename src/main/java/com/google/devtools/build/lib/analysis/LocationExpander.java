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

package com.google.devtools.build.lib.analysis;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;

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

  /**
   * List of options to tweak the LocationExpander.
   */
  public static enum Options {
    /** output the execPath instead of the relative path */
    EXEC_PATHS,
    /** Allow to take label from the data attribute */
    ALLOW_DATA,
  }

  private static final int MAX_PATHS_SHOWN = 5;
  private static final String LOCATION = "$(location";

  private final RuleContext ruleContext;
  private final ImmutableSet<Options> options;

  /**
   * This is a Map, not a Multimap, because we need to distinguish between the cases of "empty
   * value" and "absent key."
   */
  private Map<Label, Collection<Artifact>> locationMap;
  private ImmutableMap<Label, ImmutableCollection<Artifact>> labelMap;

  /**
   * Creates location expander helper bound to specific target and with default location map.
   *
   * @param ruleContext BUILD rule
   * @param labelMap A mapping of labels to build artifacts.
   * @param allowDataAttributeEntriesInLabel set to true if the <code>data</code> attribute should
   *        be used too.
   */
  public LocationExpander(
      RuleContext ruleContext, ImmutableMap<Label, ImmutableCollection<Artifact>> labelMap,
      boolean allowDataAttributeEntriesInLabel) {
    this.ruleContext = ruleContext;
    ImmutableSet.Builder<Options> builder = ImmutableSet.builder();
    builder.add(Options.EXEC_PATHS);
    if (allowDataAttributeEntriesInLabel) {
      builder.add(Options.ALLOW_DATA);
    }
    this.options = builder.build();
    this.labelMap = labelMap;
  }

  /**
   * Creates location expander helper bound to specific target.
   *
   * @param ruleContext the BUILD rule's context
   * @param options the list of options, see {@link Options}.
   */
  public LocationExpander(RuleContext ruleContext, ImmutableSet<Options> options) {
    this.ruleContext = ruleContext;
    this.options = options;
  }

  /**
   * Creates location expander helper bound to specific target.
   *
   * @param ruleContext the BUILD rule's context
   * @param options the list of options, see {@link Options}.
   */
  public LocationExpander(RuleContext ruleContext, Options... options) {
    this.ruleContext = ruleContext;
    this.options = ImmutableSet.copyOf(options);
  }

  private Map<Label, Collection<Artifact>> getLocationMap() {
    if (locationMap == null) {
      locationMap = buildLocationMap(ruleContext, labelMap, options.contains(Options.ALLOW_DATA));
    }
    return locationMap;
  }

  public String expand(String input) {
    return expand(input, new RuleErrorReporter());
  }

  /**
   * Expands attribute's location and locations tags based on the target and
   * location map.
   *
   * @param attrName  name of the attribute; only used for error reporting
   * @param attrValue initial value of the attribute
   * @return attribute value with expanded location tags or original value in
   *         case of errors
   */
  public String expandAttribute(String attrName, String attrValue) {
    return expand(attrValue, new AttributeErrorReporter(attrName));
  }

  private String expand(String value, ErrorReporter reporter) {
    int restart = 0;

    int attrLength = value.length();
    StringBuilder result = new StringBuilder(value.length());

    while (true) {
      // (1) find '$(location ' or '$(locations '
      String message = "$(location)";
      boolean multiple = false;
      int start = value.indexOf(LOCATION, restart);
      int scannedLength = LOCATION.length();
      if (start == -1 || start + scannedLength == attrLength) {
        result.append(value.substring(restart));
        break;
      }

      if (value.charAt(start + scannedLength) == 's') {
        scannedLength++;
        if (start + scannedLength == attrLength) {
          result.append(value.substring(restart));
          break;
        }
        message = "$(locations)";
        multiple = true;
      }

      if (value.charAt(start + scannedLength) != ' ') {
        result.append(value, restart, start + scannedLength);
        restart = start + scannedLength;
        continue;
      }
      scannedLength++;

      int end = value.indexOf(')', start + scannedLength);
      if (end == -1) {
        reporter.report(ruleContext, "unterminated " + message + " expression");
        return value;
      }

      message = String.format(" in %s expression", message);

      // (2) parse label
      String labelText = value.substring(start + scannedLength, end).trim();
      Label label = parseLabel(labelText, message, reporter);

      if (label == null) {
        // Error was already reported in parseLabel()
        return value;
      }

      // (3) expand label; stop this operation if there is an error
      try {
        Collection<String> paths = resolveLabel(label, message, multiple);
        result.append(value, restart, start);

        appendPaths(result, paths, multiple);
      } catch (IllegalStateException ise) {
        reporter.report(ruleContext, ise.getMessage());
        return value;
      }

      restart = end + 1;
    }

    return result.toString();
  }

  private Label parseLabel(String labelText, String message, ErrorReporter reporter) {
    try {
      return ruleContext.getLabel().getRelative(labelText);
    } catch (LabelSyntaxException e) {
      reporter.report(ruleContext, String.format("invalid label%s: %s", message, e.getMessage()));
      return null;
    }
  }

  /**
   * Returns all possible target location(s) of the given label
   * @param message Original message, for error reporting purposes only
   * @param hasMultipleTargets Describes whether the label has multiple target locations
   * @return The collection of all path strings
   */
  private Collection<String> resolveLabel(
      Label unresolved, String message, boolean hasMultipleTargets) throws IllegalStateException {
    // replace with singleton artifact, iff unique.
    Collection<Artifact> artifacts = getLocationMap().get(unresolved);

    if (artifacts == null) {
      throw new IllegalStateException(
          "label '" + unresolved + "'" + message + " is not a declared prerequisite of this rule");
    }

    Set<String> paths = getPaths(artifacts, options.contains(Options.EXEC_PATHS));

    if (paths.isEmpty()) {
      throw new IllegalStateException(
          "label '" + unresolved + "'" + message + " expression expands to no files");
    }

    if (!hasMultipleTargets && paths.size() > 1) {
      throw new IllegalStateException(
          String.format(
              "label '%s'%s expands to more than one file, "
                  + "please use $(locations %s) instead.  Files (at most %d shown) are: %s",
              unresolved,
              message,
              unresolved,
              MAX_PATHS_SHOWN,
              Iterables.limit(paths, MAX_PATHS_SHOWN)));
    }

    return paths;
  }

  private void appendPaths(StringBuilder result, Collection<String> paths, boolean multiple) {
    Stream<String> stream = paths.stream();
    if (!multiple) {
      stream = stream.limit(1);
    }

    String pathString = stream.map(LocationExpander::quotePath).collect(joining(" "));

    result.append(pathString);
  }

  private static String quotePath(String path) {
    // TODO(jcater): Handle more cases where escaping is needed.
    if (path.contains(" ")) {
      path = "'" + path + "'";
    }
    return path;
  }

  /**
   * Extracts all possible target locations from target specification.
   *
   * @param ruleContext BUILD target object
   * @param labelMap map of labels to build artifacts
   * @return map of all possible target locations
   */
  private static Map<Label, Collection<Artifact>> buildLocationMap(
      RuleContext ruleContext,
      Map<Label, ? extends Collection<Artifact>> labelMap,
      boolean allowDataAttributeEntriesInLabel) {
    Map<Label, Collection<Artifact>> locationMap = Maps.newHashMap();
    if (labelMap != null) {
      for (Map.Entry<Label, ? extends Collection<Artifact>> entry : labelMap.entrySet()) {
        mapGet(locationMap, entry.getKey()).addAll(entry.getValue());
      }
    }

    // Add all destination locations.
    for (OutputFile out : ruleContext.getRule().getOutputFiles()) {
      mapGet(locationMap, out.getLabel()).add(ruleContext.createOutputArtifact(out));
    }

    if (ruleContext.getRule().isAttrDefined("srcs", BuildType.LABEL_LIST)) {
      for (TransitiveInfoCollection src : ruleContext
          .getPrerequisitesIf("srcs", Mode.TARGET, FileProvider.class)) {
        Iterables.addAll(mapGet(locationMap, AliasProvider.getDependencyLabel(src)),
            src.getProvider(FileProvider.class).getFilesToBuild());
      }
    }

    // Add all locations associated with dependencies and tools
    List<TransitiveInfoCollection> depsDataAndTools = new ArrayList<>();
    if (ruleContext.getRule().isAttrDefined("deps", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("deps", Mode.DONT_CHECK, FilesToRunProvider.class));
    }
    if (allowDataAttributeEntriesInLabel
        && ruleContext.getRule().isAttrDefined("data", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("data", Mode.DATA, FilesToRunProvider.class));
    }
    if (ruleContext.getRule().isAttrDefined("tools", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("tools", Mode.HOST, FilesToRunProvider.class));
    }

    for (TransitiveInfoCollection dep : depsDataAndTools) {
      Label label = AliasProvider.getDependencyLabel(dep);
      FilesToRunProvider filesToRun = dep.getProvider(FilesToRunProvider.class);
      Artifact executableArtifact = filesToRun.getExecutable();

      // If the label has an executable artifact add that to the multimaps.
      if (executableArtifact != null) {
        mapGet(locationMap, label).add(executableArtifact);
      } else {
        Iterables.addAll(mapGet(locationMap, label), filesToRun.getFilesToRun());
      }
    }
    return locationMap;
  }

  /**
   * Extracts list of all executables associated with given collection of label
   * artifacts.
   *
   * @param artifacts to get the paths of
   * @param takeExecPath if false, the root relative path will be taken
   * @return all associated executable paths
   */
  private static Set<String> getPaths(Collection<Artifact> artifacts, boolean takeExecPath) {
    TreeSet<String> paths = Sets.newTreeSet();

    for (Artifact artifact : artifacts) {
      PathFragment execPath =
          takeExecPath ? artifact.getExecPath() : artifact.getRootRelativePath();
      if (execPath != null) {  // omit middlemen etc
        paths.add(execPath.getCallablePathString());
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

  private static interface ErrorReporter {
    void report(RuleContext ctx, String error);
  }

  private static final class AttributeErrorReporter implements ErrorReporter {
    private final String attrName;

    public AttributeErrorReporter(String attrName) {
      this.attrName = attrName;
    }

    @Override
    public void report(RuleContext ctx, String error) {
      ctx.attributeError(attrName, error);
    }
  }

  private static final class RuleErrorReporter implements ErrorReporter {
    @Override
    public void report(RuleContext ctx, String error) {
      ctx.ruleError(error);
    }
  }
}
