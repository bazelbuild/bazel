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
package com.google.devtools.build.lib.query2.output;

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Command-line options for the Blaze query language, revision 2.
 */
public class QueryOptions extends OptionsBase {

  @Option(name = "output",
      defaultValue = "label",
      category = "query",
      help = "The format in which the query results should be printed."
          + " Allowed values are: label, label_kind, minrank, maxrank, package, location, graph,"
          + " xml, proto, record.")
  public String outputFormat;

  @Option(name = "order_results",
      defaultValue = "true",
      category = "query",
      help = "Output the results in dependency-ordered (default) or unordered fashion. The"
          + " unordered output is faster but only supported when --output is one of label,"
          + " label_kind, location, package, proto, record, xml.")
  public boolean orderResults;

  @Option(name = "keep_going",
      abbrev = 'k',
      defaultValue = "false",
      category = "strategy",
      help = "Continue as much as possible after an error.  While the "
          + "target that failed, and those that depend on it, cannot be "
          + "analyzed, other prerequisites of these "
          + "targets can be.")
  public boolean keepGoing;

  @Option(name = "loading_phase_threads",
      defaultValue = "200",
      category = "undocumented",
      help = "Number of parallel threads to use for the loading phase.")
  public int loadingPhaseThreads;

  @Option(name = "host_deps",
      defaultValue = "true",
      category = "query",
          help = "If enabled, dependencies on 'host configuration' targets will be included in "
          + "the dependency graph over which the query operates.  A 'host configuration' "
          + "dependency edge, such as the one from any 'proto_library' rule to the Protocol "
          + "Compiler, usually points to a tool executed during the build (on the host machine) "
          + "rather than a part of the same 'target' program.  Queries whose purpose is to "
          + "discover the set of things needed during a build will typically enable this option; "
          + "queries aimed at revealing the structure of a single program will typically  disable "
          + "this option.")
  public boolean includeHostDeps;

  @Option(name = "implicit_deps",
      defaultValue = "true",
      category = "query",
      help = "If enabled, implicit dependencies will be included in the dependency graph over "
          + "which the query operates. An implicit dependency is one that is not explicitly "
          + "specified in the BUILD file but added by blaze.")
  public boolean includeImplicitDeps;

  @Option(name = "graph:node_limit",
      defaultValue = "512",
      category = "query",
      help = "The maximum length of the label string for a graph node in the output.  Longer labels"
           + " will be truncated; -1 means no truncation.  This option is only applicable to"
           + " --output=graph.")
  public int graphNodeStringLimit;

  @Option(name = "graph:factored",
      defaultValue = "true",
      category = "query",
      help = "If true, then the graph will be emitted 'factored', i.e. "
          + "topologically-equivalent nodes will be merged together and their "
          + "labels concatenated.    This option is only applicable to "
          + "--output=graph.")
  public boolean graphFactored;

  @Option(name = "xml:line_numbers",
      defaultValue = "true",
      category = "query",
      help = "If true, XML output contains line numbers.  Disabling this option "
          + "may make diffs easier to read.  This option is only applicable to "
          + "--output=xml.")
  public boolean xmlLineNumbers;

  @Option(name = "xml:default_values",
      defaultValue = "false",
      category = "query",
      help = "If true, rule attributes whose value is not explicitly specified "
          + "in the BUILD file are printed; otherwise they are omitted.")
  public boolean xmlShowDefaultValues;

  @Option(name = "strict_test_suite",
      defaultValue = "false",
      category = "query",
      help = "If true, the tests() expression gives an error if it encounters a test_suite "
          + "containing non-test targets.")
  public boolean strictTestSuite;

  @Option(name = "universe_scope",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      category = "query",
      help = "A comma-separated set of target patterns (additive and subtractive). The query may "
          + "be performed in the universe defined by the transitive closure of the specified "
          + "targets.")
  public List<String> universeScope;

  @Option(name = "relative_locations",
      defaultValue = "false",
      category = "query",
      help = "If true, the location of BUILD files in xml and proto outputs will be relative. "
        + "By default, the location output is an absolute path and will not be consistent "
        + "across machines. You can set this option to true to have a consistent result "
        + "across machines.")
  public boolean relativeLocations;

  /**
   * Return the current options as a set of QueryEnvironment settings.
   */
  public Set<Setting> toSettings() {
    Set<Setting> settings = EnumSet.noneOf(Setting.class);
    if (strictTestSuite) {
      settings.add(Setting.TESTS_EXPRESSION_STRICT);
    }
    if (!includeHostDeps) {
      settings.add(Setting.NO_HOST_DEPS);
    }
    if (!includeImplicitDeps) {
      settings.add(Setting.NO_IMPLICIT_DEPS);
    }
    return settings;
  }
}
