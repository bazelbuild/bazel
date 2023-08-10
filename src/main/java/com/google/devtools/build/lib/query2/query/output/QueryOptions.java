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
package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.TriState;
import java.util.Set;

/** Command-line options for the Blaze query language, revision 2. */
public class QueryOptions extends CommonQueryOptions {

  /** An enum converter for {@code OrderOutput} . Should be used internally only. */
  public static class OrderOutputConverter extends EnumConverter<OrderOutput> {
    public OrderOutputConverter() {
      super(OrderOutput.class, "Order output setting");
    }
  }

  @Option(
      name = "output",
      defaultValue = "label",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The format in which the query results should be printed. Allowed values for query are:"
              + " build, graph, streamed_jsonproto, label, label_kind, location, maxrank, minrank,"
              + " package, proto, xml.")
  public String outputFormat;

  @Option(
      name = "null",
      defaultValue = "null",
      expansion = {"--line_terminator_null=true"},
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "Whether each format is terminated with \\0 instead of newline.")
  public Void isNull;


  @Option(
    name = "order_results",
    defaultValue = "null",
    deprecationWarning = "Please use --order_output=auto or --order_output=no instead of this flag",
    expansion = {"--order_output=auto"},
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Output the results in dependency-ordered (default) or unordered fashion. The "
            + "unordered output is faster but only supported when --output is not minrank, "
            + "maxrank, or graph."
  )
  public Void orderResults;

  @Option(
    name = "noorder_results",
    defaultValue = "null",
    deprecationWarning = "Please use --order_output=no or --order_output=auto instead of this flag",
    expansion = {"--order_output=no"},
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Output the results in dependency-ordered (default) or unordered fashion. The "
            + "unordered output is faster but only supported when --output is not minrank, "
            + "maxrank, or graph."
  )
  public Void noOrderResults;

  /** Whether and how output should be ordered. */
  public enum OrderOutput {
    /** Make no effort to order output besides that required by output formatter. */
    NO,

    /** Output in dependency order when compatible with output formatter. */
    DEPS,

    /** Same as full unless formatter is proto, minrank, maxrank, or graph, then deps. */
    AUTO,

    /** Output in dependency order, breaking ties with alphabetical order when needed. */
    FULL
  }

  @Option(
      name = "order_output",
      converter = OrderOutputConverter.class,
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Output the results unordered (no), dependency-ordered (deps), or fully ordered (full)."
              + " The default is 'auto', meaning that results are output either dependency-ordered"
              + " or fully ordered, depending on the output formatter (dependency-ordered for"
              + " proto, minrank, maxrank, and graph, fully ordered for all others). When output"
              + " is fully ordered, nodes are printed in a fully deterministic (total) order."
              + " First, all nodes are sorted alphabetically. Then, each node in the list is used"
              + " as the start of a post-order depth-first search in which outgoing edges to"
              + " unvisited nodes are traversed in alphabetical order of the successor nodes."
              + " Finally, nodes are printed in the reverse of the order in which they were"
              + " visited.")
  public OrderOutput orderOutput;

  @Option(
      name = "incompatible_lexicographical_output",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If this option is set, sorts --order_output=auto output in lexicographical order.")
  public boolean lexicographicalOutput;

  @Option(
      name = "graph:conditional_edges_limit",
      defaultValue = "4",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The maximum number of condition labels to show. -1 means no truncation and 0 means no "
              + "annotation. This option is only applicable to --output=graph.")
  public int graphConditionalEdgesLimit;

  @Option(
    name = "xml:line_numbers",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, XML output contains line numbers. Disabling this option may make diffs easier "
            + "to read.  This option is only applicable to --output=xml."
  )
  public boolean xmlLineNumbers;

  @Option(
    name = "xml:default_values",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, rule attributes whose value is not explicitly specified in the BUILD file are "
            + "printed; otherwise they are omitted."
  )
  public boolean xmlShowDefaultValues;

  @Option(
    name = "strict_test_suite",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
    help =
        "If true, the tests() expression gives an error if it encounters a test_suite containing "
            + "non-test targets."
  )
  public boolean strictTestSuite;

  @Option(
      name = "experimental_graphless_query",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
      help =
          "If true, uses a Query implementation that does not make a copy of the graph. The new"
              + " implementation only supports --order_output=no, as well as only a subset of"
              + " output formatters.")
  public TriState useGraphlessQuery;

  /** Return the current options as a set of QueryEnvironment settings. */
  @Override
  public Set<Setting> toSettings() {
    Set<Setting> settings = super.toSettings();
    if (strictTestSuite) {
      settings.add(Setting.TESTS_EXPRESSION_STRICT);
    }
    return settings;
  }
}
