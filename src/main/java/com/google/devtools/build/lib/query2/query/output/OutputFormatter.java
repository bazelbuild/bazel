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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Comparator;
import java.util.Iterator;

/**
 * Interface for classes which order, format and print the result of a Blaze
 * graph query.
 */
public abstract class OutputFormatter implements Serializable {

  /** Where the value of an attribute comes from */
  enum AttributeValueSource {
    RULE,     // Explicitly specified on the rule
    PACKAGE,  // Package default
    DEFAULT   // Rule class default
  }

  static final Function<Node<Target>, Target> EXTRACT_NODE_LABEL = Node::getLabel;

  public static ImmutableList<OutputFormatter> getDefaultFormatters() {
    return ImmutableList.of(
        new LabelOutputFormatter(false),
        new LabelOutputFormatter(true),
        new BuildOutputFormatter(),
        new MinrankOutputFormatter(),
        new MaxrankOutputFormatter(),
        new PackageOutputFormatter(),
        new LocationOutputFormatter(),
        new GraphOutputFormatter(),
        new XmlOutputFormatter(),
        new ProtoOutputFormatter(),
        new StreamedProtoOutputFormatter());
  }

  public static String formatterNames(Iterable<OutputFormatter> formatters) {
    return Streams.stream(formatters).map(OutputFormatter::getName).collect(joining(", "));
  }

  public static String streamingFormatterNames(Iterable<OutputFormatter> formatters) {
    return Streams.stream(formatters)
        .filter(Predicates.instanceOf(StreamedFormatter.class))
        .map(OutputFormatter::getName)
        .collect(joining(", "));
  }

  /** Returns the output formatter for the specified command-line options. */
  public static OutputFormatter getFormatter(Iterable<OutputFormatter> formatters, String type) {
    for (OutputFormatter formatter : formatters) {
      if (formatter.getName().equals(type)) {
        return formatter;
      }
    }

    return null;
  }

  /**
   * Given a set of query options, returns a BinaryPredicate suitable for passing to {@link
   * Rule#getLabels()}, {@link XmlOutputFormatter}, etc.
   */
  static DependencyFilter getDependencyFilter(CommonQueryOptions queryOptions) {
    // TODO(bazel-team): Optimize: and(ALL_DEPS, x) -> x, etc.
    return DependencyFilter.and(
        queryOptions.includeHostDeps ? DependencyFilter.ALL_DEPS : DependencyFilter.NO_HOST_DEPS,
        queryOptions.includeImplicitDeps
            ? DependencyFilter.ALL_DEPS
            : DependencyFilter.NO_IMPLICIT_DEPS);
  }

  /** Returns the user-visible name of the output formatter. */
  public abstract String getName();

  /**
   * Workaround for a bug in {@link java.nio.channels.Channels#newChannel(OutputStream)}, which
   * attempts to close the output stream on interrupt, which can cause a deadlock if there is an
   * ongoing write. If this formatter uses Channels.newChannel, then it must return false here, and
   * perform its own buffering.
   */
  public boolean canBeBuffered() {
    return true;
  }

  /**
   * Verifies that the environment and expression are compatible with this formatter, throws a
   * {@link QueryException} if not.
   */
  public void verifyCompatible(QueryEnvironment<?> env, QueryExpression expr)
      throws QueryException {}

  /**
   * Format the result (a set of target nodes implicitly ordered according to the graph maintained
   * by the QueryEnvironment), and print it to "out".
   */
  public abstract void output(
      QueryOptions options,
      Digraph<Target> result,
      OutputStream out,
      AspectResolver aspectProvider,
      ConditionalEdges conditionalEdges)
      throws IOException, InterruptedException;

  /** An ordering of Targets based on the ordering of their labels. */
  static class TargetOrdering implements Comparator<Target> {
    @Override
    public int compare(Target o1, Target o2) {
      return o1.getLabel().compareTo(o2.getLabel());
    }
  }

  /** Helper class for {@link #getPossibleAttributeValues}. */
  public static class PossibleAttributeValues implements Iterable<Object> {
    final Iterable<Object> values;
    final AttributeValueSource source;

    public PossibleAttributeValues(Iterable<Object> values, AttributeValueSource source) {
      this.values = values;
      this.source = source;
    }

    @Override
    public Iterator<Object> iterator() {
      return values.iterator();
    }
  }

  public static AttributeValueSource getAttributeSource(Rule rule, Attribute attr) {
    if (attr.getName().equals("visibility")) {
      if (rule.isVisibilitySpecified()) {
        return AttributeValueSource.RULE;
      } else if (rule.getPackage().isDefaultVisibilitySet()) {
        return AttributeValueSource.PACKAGE;
      } else {
        return AttributeValueSource.DEFAULT;
      }
    } else {
      return rule.isAttributeValueExplicitlySpecified(attr)
          ? AttributeValueSource.RULE
          : AttributeValueSource.DEFAULT;
    }
  }

  /**
   * Returns the possible values of the specified attribute in the specified rule. For simple
   * attributes, this is a single value. For configurable and computed attributes, this may be a
   * list of values. See {@link AggregatingAttributeMapper#getPossibleAttributeValues} for how the
   * values are determined.
   *
   * <p>This applies an important optimization for label lists: instead of returning all possible
   * values, it only returns possible <i>labels</i>. For example, given:
   *
   * <pre>
   * select({
   *     ":c": ["//a:one", "//a:two"],
   *     ":d": ["//a:two"]
   *     })</pre>
   *
   * it returns:
   *
   * <pre>["//a:one", "//a:two"]</pre>
   *
   * which loses track of which label appears in which branch.
   *
   * <p>This avoids the memory overruns that can happen be iterating over every possible value for
   * an <code>attr = select(...) + select(...) + select(...) + ...</code> expression. Query
   * operations generally don't care about specific attribute values - they just care which labels
   * are possible.
   */
  static PossibleAttributeValues getPossibleAttributeValues(Rule rule, Attribute attr) {
    AttributeValueSource source = getAttributeSource(rule, attr);
    AggregatingAttributeMapper attributeMap = AggregatingAttributeMapper.of(rule);
    Iterable<?> list;
    if (attr.getType().equals(BuildType.LABEL_LIST)
        && attributeMap.isConfigurable(attr.getName())) {
      // TODO(gregce): Expand this to all collection types (we don't do this for scalars because
      // there's currently no syntax for expressing multiple scalar values). This unfortunately
      // isn't trivial because Bazel's label visitation logic includes special methods built
      // directly into Type.
      return new PossibleAttributeValues(
          ImmutableList.<Object>of(
              attributeMap.getReachableLabels(attr.getName(), /*includeSelectKeys=*/ false)),
          source);
    } else if ((list =
            attributeMap.getConcatenatedSelectorListsOfListType(
                attr.getName(), attr.getType()))
        != null) {
      return new PossibleAttributeValues(Lists.newArrayList(list), source);
    } else {
      // The call to getPossibleAttributeValues below is especially slow with selector lists.
      return new PossibleAttributeValues(attributeMap.getPossibleAttributeValues(rule, attr),
          source);
    }
  }

  /**
   * Returns the target location, eventually stripping out the workspace path to obtain a relative
   * target (stable across machines / workspaces).
   *
   * @param target The target to extract location from.
   * @param relative Whether to return a relative path or not.
   * @return the target location
   */
  static String getLocation(Target target, boolean relative) {
    Location location = target.getLocation();
    return relative
        ? location.print(target.getPackage().getPackageDirectory().asFragment(),
            target.getPackage().getNameFragment())
        : location.print();
  }
}
