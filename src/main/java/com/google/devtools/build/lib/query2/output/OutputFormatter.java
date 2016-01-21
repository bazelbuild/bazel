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
package com.google.devtools.build.lib.query2.output;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.common.options.EnumConverter;

import java.io.IOException;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Interface for classes which order, format and print the result of a Blaze
 * graph query.
 */
public abstract class OutputFormatter implements Serializable {

  /**
   * Discriminator for different kinds of OutputFormatter.
   */
  public enum Type {
    LABEL,
    LABEL_KIND,
    BUILD,
    MINRANK,
    MAXRANK,
    PACKAGE,
    LOCATION,
    GRAPH,
    XML,
    PROTO,
    RECORD,
  }

  /**
   * Where the value of an attribute comes from
   */
  protected enum AttributeValueSource {
    RULE,     // Explicitly specified on the rule
    PACKAGE,  // Package default
    DEFAULT   // Rule class default
  }

  public static final Function<Node<Target>, Target> EXTRACT_NODE_LABEL =
      new Function<Node<Target>, Target>() {
        @Override
        public Target apply(Node<Target> input) {
          return input.getLabel();
        }
      };

  /**
   * Converter from strings to OutputFormatter.Type.
   */
  public static class Converter extends EnumConverter<Type> {
    public Converter() { super(Type.class, "output formatter"); }
  }

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
        new ProtoOutputFormatter());
  }

  public static String formatterNames(Iterable<OutputFormatter> formatters) {
    return Joiner.on(", ").join(Iterables.transform(formatters,
        new Function<OutputFormatter, String>() {
          @Override
          public String apply(OutputFormatter input) {
            return input.getName();
          }
    }));
  }

  /**
   * Returns the output formatter for the specified command-line options.
   */
  public static OutputFormatter getFormatter(
      Iterable<OutputFormatter> formatters, String type) {
    for (OutputFormatter formatter : formatters) {
      if (formatter.getName().equals(type)) {
        return formatter;
      }
    }

    return null;
  }

  /**
   * Given a set of query options, returns a BinaryPredicate suitable for
   * passing to {@link Rule#getLabels()}, {@link XmlOutputFormatter}, etc.
   */
  public static DependencyFilter getDependencyFilter(QueryOptions queryOptions) {
    // TODO(bazel-team): Optimize: and(ALL_DEPS, x) -> x, etc.
    return DependencyFilter.and(
        queryOptions.includeHostDeps ? DependencyFilter.ALL_DEPS : DependencyFilter.NO_HOST_DEPS,
        queryOptions.includeImplicitDeps
            ? DependencyFilter.ALL_DEPS
            : DependencyFilter.NO_IMPLICIT_DEPS);
  }

  /**
   * Format the result (a set of target nodes implicitly ordered according to
   * the graph maintained by the QueryEnvironment), and print it to "out".
   */
  public abstract void output(QueryOptions options, Digraph<Target> result, PrintStream out,
      AspectResolver aspectProvider) throws IOException, InterruptedException;

  /**
   * Unordered streamed output formatter (wrt. dependency ordering).
   *
   * <p>Formatters that support streamed output may be used when only the set of query results is
   * requested but their ordering is irrelevant.
   *
   * <p>The benefit of using a streamed formatter is that we can save the potentially expensive
   * subgraph extraction step before presenting the query results and that depending on the query
   * environment used, it can be more memory performant, as it does not aggregate all the data
   * before writting in the output.
   */
  public interface StreamedFormatter {

    OutputFormatterCallback<Target> createStreamCallback(QueryOptions options, PrintStream out,
        AspectResolver aspectResolver);
  }

  /**
   * Returns the user-visible name of the output formatter.
   */
  public abstract String getName();

  abstract static class AbstractUnorderedFormatter extends OutputFormatter
      implements StreamedFormatter {
    protected Iterable<Target> getOrderedTargets(
        Digraph<Target> result, QueryOptions options) {
      Iterable<Node<Target>> orderedResult =
          options.orderOutput == OrderOutput.DEPS
              ? result.getTopologicalOrder()
              : result.getTopologicalOrder(new TargetOrdering());
      return Iterables.transform(orderedResult, EXTRACT_NODE_LABEL);
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out,
        AspectResolver aspectResolver) throws IOException, InterruptedException {
      OutputFormatterCallback.processAllTargets(
          createStreamCallback(options, out, aspectResolver),
          getOrderedTargets(result, options));
    }
  }

  /**
   * An output formatter that prints the labels of the resulting target set in
   * topological order, optionally with the target's kind.
   */
  private static class LabelOutputFormatter extends AbstractUnorderedFormatter {

    private final boolean showKind;

    private LabelOutputFormatter(boolean showKind) {
      this.showKind = showKind;
    }

    @Override
    public String getName() {
      return showKind ? "label_kind" : "label";
    }

    @Override
    public OutputFormatterCallback<Target> createStreamCallback(QueryOptions options,
        final PrintStream out, AspectResolver aspectResolver) {
      return new OutputFormatterCallback<Target>() {

        @Override
        protected void processOutput(Iterable<Target> partialResult)
            throws IOException, InterruptedException {
          for (Target target : partialResult) {
            if (showKind) {
              out.print(target.getTargetKind());
              out.print(' ');
            }
            out.println(target.getLabel());
          }
        }
      };
    }
  }

  /**
   * An ordering of Targets based on the ordering of their labels.
   */
  @VisibleForTesting
  public static class TargetOrdering implements Comparator<Target> {
    @Override
    public int compare(Target o1, Target o2) {
      return o1.getLabel().compareTo(o2.getLabel());
    }
  }

  /**
   * An output formatter that prints the names of the packages of the target
   * set, in lexicographical order without duplicates.
   */
  private static class PackageOutputFormatter extends AbstractUnorderedFormatter {
    @Override
    public String getName() {
      return "package";
    }

    @Override
    public OutputFormatterCallback<Target> createStreamCallback(QueryOptions options,
        final PrintStream out,
        AspectResolver aspectResolver) {
      return new OutputFormatterCallback<Target>() {
        private final Set<String> packageNames = Sets.newTreeSet();

        @Override
        protected void processOutput(Iterable<Target> partialResult)
            throws IOException, InterruptedException {

          for (Target target : partialResult) {
            packageNames.add(target.getLabel().getPackageName());
          }
        }

        @Override
        public void close() throws IOException {
          for (String packageName : packageNames) {
            out.println(packageName);
          }
        }
      };
    }
  }

  /**
   * An output formatter that prints the labels of the targets, preceded by
   * their locations and kinds, in topological order.  For output files, the
   * location of the generating rule is given; for input files, the location of
   * line 1 is given.
   */
  private static class LocationOutputFormatter extends AbstractUnorderedFormatter {
    @Override
    public String getName() {
      return "location";
    }

    @Override
    public OutputFormatterCallback<Target> createStreamCallback(QueryOptions options,
        final PrintStream out,
        AspectResolver aspectResolver) {
      return new OutputFormatterCallback<Target>() {

        @Override
        protected void processOutput(Iterable<Target> partialResult)
            throws IOException, InterruptedException {
          for (Target target : partialResult) {
            Location location = target.getLocation();
            out.println(location.print() + ": " + target.getTargetKind() + " " + target.getLabel());
          }
        }
      };
    }
  }

  /**
   * An output formatter that prints the generating rules using the syntax of
   * the BUILD files. If multiple targets are generated by the same rule, it is
   * printed only once.
   */
  private static class BuildOutputFormatter extends AbstractUnorderedFormatter {
    @Override
    public String getName() {
      return "build";
    }

    @Override
    public OutputFormatterCallback<Target> createStreamCallback(QueryOptions options,
        final PrintStream out,
        AspectResolver aspectResolver) {
      return new OutputFormatterCallback<Target>() {
        private final Set<Label> printed = CompactHashSet.create();

        private void outputRule(Rule rule, PrintStream out) {
          out.printf("# %s%n", rule.getLocation());
          out.printf("%s(%n", rule.getRuleClass());
          out.printf("  name = \"%s\",%n", rule.getName());

          for (Attribute attr : rule.getAttributes()) {
            Pair<Iterable<Object>, AttributeValueSource> values =
                getPossibleAttributeValuesAndSources(rule, attr);
            if (Iterables.size(values.first) != 1) {
              continue; // TODO(bazel-team): handle configurable attributes.
            }
            if (values.second != AttributeValueSource.RULE) {
              continue; // Don't print default values.
            }
            Object value = Iterables.getOnlyElement(values.first);
            out.printf("  %s = ", attr.getPublicName());
            if (value instanceof Label) {
              value = value.toString();
            } else if (value instanceof List<?> && EvalUtils.isImmutable(value)) {
              // Display it as a list (and not as a tuple). Attributes can never be tuples.
              value = new ArrayList<>((List<?>) value);
            }
            // It is *much* faster to write to a StringBuilder compared to the PrintStream object.
            StringBuilder builder = new StringBuilder();
            Printer.write(builder, value);
            out.print(builder);
            out.println(",");
          }
          out.printf(")\n%n");
        }

        @Override
        protected void processOutput(Iterable<Target> partialResult)
            throws IOException, InterruptedException {

          for (Target target : partialResult) {
            Rule rule = target.getAssociatedRule();
            if (rule == null || printed.contains(rule.getLabel())) {
              continue;
            }
            outputRule(rule, out);
            printed.add(rule.getLabel());
          }
        }
      };
    }
  }

  private static class RankAndLabel implements Comparable<RankAndLabel> {
    private final int rank;
    private final Label label;

    private RankAndLabel(int rank, Label label) {
      this.rank = rank;
      this.label = label;
    }

    @Override
    public int compareTo(RankAndLabel o) {
      if (this.rank != o.rank) {
        return this.rank - o.rank;
      }
      return this.label.compareTo(o.label);
    }

    @Override
    public String toString() {
      return rank + " " + label;
    }
  }

  /**
   * An output formatter that prints the labels in minimum rank order, preceded by
   * their rank number.  "Roots" have rank 0, their direct prerequisites have
   * rank 1, etc.  All nodes in a cycle are considered of equal rank.  MINRANK
   * shows the lowest rank for a given node, i.e. the length of the shortest
   * path from a zero-rank node to it.
   *
   * <p>If the result came from a <code>deps(x)</code> query, then the MINRANKs
   * correspond to the shortest path from x to each of its prerequisites.
   */
  private static class MinrankOutputFormatter extends OutputFormatter {
    @Override
    public String getName() {
      return "minrank";
    }

    private static void outputToStreamOrSave(
        int rank, Label label, PrintStream out, @Nullable List<RankAndLabel> toSave) {
      if (toSave != null) {
        toSave.add(new RankAndLabel(rank, label));
      } else {
        out.println(rank + " " + label);
      }
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out,
        AspectResolver aspectResolver) {
      // getRoots() isn't defined for cyclic graphs, so in order to handle
      // cycles correctly, we need work on the strong component graph, as
      // cycles should be treated a "clump" of nodes all on the same rank.
      // Graphs may contain cycles because there are errors in BUILD files.

      List<RankAndLabel> outputToOrder =
          options.orderOutput == OrderOutput.FULL ? new ArrayList<RankAndLabel>() : null;
      Digraph<Set<Node<Target>>> scGraph = result.getStrongComponentGraph();
      Set<Node<Set<Node<Target>>>> rankNodes = scGraph.getRoots();
      Set<Node<Set<Node<Target>>>> seen = new HashSet<>();
      seen.addAll(rankNodes);
      for (int rank = 0; !rankNodes.isEmpty(); rank++) {
        // Print out this rank:
        for (Node<Set<Node<Target>>> xScc : rankNodes) {
          for (Node<Target> x : xScc.getLabel()) {
            outputToStreamOrSave(rank, x.getLabel().getLabel(), out, outputToOrder);
          }
        }

        // Find the next rank:
        Set<Node<Set<Node<Target>>>> nextRankNodes = new LinkedHashSet<>();
        for (Node<Set<Node<Target>>> x : rankNodes) {
          for (Node<Set<Node<Target>>> y : x.getSuccessors()) {
            if (seen.add(y)) {
              nextRankNodes.add(y);
            }
          }
        }
        rankNodes = nextRankNodes;
      }
      if (outputToOrder != null) {
        Collections.sort(outputToOrder);
        for (RankAndLabel item : outputToOrder) {
          out.println(item);
        }
      }
    }
  }

  /**
   * An output formatter that prints the labels in maximum rank order, preceded
   * by their rank number.  "Roots" have rank 0, all other nodes have a rank
   * which is one greater than the maximum rank of each of their predecessors.
   * All nodes in a cycle are considered of equal rank.  MAXRANK shows the
   * highest rank for a given node, i.e. the length of the longest non-cyclic
   * path from a zero-rank node to it.
   *
   * <p>If the result came from a <code>deps(x)</code> query, then the MAXRANKs
   * correspond to the longest path from x to each of its prerequisites.
   */
  private static class MaxrankOutputFormatter extends OutputFormatter {
    @Override
    public String getName() {
      return "maxrank";
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out,
        AspectResolver aspectResolver) {
      // In order to handle cycles correctly, we need work on the strong
      // component graph, as cycles should be treated a "clump" of nodes all on
      // the same rank. Graphs may contain cycles because there are errors in BUILD files.

      // Dynamic programming algorithm:
      // rank(x) = max(rank(p)) + 1 foreach p in preds(x)
      // TODO(bazel-team): Move to Digraph.
      class DP {
        final Map<Node<Set<Node<Target>>>, Integer> ranks = new HashMap<>();

        int rank(Node<Set<Node<Target>>> node) {
          Integer rank = ranks.get(node);
          if (rank == null) {
            int maxPredRank = -1;
            for (Node<Set<Node<Target>>> p : node.getPredecessors()) {
              maxPredRank = Math.max(maxPredRank, rank(p));
            }
            rank = maxPredRank + 1;
            ranks.put(node, rank);
          }
          return rank;
        }
      }
      DP dp = new DP();

      // Now sort by rank...
      List<RankAndLabel> output = new ArrayList<>();
      for (Node<Set<Node<Target>>> x : result.getStrongComponentGraph().getNodes()) {
        int rank = dp.rank(x);
        for (Node<Target> y : x.getLabel()) {
          output.add(new RankAndLabel(rank, y.getLabel().getLabel()));
        }
      }
      if (options.orderOutput == OrderOutput.FULL) {
        // Use the natural order for RankAndLabels, which breaks ties alphabetically.
        Collections.sort(output);
      } else {
        Collections.sort(
            output,
            new Comparator<RankAndLabel>() {
              @Override
              public int compare(RankAndLabel o1, RankAndLabel o2) {
                return o1.rank - o2.rank;
              }
            });
      }
      for (RankAndLabel item : output) {
        out.println(item);
      }
    }
  }

  /**
   * Returns the possible values of the specified attribute in the specified rule. For simple
   * attributes, this is a single value. For configurable and computed attributes, this may be a
   * list of values. See {@link AggregatingAttributeMapper#getPossibleAttributeValues} for how the
   * value(s) is/are made.
   *
   * @return a pair, where the first value is the set of possible values and the
   *     second is an enum that tells where the values come from (declared on the
   *     rule, declared as a package level default or a
   *     global default)
   */
  protected static Pair<Iterable<Object>, AttributeValueSource>
      getPossibleAttributeValuesAndSources(Rule rule, Attribute attr) {
    AttributeValueSource source;

    if (attr.getName().equals("visibility")) {
      if (rule.isVisibilitySpecified()) {
        source = AttributeValueSource.RULE;
      } else if (rule.getPackage().isDefaultVisibilitySet()) {
        source = AttributeValueSource.PACKAGE;
      } else {
        source = AttributeValueSource.DEFAULT;
      }
    } else {
      source = rule.isAttributeValueExplicitlySpecified(attr)
          ? AttributeValueSource.RULE : AttributeValueSource.DEFAULT;
    }

    Iterable<Object> possibleAttributeValues =
        AggregatingAttributeMapper.of(rule).getPossibleAttributeValues(rule, attr);
    return Pair.of(possibleAttributeValues, source);
  }

  /**
   * Returns the target location, eventually stripping out the workspace path to obtain a relative
   * target (stable across machines / workspaces).
   *
   * @param target The target to extract location from.
   * @param relative Whether to return a relative path or not.
   * @return the target location
   */
  protected static String getLocation(Target target, boolean relative) {
    Location location = target.getLocation();
    return relative
        ? location.print(target.getPackage().getPackageDirectory().asFragment(),
            target.getPackage().getNameFragment())
        : location.print();
  }
}
