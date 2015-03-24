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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageSerializer;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.BinaryPredicate;
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
  public static BinaryPredicate<Rule, Attribute> getDependencyFilter(QueryOptions queryOptions) {
    // TODO(bazel-team): Optimize: and(ALL_DEPS, x) -> x, etc.
    return Rule.and(
          queryOptions.includeHostDeps ? Rule.ALL_DEPS : Rule.NO_HOST_DEPS,
          queryOptions.includeImplicitDeps ? Rule.ALL_DEPS : Rule.NO_IMPLICIT_DEPS);
  }

  /**
   * Format the result (a set of target nodes implicitly ordered according to
   * the graph maintained by the QueryEnvironment), and print it to "out".
   */
  public abstract void output(QueryOptions options, Digraph<Target> result, PrintStream out)
      throws IOException;

  /**
   * Unordered output formatter (wrt. dependency ordering).
   *
   * <p>Formatters that support unordered output may be used when only the set of query results is
   * requested but their ordering is irrelevant.
   *
   * <p>The benefit of using a unordered formatter is that we can save the potentially expensive
   * subgraph extraction step before presenting the query results.
   */
  public interface UnorderedFormatter {
    void outputUnordered(QueryOptions options, Iterable<Target> result, PrintStream out)
        throws IOException;
  }

  /**
   * Returns the user-visible name of the output formatter.
   */
  public abstract String getName();

  /**
   * An output formatter that prints the labels of the resulting target set in
   * topological order, optionally with the target's kind.
   */
  private static class LabelOutputFormatter extends OutputFormatter implements UnorderedFormatter{

    private final boolean showKind;

    public LabelOutputFormatter(boolean showKind) {
      this.showKind = showKind;
    }

    @Override
    public String getName() {
      return showKind ? "label_kind" : "label";
    }

    @Override
    public void outputUnordered(QueryOptions options, Iterable<Target> result, PrintStream out) {
      for (Target target : result) {
        if (showKind) {
          out.print(target.getTargetKind());
          out.print(' ');
        }
        out.println(target.getLabel());
      }
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
      Iterable<Target> ordered = Iterables.transform(
          result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
      outputUnordered(options, ordered, out);
    }
  }

  /**
   * An ordering of Targets based on the ordering of their labels.
   */
  static class TargetOrdering implements Comparator<Target> {
    @Override
    public int compare(Target o1, Target o2) {
      return o1.getLabel().compareTo(o2.getLabel());
    }
  }

  /**
   * An output formatter that prints the names of the packages of the target
   * set, in lexicographical order without duplicates.
   */
  private static class PackageOutputFormatter extends OutputFormatter implements
      UnorderedFormatter {
    @Override
    public String getName() {
      return "package";
    }

    @Override
    public void outputUnordered(QueryOptions options, Iterable<Target> result, PrintStream out) {
      Set<String> packageNames = Sets.newTreeSet();
      for (Target target : result) {
        packageNames.add(target.getLabel().getPackageName());
      }
      for (String packageName : packageNames) {
        out.println(packageName);
      }
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
      Iterable<Target> ordered = Iterables.transform(
          result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
      outputUnordered(options, ordered, out);
    }
  }

  /**
   * An output formatter that prints the labels of the targets, preceded by
   * their locations and kinds, in topological order.  For output files, the
   * location of the generating rule is given; for input files, the location of
   * line 1 is given.
   */
  private static class LocationOutputFormatter extends OutputFormatter implements
      UnorderedFormatter {
    @Override
    public String getName() {
      return "location";
    }

    @Override
    public void outputUnordered(QueryOptions options, Iterable<Target> result, PrintStream out) {
      for (Target target : result) {
        Location location = target.getLocation();
        out.println(location.print()  + ": " + target.getTargetKind() + " " + target.getLabel());
      }
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
      Iterable<Target> ordered = Iterables.transform(
          result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
      outputUnordered(options, ordered, out);
    }
  }

  /**
   * An output formatter that prints the generating rules using the syntax of
   * the BUILD files. If multiple targets are generated by the same rule, it is
   * printed only once.
   */
  private static class BuildOutputFormatter extends OutputFormatter implements UnorderedFormatter {
    @Override
    public String getName() {
      return "build";
    }

    private void outputRule(Rule rule, PrintStream out) {
      out.printf("# %s%n", rule.getLocation());
      out.printf("%s(%n", rule.getRuleClass());
      out.printf("  name = \"%s\",%n", rule.getName());

      for (Attribute attr : rule.getAttributes()) {
        Pair<Iterable<Object>, AttributeValueSource> values = getAttributeValues(rule, attr);
        if (Iterables.size(values.first) != 1) {
          continue;  // TODO(bazel-team): handle configurable attributes.
        }
        if (values.second != AttributeValueSource.RULE) {
          continue;  // Don't print default values.
        }
        Object value = Iterables.getOnlyElement(values.first);
        out.printf("  %s = ", attr.getName());
        if (value instanceof Label) {
          value = value.toString();
        } else if (value instanceof List<?> && EvalUtils.isImmutable(value)) {
          // Display it as a list (and not as a tuple). Attributes can never be tuples.
          value = new ArrayList<>((List<?>) value);
        }
        EvalUtils.prettyPrintValue(value, out);
        out.println(",");
      }
      out.printf(")\n%n");
    }

    @Override
    public void outputUnordered(QueryOptions options, Iterable<Target> result, PrintStream out) {
      Set<Label> printed = new HashSet<>();
      for (Target target : result) {
        Rule rule = target.getAssociatedRule();
        if (rule == null || printed.contains(rule.getLabel())) {
          continue;
        }
        outputRule(rule, out);
        printed.add(rule.getLabel());
      }
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
      Iterable<Target> ordered = Iterables.transform(
          result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
      outputUnordered(options, ordered, out);
    }
  }

  /**
   * An output formatter that prints the labels in minimum rank order, preceded by
   * their rank number.  "Roots" have rank 0, their direct prerequisites have
   * rank 1, etc.  All nodes in a cycle are considered of equal rank.  MINRANK
   * shows the lowest rank for a given node, i.e. the length of the shortest
   * path from a zero-rank node to it.
   *
   * If the result came from a <code>deps(x)</code> query, then the MINRANKs
   * correspond to the shortest path from x to each of its prerequisites.
   */
  private static class MinrankOutputFormatter extends OutputFormatter {
    @Override
    public String getName() {
      return "minrank";
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
      // getRoots() isn't defined for cyclic graphs, so in order to handle
      // cycles correctly, we need work on the strong component graph, as
      // cycles should be treated a "clump" of nodes all on the same rank.
      // Graphs may contain cycles because there are errors in BUILD files.

      Digraph<Set<Node<Target>>> scGraph = result.getStrongComponentGraph();
      Set<Node<Set<Node<Target>>>> rankNodes = scGraph.getRoots();
      Set<Node<Set<Node<Target>>>> seen = new HashSet<>();
      seen.addAll(rankNodes);
      for (int rank = 0; !rankNodes.isEmpty(); rank++) {
        // Print out this rank:
        for (Node<Set<Node<Target>>> xScc : rankNodes) {
          for (Node<Target> x : xScc.getLabel()) {
            out.println(rank + " " + x.getLabel().getLabel());
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
   * If the result came from a <code>deps(x)</code> query, then the MAXRANKs
   * correspond to the longest path from x to each of its prerequisites.
   */
  private static class MaxrankOutputFormatter extends OutputFormatter {
    @Override
    public String getName() {
      return "maxrank";
    }

    @Override
    public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
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
      List<Pair<Integer, Label>> output = new ArrayList<>();
      for (Node<Set<Node<Target>>> x : result.getStrongComponentGraph().getNodes()) {
        int rank = dp.rank(x);
        for (Node<Target> y : x.getLabel()) {
          output.add(Pair.of(rank, y.getLabel().getLabel()));
        }
      }
      Collections.sort(output, new Comparator<Pair<Integer, Label>>() {
          @Override
          public int compare(Pair<Integer, Label> x, Pair<Integer, Label> y) {
            return x.first - y.first;
          }
        });

      for (Pair<Integer, Label> pair : output) {
        out.println(pair.first + " " + pair.second);
      }
    }
  }

  /**
   * Returns the possible values of the specified attribute in the specified rule. For
   * non-configured attributes, this is a single value. For configurable attributes, this
   * may be multiple values.
   *
   * @return a pair, where the first value is the set of possible values and the
   *     second is an enum that tells where the values come from (declared on the
   *     rule, declared as a package level default or a
   *     global default)
   */
  protected static Pair<Iterable<Object>, AttributeValueSource> getAttributeValues(
      Rule rule, Attribute attr) {
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

    return Pair.of(PackageSerializer.getAttributeValues(rule, attr), source);
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
