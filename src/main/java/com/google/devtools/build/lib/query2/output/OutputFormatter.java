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

import static java.util.Comparator.comparingInt;
import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.AggregatingQueryExpressionVisitor.ContainsFunctionQueryExpressionVisitor;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.common.options.EnumConverter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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
  public enum OutputType {
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
  }

  /**
   * Where the value of an attribute comes from
   */
  protected enum AttributeValueSource {
    RULE,     // Explicitly specified on the rule
    PACKAGE,  // Package default
    DEFAULT   // Rule class default
  }

  public static final Function<Node<Target>, Target> EXTRACT_NODE_LABEL = Node::getLabel;

  /**
   * Converter from strings to OutputFormatter.OutputType.
   */
  public static class Converter extends EnumConverter<OutputType> {
    public Converter() { super(OutputType.class, "output formatter"); }
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
    return Streams.stream(formatters).map(OutputFormatter::getName).collect(joining(", "));
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
  public static DependencyFilter getDependencyFilter(
      CommonQueryOptions queryOptions) {
    // TODO(bazel-team): Optimize: and(ALL_DEPS, x) -> x, etc.
    return DependencyFilter.and(
        queryOptions.includeHostDeps ? DependencyFilter.ALL_DEPS : DependencyFilter.NO_HOST_DEPS,
        queryOptions.includeImplicitDeps
            ? DependencyFilter.ALL_DEPS
            : DependencyFilter.NO_IMPLICIT_DEPS);
  }

  /**
   * Workaround for a bug in {@link java.nio.channels.Channels#newChannel(OutputStream)}, which
   * attempts to close the output stream on interrupt, which can cause a deadlock if there is an
   * ongoing write. If this formatter uses Channels.newChannel, then it must return false here, and
   * perform its own buffering.
   */
  public boolean canBeBuffered() {
    return true;
  }

  public void verifyCompatible(QueryEnvironment<?> env, QueryExpression expr)
      throws QueryException {
  }

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

  /**
   * Unordered streamed output formatter (wrt. dependency ordering).
   *
   * <p>Formatters that support streamed output may be used when only the set of query results is
   * requested but their ordering is irrelevant.
   *
   * <p>The benefit of using a streamed formatter is that we can save the potentially expensive
   * subgraph extraction step before presenting the query results and that depending on the query
   * environment used, it can be more memory performant, as it does not aggregate all the data
   * before writing in the output.
   */
  public interface StreamedFormatter {
    /** Specifies options to be used by subsequent calls to {@link #createStreamCallback}. */
    void setOptions(CommonQueryOptions options, AspectResolver aspectResolver);

    /**
     * Returns a {@link ThreadSafeOutputFormatterCallback} whose
     * {@link OutputFormatterCallback#process} outputs formatted {@link Target}s to the given
     * {@code out}.
     *
     * <p>Takes any options specified via the most recent call to {@link #setOptions} into
     * consideration.
     *
     * <p>Intended to be use for streaming out during evaluation of a query.
     */
    ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
        OutputStream out, QueryOptions options, QueryEnvironment<?> env);

    /**
     * Same as {@link #createStreamCallback}, but intended to be used for outputting the
     * already-computed result of a query.
     */
    OutputFormatterCallback<Target> createPostFactoStreamCallback(
        OutputStream out, QueryOptions options);
  }

  /**
   * Returns the user-visible name of the output formatter.
   */
  public abstract String getName();

  abstract static class AbstractUnorderedFormatter extends OutputFormatter
      implements StreamedFormatter {
    protected CommonQueryOptions options;
    protected AspectResolver aspectResolver;
    protected DependencyFilter dependencyFilter;

    protected Iterable<Target> getOrderedTargets(
        Digraph<Target> result, QueryOptions options) {
      Iterable<Node<Target>> orderedResult =
          options.orderOutput == OrderOutput.DEPS
              ? result.getTopologicalOrder()
              : result.getTopologicalOrder(new TargetOrdering());
      return Iterables.transform(orderedResult, EXTRACT_NODE_LABEL);
    }

    @Override
    public void setOptions(CommonQueryOptions options, AspectResolver aspectResolver) {
      this.options = options;
      this.aspectResolver = aspectResolver;
      this.dependencyFilter = OutputFormatter.getDependencyFilter(options);
    }

    @Override
    public void output(
        QueryOptions options,
        Digraph<Target> result,
        OutputStream out,
        AspectResolver aspectResolver,
        ConditionalEdges conditionalEdges)
        throws IOException, InterruptedException {
      setOptions(options, aspectResolver);
      OutputFormatterCallback.processAllTargets(
          createPostFactoStreamCallback(out, options), getOrderedTargets(result, options));
    }
  }

  /** Abstract class supplying a {@link PrintStream} to implementations, flushing it on close. */
  private abstract static class TextOutputFormatterCallback<T> extends OutputFormatterCallback<T> {
    protected PrintStream printStream;

    private TextOutputFormatterCallback(OutputStream out) {
      this.printStream = new PrintStream(out);
    }

    @Override
    public void close(boolean failFast) throws IOException {
      if (!failFast) {
        flushAndCheckError(printStream);
      }
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
    public OutputFormatterCallback<Target> createPostFactoStreamCallback(
        OutputStream out, final QueryOptions options) {
      return new TextOutputFormatterCallback<Target>(out) {
        @Override
        public void processOutput(Iterable<Target> partialResult) {
          for (Target target : partialResult) {
            if (showKind) {
              printStream.print(target.getTargetKind());
              printStream.print(' ');
            }
            printStream.printf(
                "%s%s", target.getLabel().getDefaultCanonicalForm(), options.getLineTerminator());
          }
        }
      };
    }

    @Override
    public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
        OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
      return new SynchronizedDelegatingOutputFormatterCallback<>(
          createPostFactoStreamCallback(out, options));
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
    public OutputFormatterCallback<Target> createPostFactoStreamCallback(
        OutputStream out, final QueryOptions options) {
      return new TextOutputFormatterCallback<Target>(out) {
        private final Set<String> packageNames = Sets.newTreeSet();

        @Override
        public void processOutput(Iterable<Target> partialResult) {

          for (Target target : partialResult) {
            packageNames.add(target.getLabel().getPackageIdentifier().toString());
          }
        }

        @Override
        public void close(boolean failFast) throws IOException {
          if (!failFast) {
            final String lineTerm = options.getLineTerminator();
            for (String packageName : packageNames) {
              printStream.printf("%s%s", packageName, lineTerm);
            }
          }
          super.close(failFast);
        }
      };
    }

    @Override
    public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
        OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
      return new SynchronizedDelegatingOutputFormatterCallback<>(
          createPostFactoStreamCallback(out, options));
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
    public void verifyCompatible(QueryEnvironment<?> env, QueryExpression expr)
        throws QueryException {
      if (!(env instanceof AbstractBlazeQueryEnvironment)) {
        return;
      }

      ContainsFunctionQueryExpressionVisitor noteBuildFilesAndLoadLilesVisitor =
          new ContainsFunctionQueryExpressionVisitor(ImmutableList.of("loadfiles", "buildfiles"));

      if (expr.accept(noteBuildFilesAndLoadLilesVisitor)) {
        throw new QueryException(
            "Query expressions involving 'buildfiles' or 'loadfiles' cannot be used with "
            + "--output=location");
      }
    }

    @Override
    public OutputFormatterCallback<Target> createPostFactoStreamCallback(
        OutputStream out, final QueryOptions options) {
      return new TextOutputFormatterCallback<Target>(out) {

        @Override
        public void processOutput(Iterable<Target> partialResult) {
          final String lineTerm = options.getLineTerminator();
          for (Target target : partialResult) {
            Location location = target.getLocation();
            printStream.print(
                location.print()
                    + ": "
                    + target.getTargetKind()
                    + " "
                    + target.getLabel().getDefaultCanonicalForm()
                    + lineTerm);
          }
        }
      };
    }

    @Override
    public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
        OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
      return new SynchronizedDelegatingOutputFormatterCallback<>(
          createPostFactoStreamCallback(out, options));
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
    public OutputFormatterCallback<Target> createPostFactoStreamCallback(
        OutputStream out, final QueryOptions options) {
      return new TextOutputFormatterCallback<Target>(out) {
        private final Set<Label> printed = CompactHashSet.create();

        private void outputRule(Rule rule, PrintStream printStream) throws InterruptedException {
          final String lineTerm = options.getLineTerminator();
          final String outputAttributePattern = "  %s = %s," + lineTerm;
          printStream.printf("# %s%s", rule.getLocation(), lineTerm);
          printStream.printf("%s(%s", rule.getRuleClass(), lineTerm);
          printStream.printf("  name = \"%s\",%s", rule.getName(), lineTerm);

          RawAttributeMapper attributeMap = RawAttributeMapper.of(rule);
          for (Attribute attr : rule.getAttributes()) {
            // Ignore the "name" attribute here, as we already print it above.
            // This is not strictly necessary, but convention has it that the
            // name attribute is printed first.
            if ("name".equals(attr.getName())) {
              continue;
            }
            if (attributeMap.isConfigurable(attr.getName())) {
              // We don't know the actual value for configurable attributes, so we reconstruct
              // the select without trying to resolve it.
              printStream.printf(
                  outputAttributePattern,
                  attr.getPublicName(),
                  outputConfigurableAttrValue(rule, attributeMap, attr));
              continue;
            }
            PossibleAttributeValues values = getPossibleAttributeValues(rule, attr);
            if (values.source != AttributeValueSource.RULE) {
              continue; // Don't print default values.
            }
            if (Iterables.size(values) != 1) {
              // Computed defaults that depend on configurable attributes can have multiple values.
              continue;
            }
            printStream.printf(
                outputAttributePattern,
                attr.getPublicName(),
                outputAttrValue(Iterables.getOnlyElement(values)));
          }
          printStream.printf(")\n%s", lineTerm);
        }

        /** Returns the given attribute value with BUILD output syntax. Does not support selects. */
        private String outputAttrValue(Object value) {
          if (value instanceof License) {
            List<String> licenseTypes = new ArrayList<>();
            for (License.LicenseType licenseType : ((License) value).getLicenseTypes()) {
              licenseTypes.add(licenseType.toString().toLowerCase());
            }
            value = licenseTypes;
          } else if (value instanceof List<?> && EvalUtils.isImmutable(value)) {
            // Display it as a list (and not as a tuple). Attributes can never be tuples.
            value = new ArrayList<>((List<?>) value);
          } else if (value instanceof TriState) {
            value = ((TriState) value).toInt();
          }
          return new LabelPrinter().repr(value).toString();
        }

        /**
         * Returns the given configurable attribute value with BUILD output syntax.
         *
         * <p>Since query doesn't know which select path should be chosen, this doesn't try to
         * resolve the final value. Instead it just reconstructs the select.
         */
        private String outputConfigurableAttrValue(
            Rule rule, RawAttributeMapper attributeMap, Attribute attr) {
          List<String> selectors = new ArrayList<>();
          for (BuildType.Selector<?> selector :
              ((BuildType.SelectorList<?>) attributeMap.getRawAttributeValue(rule, attr))
                  .getSelectors()) {
            if (selector.isUnconditional()) {
              selectors.add(
                  outputAttrValue(
                      Iterables.getOnlyElement(selector.getEntries().entrySet()).getValue()));
            } else {
              selectors.add(String.format("select(%s)", outputAttrValue(selector.getEntries())));
            }
          }
          return String.join(" + ", selectors);
        }

        @Override
        public void processOutput(Iterable<Target> partialResult) throws InterruptedException {

          for (Target target : partialResult) {
            Rule rule = target.getAssociatedRule();
            if (rule == null || printed.contains(rule.getLabel())) {
              continue;
            }
            outputRule(rule, printStream);
            printed.add(rule.getLabel());
          }
        }
      };
    }

    @Override
    public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
        OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
      return new SynchronizedDelegatingOutputFormatterCallback<>(
          createPostFactoStreamCallback(out, options));
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
      return rank + " " + label.getDefaultCanonicalForm();
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
        int rank,
        Label label,
        PrintStream out,
        @Nullable List<RankAndLabel> toSave,
        final String lineTerminator) {
      if (toSave != null) {
        toSave.add(new RankAndLabel(rank, label));
      } else {
        out.print(rank + " " + label.getDefaultCanonicalForm() + lineTerminator);
      }
    }

    @Override
    public void output(
        QueryOptions options,
        Digraph<Target> result,
        OutputStream out,
        AspectResolver aspectResolver,
        ConditionalEdges conditionalEdges)
        throws IOException {
      PrintStream printStream = new PrintStream(out);
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
      final String lineTerm = options.getLineTerminator();
      for (int rank = 0; !rankNodes.isEmpty(); rank++) {
        // Print out this rank:
        for (Node<Set<Node<Target>>> xScc : rankNodes) {
          for (Node<Target> x : xScc.getLabel()) {
            outputToStreamOrSave(
                rank, x.getLabel().getLabel(), printStream, outputToOrder, lineTerm);
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
          printStream.printf("%s%s", item, lineTerm);
        }
      }

      flushAndCheckError(printStream);
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
    public void output(
        QueryOptions options,
        Digraph<Target> result,
        OutputStream out,
        AspectResolver aspectResolver,
        ConditionalEdges conditionalEdges)
        throws IOException {
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
        Collections.sort(output, comparingInt(arg -> arg.rank));
      }
      final String lineTerm = options.getLineTerminator();
      PrintStream printStream = new PrintStream(out);
      for (RankAndLabel item : output) {
        printStream.printf("%s%s", item, lineTerm);
      }
      flushAndCheckError(printStream);
    }
  }

  /**
   * Helper class for {@link #getPossibleAttributeValues}.
   */
  static class PossibleAttributeValues implements Iterable<Object> {
    final Iterable<Object> values;
    final AttributeValueSource source;

    PossibleAttributeValues(Iterable<Object> values, AttributeValueSource source) {
      this.values = values;
      this.source = source;
    }

    @Override
    public Iterator<Object> iterator() {
      return values.iterator();
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
   * <p>This avoids the memory overruns that can happen be iterating over every possible value
   * for an <code>attr = select(...) + select(...) + select(...) + ...</code> expression. Query
   * operations generally don't care about specific attribute values - they just care which labels
   * are possible.
   */
  protected static PossibleAttributeValues getPossibleAttributeValues(Rule rule, Attribute attr)
    throws InterruptedException {
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

  private static void flushAndCheckError(PrintStream printStream) throws IOException {
    if (printStream.checkError()) {
      throw new IOException("PrintStream encountered an error");
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
  protected static String getLocation(Target target, boolean relative) {
    Location location = target.getLocation();
    return relative
        ? location.print(target.getPackage().getPackageDirectory().asFragment(),
            target.getPackage().getNameFragment())
        : location.print();
  }

  private static class LabelPrinter extends Printer.BasePrinter {
    @Override
    public LabelPrinter repr(Object o) {
      if (o instanceof Label) {
        writeString(((Label) o).getCanonicalForm());
      } else {
        super.repr(o);
      }
      return this;
    }
  }
}
