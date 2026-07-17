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
package com.google.devtools.build.lib.rules.java;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** Utility methods for use by Java-related parts of Bazel. */
// TODO(bazel-team): Merge with JavaUtil.
public abstract class JavaHelper {

  private JavaHelper() {}

  /**
   * Flattens a set of javacopts and tokenizes the contents.
   *
   * <p>Since javac allows passing the same option multiple times, and the right-most one wins,
   * multiple instances of the same option+value get de-duped by the NestedSet. So when combining
   * multiple NestedSets, we store them in reverse order, and reverse again after flattening. This
   * preserves the right-most occurrence in its correct position, thus achieving the correct
   * semantics.
   *
   * @param inOpts the set of opts to tokenize
   */
  public static ImmutableList<String> tokenizeJavaOptions(NestedSet<String> inOpts) {
    return tokenizeJavaOptions(inOpts.toList().reverse());
  }

  /**
   * Javac options require special processing - People use them and expect the options to be
   * tokenized.
   */
  public static ImmutableList<String> tokenizeJavaOptions(Iterable<String> inOpts) {
    // Ideally, this would be in the options parser. Unfortunately,
    // the options parser can't handle a converter that expands
    // from a value X into a List<X> and allow-multiple at the
    // same time.
    List<String> result = new ArrayList<>();
    for (String current : inOpts) {
      try {
        ShellUtils.tokenize(result, current);
      } catch (ShellUtils.TokenizationException ex) {
        // Tokenization failed; this likely means that the user
        // did not want tokenization to happen on their argument.
        // (Any tokenization where we should produce an error
        // has already been done by the shell that invoked
        // blaze). Therefore, pass the argument through to
        // the tool, so that we can see the original error.
        result.add(current);
      }
    }
    return ImmutableList.copyOf(result);
  }

  /**
   * De-tokenizes a collection of {@code javac} options into a {@link NestedSet}.
   *
   * <p>Each option is shell-escaped to get back the original option as-is when we tokenize the
   * depset.
   *
   * @param javacOpts the {@code javac} options to detokenize
   * @return A {@link NestedSet} of the supplied options concatenated into a single string separated
   *     by ' '.
   */
  public static NestedSet<String> detokenizeJavaOptions(Collection<String> javacOpts) {
    if (javacOpts.isEmpty()) {
      return NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    return NestedSetBuilder.create(
        Order.NAIVE_LINK_ORDER,
        javacOpts.stream().map(ShellUtils::shellEscape).collect(joining(" ")));
  }

  public static PathFragment getJavaResourcePath(
      JavaSemantics semantics, RuleContext ruleContext, Artifact resource) {
    boolean siblingRepositoryLayout = ruleContext.getConfiguration().isSiblingRepositoryLayout();
    PathFragment resourcePath = resource.getOutputDirRelativePath(siblingRepositoryLayout);
    PathFragment repoExecPath =
        ruleContext
            .getLabel()
            .getRepository()
            .getExecPath(siblingRepositoryLayout);
    if (!repoExecPath.isEmpty() && resourcePath.startsWith(repoExecPath)) {
      resourcePath = resourcePath.relativeTo(repoExecPath);
    }

    if (!ruleContext.attributes().has("resource_strip_prefix", Type.STRING)
        || !ruleContext.attributes().isAttributeValueExplicitlySpecified("resource_strip_prefix")) {
      return semantics.getDefaultJavaResourcePath(resourcePath);
    }

    PathFragment prefix =
        PathFragment.create(ruleContext.attributes().get("resource_strip_prefix", Type.STRING));

    if (!resourcePath.startsWith(prefix)) {
      ruleContext.attributeError(
          "resource_strip_prefix",
          String.format(
              "Resource file '%s' is not under the specified prefix to strip", resourcePath));
      return resourcePath;
    }

    return resourcePath.relativeTo(prefix);
  }
}
