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

import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.PackageContext;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Helper class encapsulating string scanning state used during "heuristic" expansion of labels
 * embedded within rules.
 */
public final class LabelExpander {
  /**
   * An exception that is thrown when a label is expanded to zero or multiple files during
   * expansion.
   */
  public static class NotUniqueExpansionException extends Exception {
    public NotUniqueExpansionException(int sizeOfResultSet, String labelText) {
      super(
          "heuristic label expansion found '"
              + labelText
              + "', which expands to "
              + sizeOfResultSet
              + " files"
              + (sizeOfResultSet > 1 ? ", please use $(locations " + labelText + ") instead" : ""));
    }
  }

  // This is a utility class, no need to instantiate.
  private LabelExpander() {}

  /**
   * CharMatcher to determine if a given character is valid for labels.
   *
   * <p>The Build Concept Reference additionally allows '=' and ',' to appear in labels, but for the
   * purposes of the heuristic, this function does not, as it would cause "--foo=:rule1,:rule2" to
   * scan as a single possible label, instead of three ("--foo", ":rule1", ":rule2").
   */
  private static final CharMatcher LABEL_CHAR_MATCHER =
      CharMatcher.inRange('a', 'z')
          .or(CharMatcher.inRange('A', 'Z'))
          .or(CharMatcher.inRange('0', '9'))
          .or(CharMatcher.anyOf(":/_.-+" + PathFragment.SEPARATOR_CHAR))
          .precomputed();

  /**
   * Expands all references to labels embedded within a string using the provided expansion mapping
   * from labels to artifacts.
   *
   * <p>Since this pass is heuristic, references to non-existent labels (such as arbitrary words) or
   * invalid labels are simply ignored and are unchanged in the output. However, if the heuristic
   * discovers a label, which identifies an existing target producing zero or multiple files, an
   * error is reported.
   *
   * @param expression the expression to expand.
   * @param labelMap the mapping from labels to artifacts, whose relative path is to be used as the
   *     expansion.
   * @param labelResolver the {@code Label} that can resolve label strings to {@code Label} objects.
   *     The resolved label is either relative to {@code labelResolver} or is a global label (i.e.
   *     starts with "//").
   * @return the expansion of the string.
   * @throws NotUniqueExpansionException if a label that is present in the mapping expands to zero
   *     or multiple files.
   */
  public static <T extends Iterable<Artifact>> String expand(
      @Nullable String expression, Map<Label, T> labelMap, Label labelResolver)
      throws NotUniqueExpansionException {
    if (Strings.isNullOrEmpty(expression)) {
      return "";
    }
    Preconditions.checkNotNull(labelMap);
    Preconditions.checkNotNull(labelResolver);

    int offset = 0;
    StringBuilder result = new StringBuilder();
    while (offset < expression.length()) {
      String labelText = scanLabel(expression, offset);
      if (labelText != null) {
        offset += labelText.length();
        result.append(tryResolvingLabelTextToArtifactPath(labelText, labelMap, labelResolver));
      } else {
        result.append(expression.charAt(offset));
        offset++;
      }
    }
    return result.toString();
  }

  /**
   * Tries resolving a label text to a full label for the associated {@code Artifact}, using the
   * provided mapping.
   *
   * <p>The method succeeds if the label text can be resolved to a {@code Label} object, which is
   * present in the {@code labelMap} and maps to exactly one {@code Artifact}.
   *
   * @param labelText the text to resolve.
   * @param labelMap the mapping from labels to artifacts, whose relative path is to be used as the
   *     expansion.
   * @param labelResolver the {@code Label} that can resolve label strings to {@code Label} objects.
   *     The resolved label is either relative to {@code labelResolver} or is a global label (i.e.
   *     starts with "//").
   * @return an absolute label to an {@code Artifact} if the resolving was successful or the
   *     original label text.
   * @throws NotUniqueExpansionException if a label that is present in the mapping expands to zero
   *     or multiple files.
   */
  private static <T extends Iterable<Artifact>> String tryResolvingLabelTextToArtifactPath(
      String labelText, Map<Label, T> labelMap, Label labelResolver)
      throws NotUniqueExpansionException {
    Label resolvedLabel = resolveLabelText(labelText, labelResolver);
    if (resolvedLabel == null) {
      return labelText;
    }
    Iterable<Artifact> artifacts = labelMap.get(resolvedLabel);
    if (artifacts == null) {
      return labelText;
    }
    // resolvedLabel identifies an existing target
    List<String> locations = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      if (!artifact.isRunfilesTree()) {
        locations.add(artifact.getExecPathString());
      }
    }
    int resultSetSize = locations.size();
    if (resultSetSize == 1) {
      return Iterables.getOnlyElement(locations); // success!
    } else {
      throw new NotUniqueExpansionException(resultSetSize, labelText);
    }
  }

  /**
   * Resolves a string to a label text. Uses {@code labelResolver} to do so. The result is either
   * relative to {@code labelResolver} or is an absolute label. In case of an invalid label text,
   * the return value is null.
   */
  @Nullable
  private static Label resolveLabelText(String labelText, Label labelResolver) {
    try {
      return Label.parseWithPackageContext(
          labelText,
          PackageContext.of(labelResolver.getPackageIdentifier(), RepositoryMapping.EMPTY));
    } catch (LabelSyntaxException e) {
      // It's a heuristic, so quietly ignore "errors".
      return null;
    }
  }

  /**
   * Scans the argument string from a given start position until the name of a potential label has
   * been consumed, then returns the label text. If the expression contains no possible label
   * starting at the start position, the return value is null.
   */
  @Nullable
  private static String scanLabel(String expression, int start) {
    int offset = start;
    while (offset < expression.length() && LABEL_CHAR_MATCHER.matches(expression.charAt(offset))) {
      ++offset;
    }
    if (offset > start) {
      return expression.substring(start, offset);
    } else {
      return null;
    }
  }
}
