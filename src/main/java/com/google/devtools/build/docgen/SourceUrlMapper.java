// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import java.io.File;
import java.nio.file.Path;
import java.util.Map;

/**
 * Given a Build Encyclopedia generator tool's input filename or label, the mapper produces the URL
 * for the corresponding source file in the source code repository.
 */
class SourceUrlMapper {
  // TODO(arostovtsev): eventually, we will want to support multiple URL roots (e.g. for
  // language-specific rules).
  private final String sourceUrlRoot;
  private final Path inputRoot;
  private final ImmutableMap<String, String> labelRewrites;

  /**
   * @param sourceUrlRoot root URL for the source code repository
   * @param inputRoot input directory corresponding to the source tree root
   * @param labelRewrites an ordered map of label string prefixes; a map entry of the form "foo" ->
   *     "bar" indicates that if a label string starts with "foo", that prefix should be replaced
   *     with "bar". The intended use case is transforming labels of .bzl files in @_builtins, whose
   *     corresponding source files live elsewhere in the repo.
   */
  SourceUrlMapper(
      String sourceUrlRoot, String inputRoot, ImmutableMap<String, String> labelRewrites) {
    this.sourceUrlRoot = sourceUrlRoot;
    this.inputRoot = new File(inputRoot).toPath();
    this.labelRewrites = labelRewrites;
  }

  SourceUrlMapper(DocLinkMap linkMap, String inputRoot) {
    this(linkMap.sourceUrlRoot, inputRoot, ImmutableMap.copyOf(linkMap.labelRewrites));
  }

  /**
   * Returns the source code repository URL of a Java source file which was passed as an input to
   * the Build Encyclopedia generator tool.
   *
   * @throws InvalidArgumentException if the URL could not be produced.
   */
  String urlOfFile(File file) {
    Path path = file.toPath();
    Preconditions.checkArgument(
        path.startsWith(inputRoot), "File '%s' is expected to be under '%s'", path, inputRoot);
    return sourceUrlRoot + inputRoot.toUri().relativize(path.toUri());
  }

  /**
   * Returns the source code repository URL of a .bzl file label which was passed as an input to the
   * Build Encyclopedia generator tool.
   *
   * <p>A label is first rewritten via {@link labelRewrites}: an entry of the form "foo" -> "bar"
   * means that if {@code labelString} starts with "foo", the "foo" prefix is replaced with "bar".
   * Rewrite rules in {@link labelRewrites} are examined in order, and only the first matching
   * rewrite is applied.
   *
   * <p>If the result is a label in the main repo, the (possibly rewritten) label is transformed
   * into a URL.
   *
   * @throws InvalidArgumentException if the URL could not be produced.
   */
  String urlOfLabel(String labelString) {
    String originalLabelString = labelString;
    for (Map.Entry<String, String> entry : labelRewrites.entrySet()) {
      if (labelString.startsWith(entry.getKey())) {
        labelString = entry.getValue() + labelString.substring(entry.getKey().length());
        break;
      }
    }
    Label label;
    try {
      label = Label.parseCanonical(labelString);
    } catch (LabelSyntaxException e) {
      String message = String.format("Failed to parse label '%s'", labelString);
      if (!labelString.equals(originalLabelString)) {
        message = String.format("%s (rewritten; originally '%s')", message, originalLabelString);
      }
      throw new IllegalArgumentException(message, e);
    }
    Preconditions.checkArgument(
        label.getRepository().isMain(), "Label '%s' is not in the main repository", labelString);

    return sourceUrlRoot + label.toPathFragment().getPathString();
  }
}
