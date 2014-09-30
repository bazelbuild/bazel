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
// All Rights Reserved.

package com.google.devtools.build.lib.graph;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *  <p> Utility methods for manipulating AT&amp;T "dot" (GraphViz) format  files.
 *  (Actually, a simple subset.)</p>
 */
public abstract class Dot {

  private Dot() {} // uninstantiable.

  private final static String QUOTED_LABEL = "\"([^\";]+)\"";
  private final static String NON_QUOTED_LABEL = "([^\"\\s;]+)";

  /**
   *  Pattern for edge specifiers in dot files.
   */
  private static Pattern edgePattern = Pattern.compile(
      "(?:^\\s*" + NON_QUOTED_LABEL + "\\s*->\\s*" + NON_QUOTED_LABEL + "\\s*.*;?+)|" +
      "(?:^\\s*" + QUOTED_LABEL     + "\\s*->\\s*" + NON_QUOTED_LABEL + "\\s*.*;?+)|" +
      "(?:^\\s*" + NON_QUOTED_LABEL + "\\s*->\\s*" + QUOTED_LABEL     + "\\s*.*;?+)|" +
      "(?:^\\s*" + QUOTED_LABEL     + "\\s*->\\s*" + QUOTED_LABEL     + "\\s*.*;?+)");

  /**
   *  Pattern for node specifiers in dot files.
   */
  private static Pattern nodePattern =
    Pattern.compile("(?:^\\s*" + QUOTED_LABEL     + "\\s*.*;?+)|" +
                    "(?:^\\s*" + NON_QUOTED_LABEL + "\\s*.*;?+)");

  static final class EdgeLabel<T> {
    final T from;
    final T to;
    EdgeLabel(T from, T to) {
      this.from = from;
      this.to = to;
    }
  }

  /**
   * Extracts the "from" and "to" labels from a dot file line.
   * @param s the line from which to extract the from an to labels
   * @param deserializer the label deserializer to use
   * @return a pair with from (first) and to (second) or <code>null</code>
   * if no pair was extracted
   */
  static <T> EdgeLabel<T> extractFromAndToLabels(String s,
        LabelDeserializer<T> deserializer) throws DotSyntaxException {
    Matcher edgeMatcher = edgePattern.matcher(s); // "from" -> "to"
    if (edgeMatcher.matches()) {
      String fromLabel = edgeMatcher.group(1);
      String toLabel = edgeMatcher.group(2);
      if (fromLabel == null && toLabel == null) {
        fromLabel = edgeMatcher.group(3);
        toLabel = edgeMatcher.group(4);
      }
      if (fromLabel == null && toLabel == null) {
        fromLabel = edgeMatcher.group(5);
        toLabel = edgeMatcher.group(6);
      }
      if (fromLabel == null && toLabel == null) {
        fromLabel = edgeMatcher.group(7);
        toLabel = edgeMatcher.group(8);
      }
      return new EdgeLabel<T>(deserializer.deserialize(fromLabel),
          deserializer.deserialize(toLabel));
    } else {
      return null;
    }
  }

  /**
   * Extracts the node label from a dot file line.
   * @param s the line from which to extract the node label
   * @param deserializer the label deserializer to use
   * @return a label or <code>null</code> if none was found
   */
  protected static <T> T extractNodeLabel(String s,
        LabelDeserializer<T> deserializer) throws DotSyntaxException {
    Matcher nodeMatcher = nodePattern.matcher(s); // "node"
    if (nodeMatcher.matches()) {
      String nodeLabel = nodeMatcher.group(1);
      if (nodeLabel != null) {
        return deserializer.deserialize(nodeLabel);
      } else {
        return deserializer.deserialize(nodeMatcher.group(2));
      }
    } else {
      return null;
    }
  }

  /**
   *  The default implementation of LabelSerializer simply serializes
   *  each node using its toString method.
   */
  public static class DefaultLabelSerializer<T>
      implements LabelSerializer<T> {
    @Override
    public String serialize(Node<T> node) {
      return node.getLabel().toString();
    }
  }

  /**
   *  The default implementation of LabelDeserializer simply uses the
   *  strings in the dot file representation as the graph node labels.
   */
  public static class DefaultLabelDeserializer
      implements LabelDeserializer<String>
  {
    @Override
    public String deserialize(String rep) { return rep; }
  }
}
