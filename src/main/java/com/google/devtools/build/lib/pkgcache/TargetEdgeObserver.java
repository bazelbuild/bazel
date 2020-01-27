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

package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import javax.annotation.Nullable;

/**
 * An observer of the visitation over a target graph.
 */
public interface TargetEdgeObserver {

  /**
   * Called when an edge is discovered.
   * May be called more than once for the same
   * (from, to) pair.
   *
   * @param from the originating node.
   * @param attribute The attribute which defines the edge.
   *     Non-null iff (from instanceof Rule).
   * @param to the target node.
   */
  void edge(Target from, Attribute attribute, Target to);

  /**
   * Called when a Target has a reference to a non-existent target.
   *
   * @param target the target. May be null (e.g. in the case of an implicit dependency on a
   *     subincluded file).
   * @param to a label reference in the rule, which does not correspond to a valid target.
   * @param e the corresponding exception thrown
   */
  void missingEdge(@Nullable Target target, Label to, NoSuchThingException e);

  /**
   * Called when a node is discovered. May be called
   * more than once for the same node.
   *
   * @param node the target.
   */
  void node(Target node);
}
