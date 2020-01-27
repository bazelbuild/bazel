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
// All Rights Reserved.

package com.google.devtools.build.lib.graph;

/**
 *  <p> A stub implementation of GraphVisitor providing default behaviour (do
 *  nothing) for all its methods. </p>
 */
public class AbstractGraphVisitor<T> implements GraphVisitor<T> {
  @Override
  public void beginVisit() {}
  @Override
  public void endVisit() {}
  @Override
  public void visitEdge(Node<T> lhs, Node<T> rhs) {}
  @Override
  public void visitNode(Node<T> node) {}
}
