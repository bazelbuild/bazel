// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax.compiler;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.ASTNode;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodInvocation;

import java.util.ArrayList;
import java.util.List;

/**
 * A single point of access for references to the AST for debug location purposes.
 */
public final class DebugInfo {

  /**
   * Contains byte code instructions for calls to a DebugInfo instances methods.
   */
  public static final class AstAccessors {
    public final StackManipulation loadAstNode;
    public final StackManipulation loadLocation;

    private AstAccessors(int index, StackManipulation getAstNode, StackManipulation getLocation) {
      StackManipulation indexValue = IntegerConstant.forValue(index);
      this.loadAstNode = new StackManipulation.Compound(indexValue, getAstNode);
      this.loadLocation = new StackManipulation.Compound(indexValue, getLocation);
    }
  }

  private final List<ASTNode> astNodes;
  private final StackManipulation getAstNode;
  private final StackManipulation getLocation;

  /**
   * @param getAstNode A {@link MethodDescription} which can be used to access this instance's
   *     {@link #getAstNode(int)} in a static way.
   * @param getLocation A {@link MethodDescription} which can be used to access this instance's
   *     {@link #getLocation(int)} in a static way.
   */
  public DebugInfo(MethodDescription getAstNode, MethodDescription getLocation) {
    astNodes = new ArrayList<>();
    this.getAstNode = MethodInvocation.invoke(getAstNode);
    this.getLocation = MethodInvocation.invoke(getLocation);
  }

  /**
   * Get an {@link ASTNode} for reference at runtime.
   *
   * <p>Needed for rule construction which refers back to the function call node to get argument
   * locations.
   */
  public ASTNode getAstNode(int index) {
    return astNodes.get(index);
  }

  /**
   * Get a {@link Location} for reference at runtime.
   *
   * <p>Needed to provide source code error locations at runtime.
   */
  public Location getLocation(int index) {
    return getAstNode(index).getLocation();
  }

  /**
   * Use this during compilation to add AST nodes needed at runtime.
   * @return an {@link AstAccessors} instance which can be used to get the info at runtime in the
   * static context of the byte code
   */
  public AstAccessors add(ASTNode node) {
    astNodes.add(node);
    return new AstAccessors(astNodes.size() - 1, getAstNode, getLocation);
  }
}
