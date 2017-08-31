// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ForStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.IfStatement;
import com.google.devtools.build.lib.syntax.IfStatement.ConditionalStatements;
import com.google.devtools.build.lib.syntax.ListLiteral;
import com.google.devtools.skylark.skylint.Environment.NameInfo;
import com.google.devtools.skylark.skylint.Environment.NameInfo.Kind;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Checks that every import, private function or variable definition is used somewhere. */
public class UsageChecker extends AstVisitorWithNameResolution {
  private final List<Issue> issues = new ArrayList<>();
  private UsageInfo ui = UsageInfo.empty();
  private SetMultimap<Integer, Node> idToAllDefinitions = LinkedHashMultimap.create();

  private UsageChecker(Environment env) {
    super(env);
  }

  public static List<Issue> check(BuildFileAST ast) {
    UsageChecker checker = new UsageChecker(Environment.defaultBazel());
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(Identifier node) {
    super.visit(node);
    NameInfo info = env.resolveName(node.getName());
    // TODO(skylark-team): Don't ignore unresolved symbols in the future but report an error
    if (info != null) {
      ui.usedDefinitions.addAll(ui.idToLastDefinitions.get(info.id));
    }
  }

  @Override
  protected void visitLvalue(Expression expr) {
    if (expr instanceof Identifier) {
      super.visit((Identifier) expr); // don't call this.visit because it doesn't count as usage
    } else if (expr instanceof ListLiteral) {
      for (Expression e : ((ListLiteral) expr).getElements()) {
        visitLvalue(e);
      }
    } else {
      visit(expr);
    }
  }

  @Override
  public void visit(IfStatement node) {
    UsageInfo input = ui;
    List<UsageInfo> outputs = new ArrayList<>();
    for (ConditionalStatements clause : node.getThenBlocks()) {
      ui = input.copy();
      visit(clause);
      outputs.add(ui);
    }
    ui = input.copy();
    visitBlock(node.getElseBlock());
    outputs.add(ui);
    ui = UsageInfo.join(outputs);
  }

  @Override
  public void visit(ForStatement node) {
    visit(node.getCollection());
    visit(node.getVariable());
    UsageInfo noIteration = ui.copy();
    visitBlock(node.getBlock());
    UsageInfo oneIteration = ui.copy();
    // We need to visit the block again in case a variable was reassigned in the last iteration and
    // the new value isn't used until the next iteration
    visit(node.getVariable());
    visitBlock(node.getBlock());
    UsageInfo manyIterations = ui.copy();
    ui = UsageInfo.join(Arrays.asList(noIteration, oneIteration, manyIterations));
  }

  @Override
  protected void declare(String name, ASTNode node) {
    int id = env.resolveExistingName(name).id;
    ui.idToLastDefinitions.removeAll(id);
    ui.idToLastDefinitions.put(id, new Node(node));
    idToAllDefinitions.put(id, new Node(node));
  }

  @Override
  protected void reassign(String name, Identifier ident) {
    declare(name, ident);
  }

  @Override
  public void exitBlock() {
    Collection<Integer> ids = env.getNameIdsInCurrentBlock();
    for (Integer id : ids) {
      checkUsed(id);
    }
    super.exitBlock();
  }

  private void checkUsed(Integer id) {
    Set<Node> unusedDefinitions = idToAllDefinitions.get(id);
    unusedDefinitions.removeAll(ui.usedDefinitions);
    NameInfo nameInfo = env.getNameInfo(id);
    String name = nameInfo.name;
    if ("_".equals(name) || nameInfo.kind == Kind.BUILTIN) {
      return;
    }
    if ((nameInfo.kind == Kind.LOCAL || nameInfo.kind == Kind.PARAMETER) && name.startsWith("_")) {
      // local variables starting with an underscore need not be used
      return;
    }
    if (nameInfo.kind == Kind.GLOBAL && !name.startsWith("_")) {
      // symbol might be loaded in another file
      return;
    }
    String message = "unused definition of '" + name + "'";
    if (nameInfo.kind == Kind.PARAMETER) {
      message +=
          ". If this is intentional, "
              + "you can add `_ignore = [<param1>, <param2>, ...]` to the function body.";
    } else if (nameInfo.kind == Kind.LOCAL) {
      message += ". If this is intentional, you can use '_' or rename it to '_" + name + "'.";
    }
    for (Node definition : unusedDefinitions) {
      issues.add(new Issue(message, definition.node.getLocation()));
    }
  }

  private static class UsageInfo {
    /**
     * Stores for each variable ID the definitions that are "live", i.e. are the most recent ones on
     * some execution path.
     *
     * <p>There can be more than one last definition if branches are involved, e.g. if foo: x = 1;
     * else x = 2;
     */
    private final SetMultimap<Integer, Node> idToLastDefinitions;
    /** Set of definitions that have already been used at some point. */
    private final Set<Node> usedDefinitions;

    private UsageInfo(SetMultimap<Integer, Node> idToLastDefinitions, Set<Node> usedDefinitions) {
      this.idToLastDefinitions = idToLastDefinitions;
      this.usedDefinitions = usedDefinitions;
    }

    static UsageInfo empty() {
      return new UsageInfo(LinkedHashMultimap.create(), new HashSet<>());
    }

    UsageInfo copy() {
      return new UsageInfo(
          LinkedHashMultimap.create(idToLastDefinitions), new HashSet<>(usedDefinitions));
    }

    static UsageInfo join(Collection<UsageInfo> uis) {
      UsageInfo result = UsageInfo.empty();
      for (UsageInfo ui : uis) {
        result.idToLastDefinitions.putAll(ui.idToLastDefinitions);
        result.usedDefinitions.addAll(ui.usedDefinitions);
      }
      return result;
    }
  }

  /** Wrapper for ASTNode that can be put in a HashSet. */
  private static class Node {
    ASTNode node;

    public Node(ASTNode node) {
      this.node = node;
    }

    @Override
    public boolean equals(@Nullable Object other) {
      return other instanceof Node && ((Node) other).node == node;
    }

    @Override
    public int hashCode() {
      return System.identityHashCode(node);
    }
  }
}
