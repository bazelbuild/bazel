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

import com.google.common.base.Equivalence;
import com.google.common.base.Equivalence.Wrapper;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.syntax.ASTNode;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ExpressionStatement;
import com.google.devtools.build.lib.syntax.FlowStatement;
import com.google.devtools.build.lib.syntax.ForStatement;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.IfStatement;
import com.google.devtools.build.lib.syntax.IfStatement.ConditionalStatements;
import com.google.devtools.build.lib.syntax.ReturnStatement;
import com.google.devtools.skylark.skylint.Environment.NameInfo;
import com.google.devtools.skylark.skylint.Environment.NameInfo.Kind;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** Checks that every import, private function or variable definition is used somewhere. */
public class UsageChecker extends AstVisitorWithNameResolution {
  private static final String UNUSED_BINDING_CATEGORY = "unused-binding";
  private static final String UNINITIALIZED_VARIABLE_CATEGORY = "uninitialized-variable";

  private final List<Issue> issues = new ArrayList<>();
  private UsageInfo ui = UsageInfo.empty();
  private final SetMultimap<Integer, Wrapper<ASTNode>> idToAllDefinitions =
      LinkedHashMultimap.create();
  private final Set<Wrapper<ASTNode>> initializationsWithNone = new LinkedHashSet<>();

  public static List<Issue> check(BuildFileAST ast) {
    UsageChecker checker = new UsageChecker();
    checker.visit(ast);
    return checker.issues;
  }

  @Override
  public void visit(FunctionDefStatement node) {
    UsageInfo saved = ui.copy();
    super.visit(node);
    ui = UsageInfo.join(Arrays.asList(saved, ui));
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
  public void visit(FlowStatement node) {
    ui.reachable = false;
  }

  @Override
  public void visit(ReturnStatement node) {
    super.visit(node);
    ui.reachable = false;
  }

  @Override
  public void visit(ExpressionStatement node) {
    super.visit(node);
    if (ControlFlowChecker.isFail(node.getExpression())) {
      ui.reachable = false;
    }
  }

  @Override
  public void visit(AssignmentStatement node) {
    super.visit(node);
    /* If a variable is initialized with None, and there exist other assignments to the variable,
     * then this initialization is itself considered as a usage. This is because it's good practice
     * to place a "declaration" of a variable in a location that dominates all its uses, especially
     * so if you want to document the variable. Example:
     *
     *     var = None  # don't warn about the unused binding
     *     if condition:
     *       var = 0
     *     else:
     *       var = 1
     *
     * Unfortunately, as a side-effect, the following won't trigger a warning either:
     *
     *     var = None  # doesn't warn either but ideally should
     *     var = 0
     */
    Expression lhs = node.getLValue().getExpression();
    Expression rhs = node.getExpression();
    if (lhs instanceof Identifier
        && rhs instanceof Identifier
        && ((Identifier) rhs).getName().equals("None")) {
      NameInfo info = env.resolveName(((Identifier) lhs).getName());
      // if it's an initialization:
      if (info != null && idToAllDefinitions.get(info.id).size() == 1) {
        initializationsWithNone.add(wrapNode(lhs));
      }
    }
  }

  private void defineIdentifier(NameInfo name, ASTNode node) {
    ui.idToLastDefinitions.removeAll(name.id);
    ui.idToLastDefinitions.put(name.id, wrapNode(node));
    ui.initializedIdentifiers.add(name.id);
    idToAllDefinitions.put(name.id, wrapNode(node));
  }

  @Override
  protected void use(Identifier identifier) {
    NameInfo info = env.resolveName(identifier.getName());
    // TODO(skylark-team): Don't ignore unresolved symbols in the future but report an error
    if (info != null) {
      ui.usedDefinitions.addAll(ui.idToLastDefinitions.get(info.id));
      checkInitialized(info, identifier);
    }
  }

  @Override
  protected void declare(String name, ASTNode node) {
    NameInfo info = env.resolveExistingName(name);
    defineIdentifier(info, node);
  }

  @Override
  protected void reassign(Identifier ident) {
    declare(ident.getName(), ident);
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
    Set<Wrapper<ASTNode>> unusedDefinitions = new LinkedHashSet<>(idToAllDefinitions.get(id));
    unusedDefinitions.removeAll(ui.usedDefinitions);
    NameInfo nameInfo = env.getNameInfo(id);
    String name = nameInfo.name;
    if ("_".equals(name) || nameInfo.kind == Kind.BUILTIN) {
      return;
    }
    if ((nameInfo.kind == Kind.LOCAL || nameInfo.kind == Kind.PARAMETER)
        && (name.startsWith("_") || name.startsWith("unused_") || name.startsWith("UNUSED_"))) {
      // local variables starting with an underscore need not be used
      return;
    }
    if ((nameInfo.kind == Kind.GLOBAL || nameInfo.kind == Kind.FUNCTION) && !name.startsWith("_")) {
      // symbol might be loaded in another file
      return;
    }
    String message = "unused binding of '" + name + "'";
    if (nameInfo.kind == Kind.IMPORTED && !nameInfo.name.startsWith("_")) {
      message +=
          ". If you want to re-export a symbol, use the following pattern:\n"
              + "\n"
              + "load(..., _"
              + name
              + " = '"
              + name
              + "', ...)\n"
              + name
              + " = _"
              + name
              + "\n"
              + "\n"
              + "More details in the documentation.";
    } else if (nameInfo.kind == Kind.PARAMETER) {
      message +=
          ". If this is intentional, "
              + "you can add `_ignore = [<param1>, <param2>, ...]` to the function body.";
    } else if (nameInfo.kind == Kind.LOCAL) {
      message += ". If this is intentional, you can use '_' or rename it to '_" + name + "'.";
    }
    for (Wrapper<ASTNode> definition : unusedDefinitions) {
      if (initializationsWithNone.contains(definition) && idToAllDefinitions.get(id).size() > 1) {
        // initializations with None are OK, cf. visit(AssignmentStatement) above
        continue;
      }
      issues.add(
          Issue.create(UNUSED_BINDING_CATEGORY, message, unwrapNode(definition).getLocation()));
    }
  }

  private void checkInitialized(NameInfo info, Identifier node) {
    if (ui.reachable && !ui.initializedIdentifiers.contains(info.id) && info.kind != Kind.BUILTIN) {
      issues.add(
          Issue.create(
              UNINITIALIZED_VARIABLE_CATEGORY,
              "variable '" + info.name + "' may not have been initialized",
              node.getLocation()));
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
    private final SetMultimap<Integer, Wrapper<ASTNode>> idToLastDefinitions;
    /** Set of definitions that have already been used at some point. */
    private final Set<Wrapper<ASTNode>> usedDefinitions;
    /** Set of variable IDs that are initialized. */
    private final Set<Integer> initializedIdentifiers;
    /**
     * Whether the current position in the program is reachable.
     *
     * <p>This is needed to correctly compute initialized variables.
     */
    private boolean reachable;

    private UsageInfo(
        SetMultimap<Integer, Wrapper<ASTNode>> idToLastDefinitions,
        Set<Wrapper<ASTNode>> usedDefinitions,
        Set<Integer> initializedIdentifiers,
        boolean reachable) {
      this.idToLastDefinitions = idToLastDefinitions;
      this.usedDefinitions = usedDefinitions;
      this.initializedIdentifiers = initializedIdentifiers;
      this.reachable = reachable;
    }

    static UsageInfo empty() {
      return new UsageInfo(
          LinkedHashMultimap.create(), new LinkedHashSet<>(), new LinkedHashSet<>(), true);
    }

    UsageInfo copy() {
      return new UsageInfo(
          LinkedHashMultimap.create(idToLastDefinitions),
          new LinkedHashSet<>(usedDefinitions),
          new LinkedHashSet<>(initializedIdentifiers),
          reachable);
    }

    static UsageInfo join(Collection<UsageInfo> uis) {
      Set<Integer> initializedInRelevantBranch = new LinkedHashSet<>();
      for (UsageInfo ui : uis) {
        if (ui.reachable) {
          initializedInRelevantBranch = ui.initializedIdentifiers;
          break;
        }
      }
      UsageInfo result =
          new UsageInfo(
              LinkedHashMultimap.create(),
              new LinkedHashSet<>(),
              initializedInRelevantBranch,
              false);
      for (UsageInfo ui : uis) {
        result.idToLastDefinitions.putAll(ui.idToLastDefinitions);
        result.usedDefinitions.addAll(ui.usedDefinitions);
        if (ui.reachable) {
          // Only a non-diverging branch can affect the set of initialized variables.
          result.initializedIdentifiers.retainAll(ui.initializedIdentifiers);
        }
        result.reachable |= ui.reachable;
      }
      return result;
    }
  }

  private Wrapper<ASTNode> wrapNode(ASTNode node) {
    return Equivalence.identity().wrap(node);
  }

  private ASTNode unwrapNode(Wrapper<ASTNode> wrapper) {
    return wrapper.get();
  }
}
