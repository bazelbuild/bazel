// Copyright 2022 The Bazel Authors. All rights reserved.
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

package net.starlark.java.syntax;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javax.annotation.Nullable;

public interface CoverageRecorder {

  void register(Program program);

  void recordCoverage(Node node);

  void recordVirtualJump(Node node);

  void dump(PrintWriter out);

  static CoverageRecorder getInstance() {
    return CoverageRecorderHolder.INSTANCE;
  }

  /**
   * Collect coverage for all {@link Program}s compiled after the call whose
   * {@link Program#getFilename()} matches {@code filenameMatcher}.
   */
  static void startCoverageCollection(Function<String, Boolean> filenameMatcher) {
    CoverageRecorderHolder.INSTANCE = new LcovCoverageRecorder(filenameMatcher);
  }

  class CoverageRecorderHolder {

    private static CoverageRecorder INSTANCE = new NoopCoverageRecorder();

    private CoverageRecorderHolder() {
    }
  }
}

final class NoopCoverageRecorder implements CoverageRecorder {

  @Override
  public void register(Program program) {
  }

  @Override
  public void recordCoverage(Node node) {
  }

  @Override
  public void recordVirtualJump(Node node) {
  }

  @Override
  public void dump(PrintWriter out) {
  }
}

/**
 * A {@link CoverageRecorder} that records function, line, and branch coverage for all Starlark
 * {@link Program}s matching the provided {@code filenameMatcher}. Calling
 * {@link LcovCoverageRecorder#dump(PrintWriter)} emits LCOV records for all matched files.
 */
final class LcovCoverageRecorder implements CoverageRecorder {

  private final Function<String, Boolean> filenameMatcher;

  /**
   * Tracks the number of times a given {@link Node} has been executed.
   */
  private final ConcurrentHashMap<Node, LongAdder> counts = new ConcurrentHashMap<>();

  /**
   * Tracks the number of times a conditional jump without a syntax tree representation has been
   * executed which is associated with the given {@link Node}. Examples: - The "condition not
   * satisfied" jump of an {@code if} without an {@code else}. - The "short-circuit" jump of an
   * {@code and} or {@code or}.
   */
  private final ConcurrentHashMap<Node, LongAdder> virtualJumpCounts = new ConcurrentHashMap<>();

  private final Set<Program> registeredPrograms = ConcurrentHashMap.newKeySet();

  LcovCoverageRecorder(Function<String, Boolean> filenameMatcher) {
    this.filenameMatcher = filenameMatcher;
  }

  @Override
  public void register(Program program) {
    if (!filenameMatcher.apply(program.getFilename())) {
      return;
    }
    registeredPrograms.add(program);
    for (Statement statement : program.getResolvedFunction().getBody()) {
      statement.accept(new CoverageVisitor() {
        @Override
        protected void visitFunction(String identifier, Node defStatement,
            Node firstBodyStatement) {
        }

        @Override
        protected void visitBranch(Node owner, Node condition, Node positiveUniqueSuccessor,
            @Nullable Node negativeUniqueSuccessor) {
          if (negativeUniqueSuccessor == null) {
            virtualJumpCounts.put(owner, new LongAdder());
          }
          // positiveUniqueSuccessor will be registered via a call to visitCode.
        }

        @Override
        protected void visitCode(Node node) {
          counts.put(node, new LongAdder());
        }
      });
    }
  }

  @Override
  public void recordCoverage(Node node) {
    LongAdder counter = counts.get(node);
    if (counter == null) {
      return;
    }
    counter.increment();
  }

  @Override
  public void recordVirtualJump(Node node) {
    LongAdder counter = virtualJumpCounts.get(node);
    if (counter == null) {
      return;
    }
    counter.increment();
  }

  @Override
  public void dump(PrintWriter out) {
    registeredPrograms.stream()
        .sorted(Comparator.comparing(Program::getFilename))
        .forEachOrdered(program -> {
          CoverageNodeVisitor visitor = new CoverageNodeVisitor(program);
          visitor.visitAll(program.getResolvedFunction().getBody());
          visitor.dump(out);
        });
    out.close();
  }

  class CoverageNodeVisitor extends CoverageVisitor {

    private final String filename;
    private final List<FunctionInfo> functions = new ArrayList<>();
    private final List<BranchInfo> branches = new ArrayList<>();
    private final Map<Integer, Long> lines = new HashMap<>();

    CoverageNodeVisitor(Program program) {
      filename = program.getFilename();
    }

    @Override
    protected void visitFunction(String identifier, Node defStatement, Node firstBodyStatement) {
      functions.add(new FunctionInfo(identifier, defStatement.getStartLocation().line(),
          counts.get(firstBodyStatement).sum()));
    }

    @Override
    protected void visitBranch(Node owner, Node condition, Node positiveUniqueSuccessor,
        @Nullable Node negativeUniqueSuccessor) {
      int ownerLine = owner.getStartLocation().line();
      if (counts.get(condition).sum() == 0) {
        // The branch condition has never been executed.
        branches.add(new BranchInfo(ownerLine, null, null));
      } else {
        branches.add(new BranchInfo(ownerLine,
            counts.get(positiveUniqueSuccessor).sum(),
            negativeUniqueSuccessor != null
                ? counts.get(negativeUniqueSuccessor).sum()
                : virtualJumpCounts.get(owner).sum()));
      }
    }

    @Override
    protected void visitCode(Node node) {
      // Update the coverage count for the lines spanned by this node. This is correct since the
      // CoverageVisitor visits nodes from outermost to innermost lexical scope.
      linesToMarkCovered(node).forEach(line -> lines.put(line, counts.get(node).sum()));
    }

    void dump(PrintWriter out) {
      out.println(String.format("SF:%s", filename));

      List<FunctionInfo> sortedFunctions = functions.stream()
          .sorted(Comparator.<FunctionInfo>comparingInt(fi -> fi.line)
              .thenComparing(fi -> fi.identifier))
          .collect(Collectors.toList());
      for (FunctionInfo info : sortedFunctions) {
        out.println(String.format("FN:%d,%s", info.line, info.identifier));
      }
      int numExecutedFunctions = 0;
      for (FunctionInfo info : sortedFunctions) {
        if (info.count > 0) {
          numExecutedFunctions++;
        }
        out.println(String.format("FNDA:%d,%s", info.count, info.identifier));
      }
      out.println(String.format("FNF:%d", functions.size()));
      out.println(String.format("FNH:%d", numExecutedFunctions));

      branches.sort(Comparator.comparing(lc -> lc.ownerLine));
      int numExecutedBranches = 0;
      for (int id = 0; id < branches.size(); id++) {
        BranchInfo info = branches.get(id);
        if (info.positiveCount != null && info.positiveCount > 0) {
          numExecutedBranches++;
        }
        if (info.negativeCount != null && info.negativeCount > 0) {
          numExecutedBranches++;
        }
        // By assigning the same block id to both branches, the coverage viewer will know to group
        // them together.
        out.println(String.format("BRDA:%d,%d,%d,%s",
            info.ownerLine,
            id,
            0,
            info.positiveCount == null ? "-" : info.positiveCount));
        out.println(String.format("BRDA:%d,%d,%d,%s",
            info.ownerLine,
            id,
            1,
            info.negativeCount == null ? "-" : info.negativeCount));
      }
      out.println(String.format("BRF:%d", branches.size()));
      out.println(String.format("BRH:%d", numExecutedBranches));

      List<Integer> sortedLines = lines.keySet().stream().sorted().collect(Collectors.toList());
      int numExecutedLines = 0;
      for (int line : sortedLines) {
        long count = lines.get(line);
        if (count > 0) {
          numExecutedLines++;
        }
        out.println(String.format("DA:%d,%d", line, count));
      }
      out.println(String.format("LF:%d", lines.size()));
      out.println(String.format("LH:%d", numExecutedLines));

      out.println("end_of_record");
    }

    /**
     * Given a node in the AST, returns an {@link IntStream} that yields all source file lines to
     * which the coverage information of {@code node} should be propagated.
     * <p>
     * This usually returns all lines between the start and end location of {@code node}, but may
     * return fewer lines for block statements such as {@code if}.
     */
    private IntStream linesToMarkCovered(Node node) {
      if (!(node instanceof Statement)) {
        return IntStream.rangeClosed(node.getStartLocation().line(), node.getEndLocation().line());
      }
      // Handle block statements specially so that they don't mark their entire scope as covered,
      // which would also include comments and empty lines.
      switch (((Statement) node).kind()) {
        case IF:
          return IntStream.rangeClosed(node.getStartLocation().line(),
              ((IfStatement) node).getCondition().getEndLocation().line());
        case FOR:
          return IntStream.rangeClosed(node.getStartLocation().line(),
              ((ForStatement) node).getCollection().getEndLocation().line());
        case DEF:
          DefStatement defStatement = (DefStatement) node;
          if (defStatement.getParameters().isEmpty()) {
            return IntStream.of(node.getStartLocation().line());
          }
          Parameter lastParam = defStatement.getParameters()
              .get(defStatement.getParameters().size() - 1);
          return IntStream.rangeClosed(defStatement.getStartLocation().line(),
              lastParam.getEndLocation().line());
        default:
          return IntStream.rangeClosed(node.getStartLocation().line(),
              node.getEndLocation().line());
      }
    }

    private class FunctionInfo {

      final String identifier;
      final int line;
      final long count;

      FunctionInfo(String identifier, int line, long count) {
        this.identifier = identifier;
        this.line = line;
        this.count = count;
      }
    }

    private class BranchInfo {

      final int ownerLine;
      // Both positiveCount and negativeCount are null if the branch condition hasn't been executed.
      // Otherwise, they give the number of times the positive case jump resp. the negative case
      // jump was taken (and are in particular not null).
      final Long positiveCount;
      final Long negativeCount;

      BranchInfo(int ownerLine, @Nullable Long positiveCount, @Nullable Long negativeCount) {
        this.ownerLine = ownerLine;
        this.positiveCount = positiveCount;
        this.negativeCount = negativeCount;
      }
    }
  }
}
