package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;

import java.util.*;
import java.util.stream.IntStream;

/** Starlark bytecode compiler. */
class Bc {

  /**
   * This constant enables/disables assertions in Starlark interpreter: when turned on, it checks
   * that:
   *
   * <ul>
   *   <li>Compiler generates valid opcode arguments, according to opcode spec
   *   <li>Interpreter decodes opcode arguments correctly (e. g. does not consume extra undeclared
   *       argument)
   * </ul>
   *
   * Turn assertions on when debugging the compiler or interpreter.
   *
   * <p>Note the assertions are turned on when tests are launched from Bazel.
   */
  static final boolean ASSERTIONS = Boolean.getBoolean("starlark.bc.assertions");

  static {
    if (ASSERTIONS) {
      System.err.println();
      System.err.println();
      System.err.println("Bc.ASSERTIONS = true");
      System.err.println();
      System.err.println();
    }
  }

  /** Function body as a bytecode block. */
  static class Compiled {
    /** Assuming all statements belong to the same {@link FileLocations} object. */
    final FileLocations locs;
    /** Strings references by the bytecode. */
    final String[] strings;
    /** Other objects references by the bytecode. */
    final Object[] objects;
    /** The bytecode. */
    final int[] text;
    /** Number of registers. */
    final int slotCount;
    /** Registers holding constants. */
    final Object[] constSlots;
    /** Max depths of for loops. */
    final int loopDepth;
    /**
     * Instruction pointer to a node.
     *
     * <p>Key is a beginning of an instruction.
     */
    final ImmutableMap<Integer, Node> instrToNode;

    private Compiled(
        FileLocations locs,
        String[] strings,
        Object[] objects,
        int[] text,
        int slotCount,
        Object[] constSlots,
        int loopDepth,
        ImmutableMap<Integer, Node> instrToNode) {
      this.locs = locs;
      this.strings = strings;
      this.objects = objects;
      this.text = text;
      this.slotCount = slotCount;
      this.constSlots = constSlots;
      this.loopDepth = loopDepth;
      this.instrToNode = instrToNode;
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      int ip = 0;
      while (ip != text.length) {
        if (sb.length() != 0) {
          sb.append("; ");
        }
        sb.append(ip).append(": ");
        int opcode1 = text[ip++];
        BcInstr.Opcode opcode = BcInstr.Opcode.values()[opcode1];
        sb.append(opcode);
        int[] updateIp = new int[] {ip};
        String argsString =
            opcode.operands.toStringAndCount(
                updateIp, text, Arrays.asList(strings), Arrays.asList(constSlots));
        ip = updateIp[0];
        sb.append(" ").append(argsString);
      }
      if (sb.length() != 0) {
        sb.append("; ");
      }
      // It's useful to know the final address in case someone wants to jump to that address
      sb.append(ip).append(": EOF");
      return sb.toString();
    }

    /** Instruction opcode at IP. */
    BcInstr.Opcode instrOpcodeAt(int ip) {
      return BcInstr.Opcode.values()[text[ip]];
    }

    /** Print instruction at the given pointer. */
    String instrStringAt(int ip) {
      if (ip == text.length) {
        return "EOF";
      }

      BcInstr.Opcode opcode = instrOpcodeAt(ip++);
      return opcode + " " + opcode.operands.argToString(ip, this);
    }

    /** Instruction length at IP. */
    int instrLenAt(int ip) {
      return BcInstr.INSTR_HEADER_LEN
          + instrOpcodeAt(ip)
              .operands
              .codeSize(
                  text,
                  Arrays.asList(strings),
                  Arrays.asList(constSlots),
                  ip + BcInstr.INSTR_HEADER_LEN);
    }
  }

  /** Current for block in the compiler; used to compile break and continue statements. */
  private static class CurrentFor {
    /** Instruction pointer of the for statement body. */
    private final int bodyIp;

    /**
     * Register which stores next iterator value. This register is updated by {@code FOR_INIT} and
     * {@code CONTINUE} instructions.
     */
    private final int nextValueSlot;

    /**
     * Pointers to the pointers to the end of the for statement body; patched in the end of the for
     * compilation.
     */
    private ArrayList<Integer> endsToPatch = new ArrayList<>();

    private CurrentFor(int bodyIp, int nextValueSlot) {
      this.bodyIp = bodyIp;
      this.nextValueSlot = nextValueSlot;
    }
  }

  /** Store values indexed by an integer. */
  private static class IndexedList<T> {
    private ArrayList<T> values = new ArrayList<>();
    private HashMap<T, Integer> index = new HashMap<>();

    int index(T s) {
      return index.computeIfAbsent(
          s,
          k -> {
            int r = values.size();
            values.add(s);
            return r;
          });
    }
  }

  /**
   * The compiler implementation. The entry point to the compiler is static {@link
   * #compileFunction(FileLocations, List)} function.
   */
  private static class Compiler {
    private static final int[] EMPTY_INTS = {};

    /** Assuming all nodes in the function belong to the same locations object. */
    private final FileLocations locs;
    /** {@code 0..ip} of the array is bytecode. */
    private int[] text = EMPTY_INTS;
    /** Current instruction pointer. */
    private int ip = 0;
    /** Number of currently allocated registers. */
    private int slots = 0;
    /** Total number of registers needed to execute this function. */
    private int maxSlots = 0;

    /** Starlark values as constant registers. */
    private IndexedList<Object> constSlots = new IndexedList<>();

    /** Strings referenced in currently built bytecode. */
    private IndexedList<String> strings = new IndexedList<>();

    /** Other untyped objects referenced in currently built bytecode. */
    private ArrayList<Object> objects = new ArrayList<>();

    /** The stack of for statements. */
    private ArrayList<CurrentFor> fors = new ArrayList<>();
    /** Max depth of for loops. */
    private int maxLoopDepth = 0;

    private ImmutableMap.Builder<Integer, Node> instrToNode = ImmutableMap.builder();

    private Compiler(FileLocations locs) {
      this.locs = locs;
    }

    /** Closest containing for statement. */
    private CurrentFor currentFor() {
      return fors.get(fors.size() - 1);
    }

    /** Allocate a register. */
    private int allocSlot() {
      int r = slots++;
      maxSlots = Math.max(slots, maxSlots);
      return r;
    }

    /**
     * Deallocate all registers (except constant registers); done after each statement; since
     * registered are not shared between statements, only local variables are.
     */
    private void decallocateAllSlots() {
      slots = 0;
    }

    /**
     * Store a string in a string pool, return an index of that string. Note these strings are
     * special strings like variable or field names. These are not constant registers.
     */
    private int allocString(String s) {
      return strings.index(s);
    }

    /**
     * Store an arbitrary object in an object storage; the object store is not a const registers.
     */
    private int allocObject(Object o) {
      int r = objects.size();
      objects.add(o);
      return r;
    }

    /** Write complete opcode with validation. */
    private void write(BcInstr.Opcode opcode, Node node, int... args) {
      instrToNode.put(ip, node);

      int prevIp = ip;

      int instrLen = BcInstr.INSTR_HEADER_LEN + args.length;
      if (ip + instrLen > text.length) {
        text = Arrays.copyOf(text, Math.max(text.length * 2, ip + instrLen));
      }

      text[ip++] = opcode.ordinal();
      System.arraycopy(args, 0, text, ip, args.length);
      ip += args.length;

      if (ASSERTIONS) {
        int expectedArgCount =
            opcode.operands.codeSize(
                text, strings.values, constSlots.values, prevIp + BcInstr.INSTR_HEADER_LEN);
        Preconditions.checkState(
            expectedArgCount == args.length,
            "incorrect signature for %s: expected %s, actual %s",
            opcode,
            expectedArgCount,
            args.length);
      }
    }

    /** Marker address for yet unknown forward jump. */
    private static final int FORWARD_JUMP_ADDR = -17;

    /**
     * Write forward condition jump instruction. Return an address to be patched when the jump
     * address is known.
     */
    private int writeForwardCondJump(BcInstr.Opcode opcode, Node expression, int cond) {
      Preconditions.checkState(
          opcode == BcInstr.Opcode.IF_BR || opcode == BcInstr.Opcode.IF_NOT_BR);
      write(opcode, expression, cond, FORWARD_JUMP_ADDR);
      return ip - 1;
    }

    /**
     * Write unconditional forward jump. Return an address to be patched when the jump address is
     * known.
     */
    private int writeForwardJump(Node expression) {
      write(BcInstr.Opcode.BR, expression, FORWARD_JUMP_ADDR);
      return ip - 1;
    }

    /** Patch previously registered forward jump address. */
    private void patchForwardJump(int ip) {
      Preconditions.checkState(text[ip] == FORWARD_JUMP_ADDR);
      text[ip] = this.ip;
    }

    /** Compile. */
    private void compileStatements(List<Statement> statements, boolean postAssignHook) {
      for (Statement statement : statements) {
        compileStatement(statement, postAssignHook);
      }
    }

    private void compileStatement(Statement statement, boolean postAssignHook) {
      // No registers are shared across statements.
      // We could implement precise register tracking, but there is no need for that at the moment.
      decallocateAllSlots();

      write(BcInstr.Opcode.DBG, statement);

      if (statement instanceof ExpressionStatement) {
        // Do not assign it anywhere
        compileExpression(((ExpressionStatement) statement).getExpression());
      } else if (statement instanceof AssignmentStatement) {
        compileAssignment((AssignmentStatement) statement, postAssignHook);
      } else if (statement instanceof ReturnStatement) {
        ReturnStatement returnStatement = (ReturnStatement) statement;
        if (returnStatement.getResult() == null) {
          write(BcInstr.Opcode.RETURN_NONE, returnStatement);
        } else {
          int result = compileExpression(returnStatement.getResult());
          write(BcInstr.Opcode.RETURN, returnStatement, result);
        }
      } else if (statement instanceof IfStatement) {
        compileIfStatement((IfStatement) statement);
      } else if (statement instanceof ForStatement) {
        compileForStatement((ForStatement) statement);
      } else if (statement instanceof FlowStatement) {
        compileFlowStatement((FlowStatement) statement);
      } else if (statement instanceof DefStatement || statement instanceof LoadStatement) {
        compileGenericStatement(statement);
      } else {
        throw new RuntimeException("not impl: " + statement.getClass().getSimpleName());
      }
    }

    private void compileGenericStatement(Statement statement) {
      Preconditions.checkState(
          statement instanceof DefStatement || statement instanceof LoadStatement);
      write(BcInstr.Opcode.STMT, statement, allocObject(statement));
    }

    private void compileIfStatement(IfStatement ifStatement) {
      Expression condExpr = ifStatement.getCondition();

      int cond;
      BcInstr.Opcode elseBrOpcode;
      if (condExpr instanceof UnaryOperatorExpression
          && ((UnaryOperatorExpression) condExpr).getOperator() == TokenKind.NOT) {
        // special case `if not cond: ...` micro-optimization
        cond = compileExpression(((UnaryOperatorExpression) condExpr).getX());
        elseBrOpcode = BcInstr.Opcode.IF_BR;
      } else {
        cond = compileExpression(condExpr);
        elseBrOpcode = BcInstr.Opcode.IF_NOT_BR;
      }

      int elseBlock = writeForwardCondJump(elseBrOpcode, ifStatement, cond);
      compileStatements(ifStatement.getThenBlock(), false);
      if (ifStatement.getElseBlock() != null) {
        int end = writeForwardJump(ifStatement);
        patchForwardJump(elseBlock);
        compileStatements(ifStatement.getElseBlock(), false);
        patchForwardJump(end);
      } else {
        patchForwardJump(elseBlock);
      }
    }

    private void compileFlowStatement(FlowStatement flowStatement) {
      switch (flowStatement.getKind()) {
        case BREAK:
          compileBreak(flowStatement);
          break;
        case CONTINUE:
          compileContinue(flowStatement);
          break;
        case PASS:
          // nop
          break;
        default:
          throw new IllegalStateException("unknown flow statement: " + flowStatement.getKind());
      }
    }

    private void compileContinue(Node node) {
      if (fors.isEmpty()) {
        compileThrowException(node, "continue statement must be inside a for loop");
      } else {
        write(
            BcInstr.Opcode.CONTINUE,
            node,
            currentFor().nextValueSlot,
            currentFor().bodyIp,
            FORWARD_JUMP_ADDR);
        currentFor().endsToPatch.add(ip - 1);
      }
    }

    private void compileBreak(Node node) {
      if (fors.isEmpty()) {
        compileThrowException(node, "break statement must be inside a for loop");
      } else {
        write(BcInstr.Opcode.BREAK, node, FORWARD_JUMP_ADDR);
        currentFor().endsToPatch.add(ip - 1);
      }
    }

    /** Callback invoked to compile the loop body. */
    private interface ForBody {
      void compile();
    }

    /** Generic compile for loop routine, used in for statement and in loop comprehension. */
    private void compileFor(Expression vars, Expression collection, ForBody body) {
      int iterable = compileExpression(collection);

      // Register where we are storing the next iterator value.
      // This register is update by FOR_INIT and CONTINUE instructions.
      int nextValueSlot = allocSlot();

      write(BcInstr.Opcode.FOR_INIT, collection, iterable, nextValueSlot, FORWARD_JUMP_ADDR);
      int endToPatch = ip - 1;

      CurrentFor currentFor = new CurrentFor(ip, nextValueSlot);
      fors.add(currentFor);
      currentFor.endsToPatch.add(endToPatch);

      compileAssignment(currentFor.nextValueSlot, vars, false);

      maxLoopDepth = Math.max(fors.size(), maxLoopDepth);

      body.compile();

      // We use usual CONTINUE statement in the end of the loop.
      // Note: CONTINUE does unnecessary goto e in the end of iteration.
      compileContinue(collection);

      for (int endsToPatch : currentFor.endsToPatch) {
        patchForwardJump(endsToPatch);
      }
      fors.remove(fors.size() - 1);
    }

    private void compileForStatement(ForStatement forStatement) {
      compileFor(
          forStatement.getVars(),
          forStatement.getCollection(),
          () -> compileStatements(forStatement.getBody(), false));
    }

    private void compileAssignment(
        AssignmentStatement assignmentStatement, boolean postAssignHook) {
      if (assignmentStatement.isAugmented()) {
        compileAgumentedAssignment(assignmentStatement);
      } else {
        compileAssignmentRegular(assignmentStatement, postAssignHook);
      }
    }

    private void compileAssignmentRegular(
        AssignmentStatement assignmentStatement, boolean postAssignHook) {
      Preconditions.checkState(!assignmentStatement.isAugmented());
      int rhs = compileExpression(assignmentStatement.getRHS());
      compileAssignment(rhs, assignmentStatement.getLHS(), postAssignHook);
    }

    private void compileAssignment(int rhs, Expression lhs, boolean postAssignHook) {
      if (lhs instanceof Identifier) {
        compileSet(rhs, (Identifier) lhs, postAssignHook);
      } else if (lhs instanceof ListExpression) {
        compileAssignmentToList(rhs, (ListExpression) lhs, postAssignHook);
      } else if (lhs instanceof IndexExpression) {
        IndexExpression indexExpression = (IndexExpression) lhs;
        int object = compileExpression(indexExpression.getObject());
        int key = compileExpression(indexExpression.getKey());
        write(BcInstr.Opcode.SET_INDEX, lhs, object, key, rhs);
      } else {
        compileThrowException(lhs, String.format("cannot assign to '%s'", lhs));
      }
    }

    private void compileAssignmentToList(int rhs, ListExpression list, boolean postAssignHook) {
      if (list.getElements().isEmpty()) {
        // TODO: emit an error in resolver or allow it in the spec:
        //  https://github.com/bazelbuild/starlark/issues/93
        compileThrowException(list, String.format("can't assign to %s", list));
        return;
      }

      int[] componentRegs =
          IntStream.range(0, list.getElements().size()).map(i1 -> allocSlot()).toArray();

      int[] args = new int[2 + list.getElements().size()];
      args[0] = rhs;
      args[1] = list.getElements().size();
      System.arraycopy(componentRegs, 0, args, 2, componentRegs.length);
      write(BcInstr.Opcode.UNPACK, list, args);

      for (int i = 0; i < componentRegs.length; i++) {
        int componentReg = componentRegs[i];
        compileAssignment(componentReg, list.getElements().get(i), postAssignHook);
      }
    }

    private void compileSet(int rhs, Identifier identifier, boolean postAssignHook) {
      BcInstr.Opcode opcode;
      if (identifier.getBinding() != null) {
        switch (identifier.getBinding().scope) {
          case LOCAL:
            opcode = BcInstr.Opcode.SET_LOCAL;
            break;
          case GLOBAL:
            opcode = BcInstr.Opcode.SET_GLOBAL;
            break;
          default:
            throw new IllegalStateException();
        }
      } else {
        // TODO: resolve and remove
        opcode = BcInstr.Opcode.SET_LEGACY;
      }

      write(opcode, identifier, rhs, allocString(identifier.getName()), postAssignHook ? 1 : 0);
    }

    private void compileThrowException(Node node, String message) {
      // All incorrect AST should be resolved by the resolver,
      // compile code to throw exception as a stopgap.
      write(BcInstr.Opcode.EVAL_EXCEPTION, node, allocString(message));
    }

    private void compileAgumentedAssignment(AssignmentStatement assignmentStatement) {
      Preconditions.checkState(assignmentStatement.getOperator() != null);
      if (assignmentStatement.getLHS() instanceof Identifier) {
        int rhs = compileExpression(assignmentStatement.getRHS());
        Identifier lhs = (Identifier) assignmentStatement.getLHS();
        int temp = allocSlot();
        compileGet(lhs, temp);
        write(
            BcInstr.Opcode.BINARY_IN_PLACE,
            assignmentStatement,
            temp,
            rhs,
            assignmentStatement.getOperator().ordinal(),
            temp);
        compileSet(temp, lhs, false);
      } else if (assignmentStatement.getLHS() instanceof IndexExpression) {
        IndexExpression indexExpression = (IndexExpression) assignmentStatement.getLHS();

        int object = compileExpression(indexExpression.getObject());
        int key = compileExpression(indexExpression.getKey());
        int rhs = compileExpression(assignmentStatement.getRHS());
        int temp = allocSlot();
        write(BcInstr.Opcode.INDEX, assignmentStatement, object, key, temp);
        write(
            BcInstr.Opcode.BINARY_IN_PLACE,
            assignmentStatement,
            temp,
            rhs,
            assignmentStatement.getOperator().ordinal(),
            temp);
        write(BcInstr.Opcode.SET_INDEX, assignmentStatement, object, key, temp);
      } else if (assignmentStatement.getLHS() instanceof ListExpression) {
        compileThrowException(
            assignmentStatement.getLHS(),
            "cannot perform augmented assignment on a list or tuple expression");
      } else {
        compileThrowException(
            assignmentStatement.getLHS(),
            String.format("cannot assign to '%s'", assignmentStatement.getLHS()));
      }
    }

    /** Compile a constant, return a register containing the constant. */
    private int compileConstant(Object constant) {
      return BcInstr.constSlotFromArrayIndex(constSlots.index(constant));
    }

    /** Compile a constant, store it in provided register. */
    private void compileConstantTo(Node expression, Object constant, int result) {
      Preconditions.checkState(result >= 0);
      int value = compileConstant(constant);
      write(BcInstr.Opcode.CP, expression, value, result);
    }

    /** Compile an expression, store result in provided register. */
    private void compileExpressionTo(Expression expression, int result) {
      Preconditions.checkState(result >= 0);

      if (expression instanceof SliceExpression) {
        compileSliceExpression((SliceExpression) expression, result);
      } else if (expression instanceof Comprehension) {
        compileComprehension((Comprehension) expression, result);
      } else if (expression instanceof ListExpression) {
        compileList((ListExpression) expression, result);
      } else if (expression instanceof DictExpression) {
        compileDict((DictExpression) expression, result);
      } else if (expression instanceof CallExpression) {
        compileCall((CallExpression) expression, result);
      } else if (expression instanceof ConditionalExpression) {
        compileConditional((ConditionalExpression) expression, result);
      } else if (expression instanceof DotExpression) {
        compileDot((DotExpression) expression, result);
      } else if (expression instanceof IndexExpression) {
        compileIndex((IndexExpression) expression, result);
      } else if (expression instanceof UnaryOperatorExpression) {
        compileUnaryOperator(expression, result);
      } else if (expression instanceof BinaryOperatorExpression) {
        compileBinaryOperator(expression, result);
      } else if (expression instanceof Identifier) {
        compileGet((Identifier) expression, result);
      } else if (expression instanceof StringLiteral) {
        compileConstantTo(expression, ((StringLiteral) expression).getValue(), result);
      } else if (expression instanceof IntegerLiteral) {
        compileConstantTo(expression, ((IntegerLiteral) expression).getValue(), result);
      } else {
        throw new RuntimeException("not impl: " + expression.getClass().getSimpleName());
      }
    }

    /** Compile an expression and return a register containing the result. */
    private int compileExpression(Expression expression) {
      if (expression instanceof StringLiteral) {
        return compileConstant(((StringLiteral) expression).getValue());
      } else if (expression instanceof IntegerLiteral) {
        return compileConstant(((IntegerLiteral) expression).getValue());
      } else {
        int result = allocSlot();
        compileExpressionTo(expression, result);
        return result;
      }
    }

    /**
     * Compile expression, return a register containing result. Given register may or may not be
     * used to store the result.
     */
    private int compileExpressionMaybeReuseSlot(Expression expression, int slot) {
      if (expression instanceof StringLiteral || expression instanceof IntegerLiteral) {
        return compileExpression(expression);
      } else {
        compileExpressionTo(expression, slot);
        return slot;
      }
    }

    private void compileIndex(IndexExpression expression, int result) {
      int object = compileExpressionMaybeReuseSlot(expression.getObject(), result);
      int key = compileExpression(expression.getKey());
      write(BcInstr.Opcode.INDEX, expression, object, key, result);
    }

    private void compileDot(DotExpression dotExpression, int result) {
      int object = compileExpressionMaybeReuseSlot(dotExpression.getObject(), result);
      write(
          BcInstr.Opcode.DOT,
          dotExpression,
          object,
          allocString(dotExpression.getField().getName()),
          result);
    }

    private void compileSliceExpression(SliceExpression slice, int result) {
      int object = compileExpressionMaybeReuseSlot(slice.getObject(), result);

      int start = slice.getStart() != null ? compileExpression(slice.getStart()) : BcInstr.NULL_REG;
      int stop = slice.getStop() != null ? compileExpression(slice.getStop()) : BcInstr.NULL_REG;
      int step = slice.getStep() != null ? compileExpression(slice.getStep()) : BcInstr.NULL_REG;

      write(BcInstr.Opcode.SLICE, slice, object, start, stop, step, result);
    }

    private void compileGet(Identifier identifier, int result) {
      BcInstr.Opcode opcode;
      if (identifier.getBinding() == null) {
        opcode = BcInstr.Opcode.GET_LEGACY;
      } else {
        switch (identifier.getBinding().scope) {
          case LOCAL:
            opcode = BcInstr.Opcode.GET_LOCAL;
            break;
          case GLOBAL:
            opcode = BcInstr.Opcode.GET_GLOBAL;
            break;
          case PREDECLARED:
            opcode = BcInstr.Opcode.GET_PREDECLARED;
            break;
          default:
            throw new IllegalStateException("unknown scope: " + identifier.getBinding().scope);
        }
      }
      write(opcode, identifier, allocString(identifier.getName()), result);
    }

    private void compileComprehension(Comprehension comprehension, int result) {
      LinkedHashSet<String> boundIdentifiers = new LinkedHashSet<>();
      for (Comprehension.Clause clause : comprehension.getClauses()) {
        if (clause instanceof Comprehension.For) {
          for (Identifier identifier :
              Identifier.boundIdentifiers(((Comprehension.For) clause).getVars())) {
            boundIdentifiers.add(identifier.getName());
          }
        }
      }

      String[] boundIdentifiersArray = boundIdentifiers.toArray(new String[0]);
      int boundedArrayIndex = allocObject(boundIdentifiersArray);

      write(BcInstr.Opcode.SAVE_LOCALS, comprehension, boundedArrayIndex);

      if (comprehension.isDict()) {
        write(BcInstr.Opcode.DICT, comprehension.getBody(), 0, result);
      } else {
        write(BcInstr.Opcode.LIST, comprehension.getBody(), 0, result);
      }

      // The Lambda class serves as a recursive lambda closure.
      class Lambda {
        // execClauses(index) recursively compiles the clauses starting at index,
        // and finally compiles the body and adds its value to the result.
        private void compileClauses(int index) {
          // recursive case: one or more clauses
          if (index != comprehension.getClauses().size()) {
            Comprehension.Clause clause = comprehension.getClauses().get(index);
            if (clause instanceof Comprehension.For) {
              compileFor(
                  ((Comprehension.For) clause).getVars(),
                  ((Comprehension.For) clause).getIterable(),
                  () -> compileClauses(index + 1));
            } else if (clause instanceof Comprehension.If) {
              int cond = compileExpression(((Comprehension.If) clause).getCondition());
              int end = writeForwardCondJump(BcInstr.Opcode.IF_NOT_BR, clause, cond);
              compileClauses(index + 1);
              patchForwardJump(end);
            } else {
              throw new IllegalStateException("unknown compr clause: " + clause);
            }
          } else {
            if (comprehension.isDict()) {
              DictExpression.Entry entry = (DictExpression.Entry) comprehension.getBody();
              int key = compileExpression(entry.getKey());
              int value = compileExpression(entry.getValue());
              write(BcInstr.Opcode.SET_INDEX, entry, result, key, value);
            } else {
              int value = compileExpression((Expression) comprehension.getBody());
              write(BcInstr.Opcode.LIST_APPEND, comprehension.getBody(), result, value);
            }
          }
        }
      }

      new Lambda().compileClauses(0);

      write(BcInstr.Opcode.RESTORE_LOCALS, comprehension, boundedArrayIndex);
    }

    private void compileDict(DictExpression dictExpression, int result) {
      Preconditions.checkState(result >= 0);

      int[] args = new int[1 + dictExpression.getEntries().size() * 2 + 1];
      int i = 0;
      args[i++] = dictExpression.getEntries().size();
      for (DictExpression.Entry entry : dictExpression.getEntries()) {
        args[i++] = compileExpression(entry.getKey());
        args[i++] = compileExpression(entry.getValue());
      }
      args[i++] = result;
      Preconditions.checkState(i == args.length);

      write(BcInstr.Opcode.DICT, dictExpression, args);
    }

    private void compileList(ListExpression listExpression, int result) {
      Preconditions.checkState(result >= 0);

      int[] args = new int[1 + listExpression.getElements().size() + 1];
      int i = 0;
      args[i++] = listExpression.getElements().size();
      for (Expression element : listExpression.getElements()) {
        args[i++] = compileExpression(element);
      }
      args[i++] = result;
      Preconditions.checkState(i == args.length);
      write(
          listExpression.isTuple() ? BcInstr.Opcode.TUPLE : BcInstr.Opcode.LIST,
          listExpression,
          args);
    }

    private void compileConditional(ConditionalExpression conditionalExpression, int result) {
      int cond = compileExpressionMaybeReuseSlot(conditionalExpression.getCondition(), result);
      int thenAddr = writeForwardCondJump(BcInstr.Opcode.IF_NOT_BR, conditionalExpression, cond);
      compileExpressionTo(conditionalExpression.getThenCase(), result);
      int end = writeForwardJump(conditionalExpression);
      patchForwardJump(thenAddr);
      compileExpressionTo(conditionalExpression.getElseCase(), result);
      patchForwardJump(end);
    }

    private void compileCall(CallExpression callExpression, int result) {
      ArrayList<Argument.Positional> positionals = new ArrayList<>();
      ArrayList<Argument.Keyword> nameds = new ArrayList<>();
      Argument.Star star = null;
      Argument.StarStar starStar = null;
      for (Argument argument : callExpression.getArguments()) {
        if (argument instanceof Argument.Positional) {
          positionals.add((Argument.Positional) argument);
        } else if (argument instanceof Argument.Keyword) {
          nameds.add((Argument.Keyword) argument);
        } else if (argument instanceof Argument.Star) {
          star = (Argument.Star) argument;
        } else if (argument instanceof Argument.StarStar) {
          starStar = (Argument.StarStar) argument;
        } else {
          throw new IllegalStateException();
        }
      }
      int numCallArgs = 2; // lparen + fn
      numCallArgs += 1 + positionals.size();
      numCallArgs += 1 + (2 * nameds.size());
      numCallArgs += 5; // star, star-loc, star-star, star-star-loc, result
      int[] args = new int[numCallArgs];

      int i = 0;
      args[i++] = allocObject(callExpression.getLparenLocation());
      args[i++] = compileExpressionMaybeReuseSlot(callExpression.getFunction(), result);

      args[i++] = positionals.size();
      for (Argument.Positional positional : positionals) {
        args[i++] = compileExpression(positional.getValue());
      }
      args[i++] = nameds.size();
      for (Argument.Keyword named : nameds) {
        args[i++] = allocString(named.getName());
        args[i++] = compileExpression(named.getValue());
      }
      args[i++] = star != null ? compileExpression(star.getValue()) : BcInstr.NULL_REG;
      args[i++] = star != null ? star.getStartOffset() : BcInstr.UNDEFINED_LOC;
      args[i++] = starStar != null ? compileExpression(starStar.getValue()) : BcInstr.NULL_REG;
      args[i++] = starStar != null ? starStar.getStartOffset() : BcInstr.UNDEFINED_LOC;

      args[i++] = result;

      Preconditions.checkState(i == args.length);

      write(BcInstr.Opcode.CALL, callExpression, args);
    }

    private void compileUnaryOperator(Expression expression, int result) {
      UnaryOperatorExpression unaryOperatorExpression = (UnaryOperatorExpression) expression;
      int value = compileExpressionMaybeReuseSlot(unaryOperatorExpression.getX(), result);
      if (unaryOperatorExpression.getOperator() == TokenKind.NOT) {
        write(BcInstr.Opcode.NOT, expression, value, result);
      } else {
        write(
            BcInstr.Opcode.UNARY,
            expression,
            value,
            unaryOperatorExpression.getOperator().ordinal(),
            result);
      }
    }

    private void compileBinaryOperator(Expression expression, int result) {
      Preconditions.checkState(result >= 0);

      BinaryOperatorExpression binaryOperatorExpression = (BinaryOperatorExpression) expression;
      switch (binaryOperatorExpression.getOperator()) {
        case AND:
          {
            compileExpressionTo(binaryOperatorExpression.getX(), result);
            int end = writeForwardCondJump(BcInstr.Opcode.IF_NOT_BR, expression, result);
            compileExpressionTo(binaryOperatorExpression.getY(), result);
            patchForwardJump(end);
            break;
          }
        case OR:
          {
            compileExpressionTo(binaryOperatorExpression.getX(), result);
            int end = writeForwardCondJump(BcInstr.Opcode.IF_BR, expression, result);
            compileExpressionTo(binaryOperatorExpression.getY(), result);
            patchForwardJump(end);
            break;
          }
        case EQUALS_EQUALS:
          {
            int x = compileExpression(binaryOperatorExpression.getX());
            int y = compileExpression(binaryOperatorExpression.getY());
            write(BcInstr.Opcode.EQ, expression, x, y, result);
            break;
          }
        case NOT_EQUALS:
          {
            int x = compileExpression(binaryOperatorExpression.getX());
            int y = compileExpression(binaryOperatorExpression.getY());
            write(BcInstr.Opcode.NOT_EQ, expression, x, y, result);
            break;
          }
        default:
          {
            int x = compileExpression(binaryOperatorExpression.getX());
            int y = compileExpression(binaryOperatorExpression.getY());
            write(
                BcInstr.Opcode.BINARY,
                expression,
                x,
                y,
                binaryOperatorExpression.getOperator().ordinal(),
                result);
          }
      }
    }

    private static final String[] EMPTY_STRINGS = {};
    private static final Object[] EMPTY = {};

    Compiled finish() {
      return new Compiled(
          locs,
          strings.values.toArray(EMPTY_STRINGS),
          objects.toArray(EMPTY),
          Arrays.copyOf(text, ip),
          maxSlots,
          constSlots.values.toArray(EMPTY),
          maxLoopDepth,
          instrToNode.build());
    }
  }

  static Compiled compileFunction(FileLocations locs, List<Statement> statements) {
    Compiler compiler = new Compiler(locs);
    compiler.compileStatements(statements, true);
    return compiler.finish();
  }
}
