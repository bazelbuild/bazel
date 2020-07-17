package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;

import javax.annotation.Nullable;
import java.util.*;

/** Bytecode interpreter. Takes a compiled function body and returns a result. */
class BcEval {
  private static final Object[] EMPTY = {};
  private static final TokenKind[] TOKENS = TokenKind.values();

  private final StarlarkThread.Frame fr;
  private final Bc.Compiled compiled;

  /** Registers. */
  private final Object[] slots;

  /**
   * Currently executed loops stack: pairs of (iterable, iterator).
   *
   * <p>The array is preallocated, {@link #loopDepth} holds the number of currently executed loops.
   */
  private final Object[] loops;

  /** Current loop depth. */
  private int loopDepth = 0;

  /** Stack of save locals in comprehension. */
  private ArrayList<Object[]> comprSavedLocals = new ArrayList<>();

  /** Program text. */
  private final int[] text;

  /** Current instruction pointer */
  private int currentIp;

  /** Instruction pointer while decoding operands */
  private int ip = 0;

  private BcEval(StarlarkThread.Frame fr, Bc.Compiled compiled) {
    this.fr = fr;

    this.compiled = compiled;
    this.slots = new Object[compiled.slotCount];
    this.loops = new Object[compiled.loopDepth * 2];
    text = compiled.text;
  }

  /** Public API. */
  public static Object eval(StarlarkThread.Frame fr, Bc.Compiled compiled)
      throws InterruptedException, EvalException {
    return new BcEval(fr, compiled).eval();
  }

  private Object eval() throws EvalException, InterruptedException {
    try {
      while (ip != text.length) {
        fr.thread.steps++;

        currentIp = ip;

        // Each instruction is:
        // * opcode
        // * operands which depend on opcode
        int opcode = text[ip++];
        try {
          switch (opcode) {
            case BcInstr.CP:
              cp();
              break;
            case BcInstr.EQ:
              eq();
              break;
            case BcInstr.NOT_EQ:
              notEq();
              break;
            case BcInstr.NOT:
              not();
              break;
            case BcInstr.UNARY:
              unary();
              break;
            case BcInstr.BINARY:
              binary();
              break;
            case BcInstr.BINARY_IN_PLACE:
              binaryInPlace();
              break;
            case BcInstr.BR:
              br();
              continue;
            case BcInstr.IF_BR:
              ifBr();
              continue;
            case BcInstr.IF_NOT_BR:
              ifNotBr();
              continue;
            case BcInstr.DOT:
              dot();
              break;
            case BcInstr.INDEX:
              index();
              break;
            case BcInstr.SLICE:
              slice();
              break;
            case BcInstr.CALL:
              call();
              break;
            case BcInstr.RETURN_NONE:
              return returnNone();
            case BcInstr.RETURN:
              return returnInstr();
            case BcInstr.TUPLE:
              tuple();
              break;
            case BcInstr.LIST:
              list();
              break;
            case BcInstr.DICT:
              dict();
              break;
            case BcInstr.UNPACK:
              unpack();
              break;
            case BcInstr.GET_LOCAL:
              getLocal();
              break;
            case BcInstr.GET_GLOBAL:
              getGlobal();
              break;
            case BcInstr.GET_PREDECLARED:
              getPredeclared();
              break;
            case BcInstr.GET_LEGACY:
              getLegacy();
              break;
            case BcInstr.SET_GLOBAL:
              setGlobal();
              break;
            case BcInstr.SET_LOCAL:
              setLocal();
              break;
            case BcInstr.SET_LEGACY:
              setLegacy();
              break;
            case BcInstr.STMT:
              stmt();
              break;
            case BcInstr.FOR_INIT:
              forInit();
              continue;
            case BcInstr.BREAK:
              breakInstr();
              continue;
            case BcInstr.CONTINUE:
              continueInstr();
              continue;
            case BcInstr.SAVE_LOCALS:
              saveLocals();
              break;
            case BcInstr.RESTORE_LOCALS:
              restoreLocals();
              break;
            case BcInstr.LIST_APPEND:
              listAppend();
              break;
            case BcInstr.SET_INDEX:
              setIndex();
              break;
            case BcInstr.EVAL_EXCEPTION:
              evalException();
              continue;
            case BcInstr.DBG:
              dbg();
              break;
            default:
              throw otherOpcode(opcode);
          }

          validateInstructionDecodedCorrectly();

        } catch (EvalException e) {
          if (e.canBeAddedToStackTrace()) {
            throw new EvalExceptionWithStackTrace(e, currentInstrNode());
          } else {
            throw e;
          }
        }
      }
    } finally {
      while (loopDepth != 0) {
        popFor();
      }
    }
    return Starlark.NONE;
  }

  /** Pop one for statement. */
  private void popFor() {
    EvalUtils.removeIterator(loops[(loopDepth - 1) * 2]);
    --loopDepth;
  }

  /** Next instruction operand. */
  private int nextOperand() {
    return text[ip++];
  }

  /** Get a value from the register slot. */
  private Object getSlot(int slot) throws EvalException {
    if (slot >= 0) {
      Object value = slots[slot];
      if (value == null) {
        // Now this is always IllegalStateException,
        // but it should be also EvalException when we store locals in registers.
        throw new IllegalStateException("slot value is undefined: " + slot);
      }
      return value;
    } else {
      return compiled.constSlots[BcInstr.constSlotToArrayIndex(slot)];
    }
  }

  /** Get argument with special handling of {@link BcInstr#NULL_REG}. */
  @Nullable
  private Object getSlotOrNull(int slot) throws EvalException {
    return slot != BcInstr.NULL_REG ? getSlot(slot) : null;
  }

  /** Get argument with special handling of {@link BcInstr#NULL_REG}. */
  @Nullable
  private Object getSlotNullAsNone(int slot) throws EvalException {
    return slot != BcInstr.NULL_REG ? getSlot(slot) : Starlark.NONE;
  }

  private void setSlot(int slot, Object value) {
    slots[slot] = value;
  }

  /** AST node associated with current instruction. */
  private Node currentInstrNode() {
    Node node = compiled.instrToNode.get(currentIp);
    Preconditions.checkState(node != null);
    return node;
  }

  private void cp() throws EvalException {
    Object value = getSlot(nextOperand());
    setSlot(nextOperand(), value);
  }

  private void getLocal() throws EvalException {
    String name = compiled.strings[nextOperand()];
    Object value = fr.locals.get(name);
    if (value == null) {
      throw new EvalException(
          String.format(
              "%s variable '%s' is referenced before assignment.", Resolver.Scope.LOCAL, name));
    }
    setSlot(nextOperand(), value);
  }

  private void getGlobal() throws EvalException {
    String name = compiled.strings[nextOperand()];
    Object value = Eval.fn(fr).getModule().getGlobal(name);
    if (value == null) {
      throw new EvalException(
          String.format(
              "%s variable '%s' is referenced before assignment.", Resolver.Scope.GLOBAL, name));
    }
    setSlot(nextOperand(), value);
  }

  private void getPredeclared() throws EvalException {
    String name = compiled.strings[nextOperand()];
    Object value = Eval.fn(fr).getModule().get(name);
    if (value == null) {
      throw new EvalException(
          String.format(
              "%s variable '%s' is referenced before assignment",
              Resolver.Scope.PREDECLARED, name));
    }
    setSlot(nextOperand(), value);
  }

  private void getLegacy() throws EvalException {
    String name = compiled.strings[nextOperand()];
    Object value = fr.getLocals().get(name);
    if (value == null) {
      value = Eval.fn(fr).getModule().get(name);
    }
    if (value == null) {
      // Since Scope was set, we know that the local/global variable is defined,
      // but its assignment was not yet executed.
      throw new EvalException(String.format("variable '%s' is referenced before assignment", name));
    }
    setSlot(nextOperand(), value);
  }

  private void maybeInvokePostAssignHook(String name, Object value, boolean postAssignHook) {
    if (postAssignHook && fr.thread.postAssignHook != null) {
      if (Eval.fn(fr).isToplevel()) {
        fr.thread.postAssignHook.assign(name, value);
      }
    }
  }

  private void setGlobal() throws EvalException {
    Object value = getSlot(nextOperand());
    String name = compiled.strings[nextOperand()];
    boolean postAssignHook = nextOperand() != 0;
    Eval.assignGlobal(fr, name, value);
    maybeInvokePostAssignHook(name, value, postAssignHook);
  }

  private void setLocal() throws EvalException {
    Object value = getSlot(nextOperand());
    String name = compiled.strings[nextOperand()];
    boolean postAssignHook = nextOperand() != 0;
    fr.locals.put(name, value);
    // This is likely not needed, but keeping it because AST interpreter does it
    maybeInvokePostAssignHook(name, value, postAssignHook);
  }

  private void setLegacy() throws EvalException {
    // TODO: scope should be resolved
    if (Eval.fn(fr).isToplevel() && comprSavedLocals.isEmpty()) {
      setGlobal();
    } else {
      setLocal();
    }
  }

  private void stmt() throws EvalException, InterruptedException {
    Statement statement = (Statement) compiled.objects[nextOperand()];
    TokenKind token = Eval.exec(fr, statement);
    Preconditions.checkState(token == TokenKind.PASS);
  }

  private void setIndex() throws EvalException {
    Object dict = getSlot(nextOperand());
    Object key = getSlot(nextOperand());
    Object value = getSlot(nextOperand());
    EvalUtils.setIndex(dict, key, value);
  }

  @SuppressWarnings("unchecked")
  private void listAppend() throws EvalException {
    StarlarkList<Object> list = (StarlarkList<Object>) getSlot(nextOperand());
    Object item = getSlot(nextOperand());
    list.add(item, null);
  }

  private void restoreLocals() {
    String[] localNames = (String[]) compiled.objects[nextOperand()];
    Object[] toRestore = comprSavedLocals.remove(comprSavedLocals.size() - 1);
    for (int i1 = 0; i1 < localNames.length; i1++) {
      String localName = localNames[i1];
      Object value = toRestore[i1];
      if (value != null) {
        fr.locals.put(localName, value);
      } else {
        fr.locals.remove(localName);
      }
    }
  }

  private void saveLocals() {
    String[] localNames = (String[]) compiled.objects[nextOperand()];
    comprSavedLocals.add(Arrays.stream(localNames).map(n -> fr.locals.get(n)).toArray());
  }

  private Object returnInstr() throws EvalException {
    Object result = getSlot(nextOperand());
    validateInstructionDecodedCorrectly();
    return result;
  }

  private Object returnNone() {
    validateInstructionDecodedCorrectly();
    return Starlark.NONE;
  }

  private void br() {
    int dest = nextOperand();
    validateInstructionDecodedCorrectly();
    ip = dest;
  }

  private void ifBr() throws EvalException {
    Object cond = getSlot(nextOperand());
    int dest = nextOperand();
    if (Starlark.truth(cond)) {
      validateInstructionDecodedCorrectly();
      ip = dest;
    } else {
      validateInstructionDecodedCorrectly();
    }
  }

  private void ifNotBr() throws EvalException {
    Object cond = getSlot(nextOperand());
    int dest = nextOperand();
    if (!Starlark.truth(cond)) {
      validateInstructionDecodedCorrectly();
      ip = dest;
    } else {
      validateInstructionDecodedCorrectly();
    }
  }

  private void forInit() throws EvalException {
    Object value = getSlot(nextOperand());
    int nextValueSlot = nextOperand();
    int end = nextOperand();

    Iterable<?> seq = Starlark.toIterable(value);
    Iterator<?> iterator = seq.iterator();
    if (!iterator.hasNext()) {
      validateInstructionDecodedCorrectly();
      ip = end;
      return;
    }

    EvalUtils.addIterator(seq);
    loops[loopDepth * 2] = seq;
    loops[loopDepth * 2 + 1] = iterator;
    ++loopDepth;

    Object item = iterator.next();
    setSlot(nextValueSlot, item);
    validateInstructionDecodedCorrectly();
  }

  private void continueInstr() throws InterruptedException {
    int nextValueSlot = nextOperand();
    int b = nextOperand();
    int e = nextOperand();

    fr.thread.checkInterrupt();

    Iterator<?> iterator = (Iterator<?>) loops[(loopDepth - 1) * 2 + 1];
    if (iterator.hasNext()) {
      setSlot(nextValueSlot, iterator.next());
      validateInstructionDecodedCorrectly();
      ip = b;
    } else {
      popFor();
      validateInstructionDecodedCorrectly();
      ip = e;
    }
  }

  private void breakInstr() {
    int e = nextOperand();
    popFor();
    validateInstructionDecodedCorrectly();
    ip = e;
  }

  private void unpack() throws EvalException {
    Object x = getSlot(nextOperand());
    int nrhs = Starlark.len(x);
    if (nrhs < 0) {
      throw Starlark.errorf("got '%s' in sequence assignment", Starlark.type(x));
    }
    Iterable<?> rhs = Starlark.toIterable(x); // fails if x is a string
    int nlhs = nextOperand();
    if (nrhs != nlhs) {
      throw Starlark.errorf(
          "too %s values to unpack (got %d, want %d)", nrhs < nlhs ? "few" : "many", nrhs, nlhs);
    }
    for (Object item : rhs) {
      setSlot(nextOperand(), item);
    }
  }

  private void list() throws EvalException {
    int size = nextOperand();
    StarlarkList<?> result;
    if (size == 0) {
      result = StarlarkList.newList(fr.thread.mutability());
    } else {
      Object[] data = new Object[size];
      for (int j = 0; j != data.length; ++j) {
        data[j] = getSlot(nextOperand());
      }
      result = StarlarkList.wrap(fr.thread.mutability(), data);
    }
    setSlot(nextOperand(), result);
  }

  private void tuple() throws EvalException {
    int size = nextOperand();
    Tuple<?> result;
    if (size == 0) {
      result = Tuple.empty();
    } else {
      Object[] data = new Object[size];
      for (int j = 0; j != data.length; ++j) {
        data[j] = getSlot(nextOperand());
      }
      result = Tuple.wrap(data);
    }
    setSlot(nextOperand(), result);
  }

  private void dict() throws EvalException {
    int size = nextOperand();
    Dict<?, ?> result;
    if (size == 0) {
      result = Dict.of(fr.thread.mutability());
    } else {
      LinkedHashMap<Object, Object> lhm = new LinkedHashMap<>(size);
      for (int j = 0; j != size; ++j) {
        Object key = getSlot(nextOperand());
        EvalUtils.checkHashable(key);
        Object value = getSlot(nextOperand());
        Object prev = lhm.put(key, value);
        if (prev != null) {
          throw new EvalException(
              "Duplicated key " + Starlark.repr(key) + " when creating dictionary");
        }
      }
      result = Dict.wrap(fr.thread.mutability(), lhm);
    }
    setSlot(nextOperand(), result);
  }

  /** Dot operator. */
  private void dot() throws EvalException, InterruptedException {
    Object object = getSlot(nextOperand());
    String name = compiled.strings[nextOperand()];
    Object result = EvalUtils.getAttr(fr.thread, object, name);
    if (result == null) {
      throw EvalUtils.getMissingAttrException(object, name, fr.thread.getSemantics());
    }
    setSlot(nextOperand(), result);
  }

  /** Index operator. */
  private void index() throws EvalException {
    Object object = getSlot(nextOperand());
    Object index = getSlot(nextOperand());
    setSlot(
        nextOperand(),
        EvalUtils.index(fr.thread.mutability(), fr.thread.getSemantics(), object, index));
  }

  /** Slice operator. */
  private void slice() throws EvalException {
    Object object = getSlot(nextOperand());
    Object start = getSlotNullAsNone(nextOperand());
    Object stop = getSlotNullAsNone(nextOperand());
    Object step = getSlotNullAsNone(nextOperand());
    setSlot(nextOperand(), Starlark.slice(fr.thread.mutability(), object, start, stop, step));
  }

  /** Call operator. */
  private void call() throws EvalException, InterruptedException {
    fr.thread.checkInterrupt();

    Location lparenLocation = (Location) compiled.objects[nextOperand()];
    fr.setLocation(lparenLocation);

    Object fn = getSlot(nextOperand());
    int npos = nextOperand();
    Object[] pos = npos != 0 ? new Object[npos] : EMPTY;
    for (int i = 0; i < npos; ++i) {
      pos[i] = getSlot(nextOperand());
    }
    int nnamed = nextOperand();
    Object[] named = nnamed != 0 ? new Object[nnamed * 2] : EMPTY;
    for (int i = 0; i < nnamed; ++i) {
      named[i * 2] = compiled.strings[nextOperand()];
      named[i * 2 + 1] = getSlot(nextOperand());
    }
    Object star = getSlotOrNull(nextOperand());
    int starOffset = nextOperand();
    Object starStar = getSlotOrNull(nextOperand());
    int starStarOffset = nextOperand();

    if (star != null) {
      if (!(star instanceof StarlarkIterable)) {
        throw new EvalException(
            compiled.locs.getLocation(starOffset),
            "argument after * must be an iterable, not " + Starlark.type(star));
      }
      Iterable<?> iter = (Iterable<?>) star;

      // TODO(adonovan): opt: if value.size is known, preallocate (and skip if empty).
      ArrayList<Object> list = new ArrayList<>();
      Collections.addAll(list, pos);
      Iterables.addAll(list, iter);
      pos = list.toArray();
    }

    if (starStar != null) {
      if (!(starStar instanceof Dict)) {
        throw new EvalException(
            compiled.locs.getLocation(starStarOffset),
            "argument after ** must be a dict, not " + Starlark.type(starStar));
      }
      Dict<?, ?> dict = (Dict<?, ?>) starStar;

      int j = named.length;
      named = Arrays.copyOf(named, j + 2 * dict.size());
      for (Map.Entry<?, ?> e : dict.entrySet()) {
        if (!(e.getKey() instanceof String)) {
          throw new EvalException(
              compiled.locs.getLocation(starStarOffset),
              "keywords must be strings, not " + Starlark.type(e.getKey()));
        }
        named[j++] = e.getKey();
        named[j++] = e.getValue();
      }
    }

    try {
      setSlot(nextOperand(), Starlark.fastcall(fr.thread, fn, pos, named));
    } catch (EvalExceptionWithStackTrace e) {
      e.registerNode(currentInstrNode());
      throw e;
    } catch (EvalException e) {
      if (e.canBeAddedToStackTrace()) {
        throw new EvalExceptionWithStackTrace(e, currentInstrNode());
      } else {
        throw e;
      }
    }
  }

  /** Not operator. */
  private void not() throws EvalException {
    Object value = getSlot(nextOperand());
    setSlot(nextOperand(), !Starlark.truth(value));
  }

  /** Generic unary operator. */
  private void unary() throws EvalException {
    Object value = getSlot(nextOperand());
    TokenKind op = TOKENS[nextOperand()];
    setSlot(nextOperand(), EvalUtils.unaryOp(op, value));
  }

  /**
   * Generic binary operator
   *
   * <p>Note that {@code and} and {@code or} are not emitted as binary operator instruction. .
   */
  private void binary() throws EvalException {
    Object x = getSlot(nextOperand());
    Object y = getSlot(nextOperand());
    TokenKind op = TOKENS[nextOperand()];
    setSlot(
        nextOperand(),
        EvalUtils.binaryOp(op, x, y, fr.thread.getSemantics(), fr.thread.mutability()));
  }

  /**
   * Generic binary operator
   *
   * <p>Note that {@code and} and {@code or} are not emitted as binary operator instruction. .
   */
  private void binaryInPlace() throws EvalException {
    Object x = getSlot(nextOperand());
    Object y = getSlot(nextOperand());
    TokenKind op = TOKENS[nextOperand()];
    setSlot(nextOperand(), Eval.inplaceBinaryOp(fr, op, x, y));
  }

  /** Equality. */
  private void eq() throws EvalException {
    Object lhs = getSlot(nextOperand());
    Object rhs = getSlot(nextOperand());
    setSlot(nextOperand(), lhs.equals(rhs));
  }

  /** Equality. */
  private void notEq() throws EvalException {
    Object lhs = getSlot(nextOperand());
    Object rhs = getSlot(nextOperand());
    setSlot(nextOperand(), !lhs.equals(rhs));
  }

  private void evalException() throws EvalException {
    String message = compiled.strings[nextOperand()];
    validateInstructionDecodedCorrectly();
    throw new EvalException(message);
  }

  private void dbg() {
    if (fr.dbg != null) {
      Location loc = currentInstrNode().getStartLocation(); // not very precise
      fr.setLocation(loc);
      fr.dbg.before(fr.thread, loc); // location is now redundant since it's in the thread
    }
  }

  private EvalException otherOpcode(int opcode) {
    if (opcode < BcInstr.Opcode.values().length) {
      throw new IllegalStateException("not implemented opcode: " + BcInstr.Opcode.values()[opcode]);
    } else {
      throw new IllegalStateException("wrong opcode: " + opcode);
    }
  }

  private void validateInstructionDecodedCorrectly() {
    if (Bc.ASSERTIONS) {
      // Validate the last instruction was decoded correctly
      // (got all the argument, and no extra arguments).
      // This is quite helpful, but expensive assertion, only enabled when bytecode assertions
      // are on.
      BcInstr.Opcode opcode = BcInstr.Opcode.values()[text[currentIp]];
      int prevInstrLen = compiled.instrLenAt(currentIp);
      Preconditions.checkState(
          ip == currentIp + prevInstrLen,
          "Instruction %s incorrectly handled len; expected len: %s, actual len: %s",
          opcode,
          prevInstrLen,
          ip - currentIp);
    }
  }
}
