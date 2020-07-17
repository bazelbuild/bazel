package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;

/** Instructions for the bytecode interpreter. */
class BcInstr {

  private BcInstr() {}

  /** Undefined file offset. */
  static final int UNDEFINED_LOC = -19;

  /** Special {@code null} register value, interpretation depends on opcode. */
  static final int NULL_REG = Integer.MIN_VALUE;

  /** Constants are stored in a separate array. Constant slots are negative integers. */
  static int constSlotToArrayIndex(int slot) {
    Preconditions.checkState(slot < 0);
    return -1 - slot;
  }

  /** Constants are stored in a separate array. Constant slots are negative integers. */
  static int constSlotFromArrayIndex(int arrayIndex) {
    Preconditions.checkState(arrayIndex >= 0);
    return -1 - arrayIndex;
  }

  // The instruction header is an opcode.
  static final int INSTR_HEADER_LEN = 1;

  // We assign integer constants explicitly instead of using enum for performance:
  // our bytecode stores integers, and converting each opcode to enum might be expensive.

  static final int CP = 0;
  static final int EQ = 1;
  static final int NOT_EQ = 2;
  static final int NOT = 3;
  static final int UNARY = 4;
  static final int BR = 5;
  static final int IF_BR = 6;
  static final int IF_NOT_BR = 7;
  static final int BINARY = 8;
  static final int BINARY_IN_PLACE = 9;
  static final int GET_LOCAL = 10;
  static final int GET_GLOBAL = 11;
  static final int GET_PREDECLARED = 12;
  static final int GET_LEGACY = 13;
  static final int SET_LOCAL = 14;
  static final int SET_GLOBAL = 15;
  static final int SET_LEGACY = 16;
  static final int DOT = 17;
  static final int INDEX = 18;
  static final int SLICE = 19;
  static final int CALL = 20;
  static final int RETURN = 21;
  static final int RETURN_NONE = 22;
  static final int FOR_INIT = 23;
  static final int CONTINUE = 24;
  static final int BREAK = 25;
  static final int LIST = 26;
  static final int TUPLE = 27;
  static final int DICT = 28;
  static final int STMT = 29;
  static final int SAVE_LOCALS = 30;
  static final int RESTORE_LOCALS = 31;
  static final int LIST_APPEND = 32;
  static final int SET_INDEX = 33;
  static final int EVAL_EXCEPTION = 34;
  static final int UNPACK = 35;
  static final int DBG = 36;

  /**
   * Opcodes as enum. We use enums in the compiler, but we use only raw integers in the interpreter.
   *
   * <p>Enums are much nicer to work with, but they are much more expensive. Thus we use enums only
   * in the compiler, or during debugging.
   */
  enum Opcode {
    /** {@code a1 = a0}. */
    CP(BcInstr.CP, BcInstrOperand.IN_SLOT, BcInstrOperand.OUT_SLOT),
    /**
     * {@code a2 = a0 == a1}. This is quite common operation, which deserves its own opcode to avoid
     * switching in generic binary operator handling.
     */
    EQ(BcInstr.EQ, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT, BcInstrOperand.OUT_SLOT),
    /**
     * {@code a2 = a0 != a1}. This is quite common operation, which deserves its own opcode to avoid
     * switching in generic binary operator handling.
     */
    NOT_EQ(BcInstr.NOT_EQ, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT, BcInstrOperand.OUT_SLOT),
    /**
     * {@code a1 = not a0}.
     *
     * <p>This could be handled by generic UNARY opcode, but it is specialized for performance.
     */
    NOT(BcInstr.NOT, BcInstrOperand.IN_SLOT, BcInstrOperand.OUT_SLOT),
    /** {@code a2 = (a1) a0}. */
    UNARY(
        BcInstr.UNARY, BcInstrOperand.IN_SLOT, BcInstrOperand.TOKEN_KIND, BcInstrOperand.OUT_SLOT),
    /** Goto. */
    BR(BcInstr.BR, BcInstrOperand.addr("j")),
    /** Goto if. */
    IF_BR(BcInstr.IF_BR, BcInstrOperand.IN_SLOT, BcInstrOperand.addr("t")),
    /** Goto if not. */
    IF_NOT_BR(BcInstr.IF_NOT_BR, BcInstrOperand.IN_SLOT, BcInstrOperand.addr("f")),
    /** {@code a3 = a0 (a2) a1}. */
    BINARY(
        BcInstr.BINARY,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.TOKEN_KIND,
        BcInstrOperand.OUT_SLOT),
    /** {@code a3 = a0 (a2)= a1}. */
    BINARY_IN_PLACE(
        BcInstr.BINARY_IN_PLACE,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.TOKEN_KIND,
        BcInstrOperand.OUT_SLOT),
    /** Get a local variable and store it in a given register. */
    GET_LOCAL(BcInstr.GET_LOCAL, BcInstrOperand.STRING, BcInstrOperand.OUT_SLOT),
    /** Get a global variable and store it in a given register. */
    GET_GLOBAL(BcInstr.GET_GLOBAL, BcInstrOperand.STRING, BcInstrOperand.OUT_SLOT),
    /** Get a predeclared variable and store it in a given register. */
    GET_PREDECLARED(BcInstr.GET_PREDECLARED, BcInstrOperand.STRING, BcInstrOperand.OUT_SLOT),
    /** Deprecated way to get a variable from any scope. */
    GET_LEGACY(BcInstr.GET_LEGACY, BcInstrOperand.STRING, BcInstrOperand.OUT_SLOT),
    /** Assign a value without destructuring to a local variable. */
    SET_LOCAL(BcInstr.SET_LOCAL,
      // value
      BcInstrOperand.IN_SLOT,
      // name
      BcInstrOperand.STRING,
      // 1 if need to invoke post-assign hook, 0 otherwise
      BcInstrOperand.NUMBER),
    /** Assign a value without destructuring to a global variable. */
    SET_GLOBAL(
        BcInstr.SET_GLOBAL,
        // value
        BcInstrOperand.IN_SLOT,
        // name
        BcInstrOperand.STRING,
        // 1 if need to invoke post-assign hook, 0 otherwise
        BcInstrOperand.NUMBER),
    SET_LEGACY(
        BcInstr.SET_LEGACY,
        // value
        BcInstrOperand.IN_SLOT,
        // name
        BcInstrOperand.STRING,
        // 1 if need to invoke post-assign hook, 0 otherwise
        BcInstrOperand.NUMBER),
    /** {@code a2 = a0.a1} */
    DOT(BcInstr.DOT, BcInstrOperand.IN_SLOT, BcInstrOperand.STRING, BcInstrOperand.OUT_SLOT),
    /** {@code a2 = a0[a1]} */
    INDEX(BcInstr.INDEX, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT, BcInstrOperand.OUT_SLOT),
    /** {@code a4 = a0[a1:a2:a3]} */
    SLICE(
        BcInstr.SLICE,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.OUT_SLOT),
    /** Generic call invocation. */
    CALL(
        BcInstr.CALL,
        // prematerialized LParen location
        BcInstrOperand.OBJECT,
        // Function
        BcInstrOperand.IN_SLOT,
        // Positional arguments
        BcInstrOperand.lengthDelimited(BcInstrOperand.IN_SLOT),
        // Named arguments
        BcInstrOperand.lengthDelimited(
            BcInstrOperand.fixed(BcInstrOperand.STRING, BcInstrOperand.IN_SLOT)),
        // *args
        BcInstrOperand.IN_SLOT,
        // *args location
        BcInstrOperand.NUMBER,
        // **kwargs
        BcInstrOperand.IN_SLOT,
        // **kwargs location
        BcInstrOperand.NUMBER,
        // Where to store result
        BcInstrOperand.OUT_SLOT),
    /** {@code return a0} */
    RETURN(BcInstr.RETURN, BcInstrOperand.IN_SLOT),
    /** {@code return None} */
    RETURN_NONE(BcInstr.RETURN_NONE),
    /**
     * For loop init:
     *
     * <ul>
     *   <li>Check if operand is iterable
     *   <li>Lock the iterable
     *   <li>Create an iterator
     *   <li>If iterator has no elements, go to "e".
     *   <li>Otherwise push iterable and iterator onto the stack
     *   <li>Fetch the first element of the iterator and store it in the provided register
     * </ul>
     */
    FOR_INIT(
        BcInstr.FOR_INIT,
        // Collection parameter
        BcInstrOperand.IN_SLOT,
        // Next value register
        BcInstrOperand.OUT_SLOT,
        BcInstrOperand.addr("e")),
    /**
     * Continue the loop:
     *
     * <ul>
     *   <li>If current iterator (stored on the stack) is empty, unlock the iterable and pop
     *       iterable and iterable from the stack and go to the label "e" after the end of the loop.
     *   <li>Otherwise assign the next iterator item to the provided register and go to the label
     *       "b", loop body.
     * </ul>
     */
    CONTINUE(
        BcInstr.CONTINUE,
        // Iterator next value.
        BcInstrOperand.OUT_SLOT,
        // Beginning of the loop
        BcInstrOperand.addr("b"),
        // End of the loop
        BcInstrOperand.addr("e")),
    /**
     * Exit the loop: unlock the iterable, pop it from the loop stack and goto a label after the
     * loop.
     */
    BREAK(BcInstr.BREAK, BcInstrOperand.addr("e")),
    /** List constructor. */
    LIST(
        BcInstr.LIST,
        // List size followed by list items.
        BcInstrOperand.lengthDelimited(BcInstrOperand.IN_SLOT),
        BcInstrOperand.OUT_SLOT),
    /** Tuple constructor; similar to the list constructor above. */
    TUPLE(
        BcInstr.TUPLE,
        BcInstrOperand.lengthDelimited(BcInstrOperand.IN_SLOT),
        BcInstrOperand.OUT_SLOT),
    /** Dict constructor. */
    DICT(
        BcInstr.DICT,
        BcInstrOperand.lengthDelimited(
            BcInstrOperand.fixed(BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT)),
        BcInstrOperand.OUT_SLOT),
    /**
     * Invoke a statement using old AST interpreter.
     *
     * <p>This is used only to implement def and load statements since they don't interfere with the
     * rest of the bytecode interpreter. Statements like if or for must not be encoded using this
     * opcode.
     */
    // TODO: implement opcodes for def and load
    STMT(BcInstr.STMT, BcInstrOperand.OBJECT),
    /** Save locals before invoking a comprehension. */
    SAVE_LOCALS(BcInstr.SAVE_LOCALS, BcInstrOperand.OBJECT),
    /** Restore locals after returning from a comprehension. */
    RESTORE_LOCALS(BcInstr.RESTORE_LOCALS, BcInstrOperand.OBJECT),
    /** {@code a0.append(a1)}. */
    LIST_APPEND(BcInstr.LIST_APPEND, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT),
    /** {@code a0[a1] = a2}. */
    SET_INDEX(
        BcInstr.SET_INDEX, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT, BcInstrOperand.IN_SLOT),
    /** Throw an {@code EvalException} on execution of this instruction. */
    EVAL_EXCEPTION(BcInstr.EVAL_EXCEPTION, BcInstrOperand.STRING),
    /** {@code (a1[0], a1[1], a1[2], ...) = a0}. */
    UNPACK(
        BcInstr.UNPACK,
        BcInstrOperand.IN_SLOT,
        BcInstrOperand.lengthDelimited(BcInstrOperand.OUT_SLOT)),
    /** Debugger callback. */
    DBG(BcInstr.DBG);
    ;

    /** Type of opcode operands. */
    final BcInstrOperand.Operands operands;

    Opcode(int opcode, BcInstrOperand.Operands... operands) {
      this(opcode, operands.length != 1 ? BcInstrOperand.fixed(operands) : operands[0]);
    }

    Opcode(int opcode, BcInstrOperand.Operands operands) {
      // We maintain the invariant: the opcode is equal to enum variant ordinal.
      // It is a bit inconvenient to maintain, but make is much easier/safer to work with.
      Preconditions.checkState(
          opcode == ordinal(),
          String.format("wrong index for %s: expected %s, actual %s", name(), ordinal(), opcode));
      this.operands = operands;
    }
  }
}
