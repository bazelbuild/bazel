/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.classfile.instruction;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.instruction.visitor.InstructionVisitor;

/**
 * Base class for representing instructions.
 *
 * @author Eric Lafortune
 */
public abstract class Instruction
{
    // An array for marking instructions that may throw exceptions.
    private static final boolean[] MAY_THROW_EXCEPTIONS = new boolean[]
    {
        false, // nop
        false, // aconst_null
        false, // iconst_m1
        false, // iconst_0
        false, // iconst_1
        false, // iconst_2
        false, // iconst_3
        false, // iconst_4
        false, // iconst_5
        false, // lconst_0
        false, // lconst_1
        false, // fconst_0
        false, // fconst_1
        false, // fconst_2
        false, // dconst_0
        false, // dconst_1
        false, // bipush
        false, // sipush
        false, // ldc
        false, // ldc_w
        false, // ldc2_w
        false, // iload
        false, // lload
        false, // fload
        false, // dload
        false, // aload
        false, // iload_0
        false, // iload_1
        false, // iload_2
        false, // iload_3
        false, // lload_0
        false, // lload_1
        false, // lload_2
        false, // lload_3
        false, // fload_0
        false, // fload_1
        false, // fload_2
        false, // fload_3
        false, // dload_0
        false, // dload_1
        false, // dload_2
        false, // dload_3
        false, // aload_0
        false, // aload_1
        false, // aload_2
        false, // aload_3
        true,  // iaload
        true,  // laload
        true,  // faload
        true,  // daload
        true,  // aaload
        true,  // baload
        true,  // caload
        true,  // saload
        false, // istore
        false, // lstore
        false, // fstore
        false, // dstore
        false, // astore
        false, // istore_0
        false, // istore_1
        false, // istore_2
        false, // istore_3
        false, // lstore_0
        false, // lstore_1
        false, // lstore_2
        false, // lstore_3
        false, // fstore_0
        false, // fstore_1
        false, // fstore_2
        false, // fstore_3
        false, // dstore_0
        false, // dstore_1
        false, // dstore_2
        false, // dstore_3
        false, // astore_0
        false, // astore_1
        false, // astore_2
        false, // astore_3
        true,  // iastore
        true,  // lastore
        true,  // fastore
        true,  // dastore
        true,  // aastore
        true,  // bastore
        true,  // castore
        true,  // sastore
        false, // pop
        false, // pop2
        false, // dup
        false, // dup_x1
        false, // dup_x2
        false, // dup2
        false, // dup2_x1
        false, // dup2_x2
        false, // swap
        false, // iadd
        false, // ladd
        false, // fadd
        false, // dadd
        false, // isub
        false, // lsub
        false, // fsub
        false, // dsub
        false, // imul
        false, // lmul
        false, // fmul
        false, // dmul
        true,  // idiv
        true,  // ldiv
        false, // fdiv
        false, // ddiv
        true,  // irem
        true,  // lrem
        false, // frem
        false, // drem
        false, // ineg
        false, // lneg
        false, // fneg
        false, // dneg
        false, // ishl
        false, // lshl
        false, // ishr
        false, // lshr
        false, // iushr
        false, // lushr
        false, // iand
        false, // land
        false, // ior
        false, // lor
        false, // ixor
        false, // lxor
        false, // iinc
        false, // i2l
        false, // i2f
        false, // i2d
        false, // l2i
        false, // l2f
        false, // l2d
        false, // f2i
        false, // f2l
        false, // f2d
        false, // d2i
        false, // d2l
        false, // d2f
        false, // i2b
        false, // i2c
        false, // i2s
        false, // lcmp
        false, // fcmpl
        false, // fcmpg
        false, // dcmpl
        false, // dcmpg
        false, // ifeq
        false, // ifne
        false, // iflt
        false, // ifge
        false, // ifgt
        false, // ifle
        false, // ificmpeq
        false, // ificmpne
        false, // ificmplt
        false, // ificmpge
        false, // ificmpgt
        false, // ificmple
        false, // ifacmpeq
        false, // ifacmpne
        false, // goto
        false, // jsr
        false, // ret
        false, // tableswitch
        false, // lookupswitch
        false, // ireturn
        false, // lreturn
        false, // freturn
        false, // dreturn
        false, // areturn
        false, // return
        true,  // getstatic
        true,  // putstatic
        true,  // getfield
        true,  // putfield
        true,  // invokevirtual
        true,  // invokespecial
        true,  // invokestatic
        true,  // invokeinterface
        true,  // invokedynamic
        true,  // new
        true,  // newarray
        true,  // anewarray
        true,  // arraylength
        true,  // athrow
        true,  // checkcast
        false, // instanceof
        true,  // monitorenter
        true,  // monitorexit
        false, // wide
        true,  // multianewarray
        false, // ifnull
        false, // ifnonnull
        false, // goto_w
        false, // jsr_w
    };


    // An array for marking Category 2 instructions.
    private static final boolean[] IS_CATEGORY2 = new boolean[]
    {
        false, // nop
        false, // aconst_null
        false, // iconst_m1
        false, // iconst_0
        false, // iconst_1
        false, // iconst_2
        false, // iconst_3
        false, // iconst_4
        false, // iconst_5
        true,  // lconst_0
        true,  // lconst_1
        false, // fconst_0
        false, // fconst_1
        false, // fconst_2
        true,  // dconst_0
        true,  // dconst_1
        false, // bipush
        false, // sipush
        false, // ldc
        false, // ldc_w
        true,  // ldc2_w
        false, // iload
        true,  // lload
        false, // fload
        true,  // dload
        false, // aload
        false, // iload_0
        false, // iload_1
        false, // iload_2
        false, // iload_3
        true,  // lload_0
        true,  // lload_1
        true,  // lload_2
        true,  // lload_3
        false, // fload_0
        false, // fload_1
        false, // fload_2
        false, // fload_3
        true,  // dload_0
        true,  // dload_1
        true,  // dload_2
        true,  // dload_3
        false, // aload_0
        false, // aload_1
        false, // aload_2
        false, // aload_3
        false, // iaload
        true,  // laload
        false, // faload
        true,  // daload
        false, // aaload
        false, // baload
        false, // caload
        false, // saload
        false, // istore
        true,  // lstore
        false, // fstore
        true,  // dstore
        false, // astore
        false, // istore_0
        false, // istore_1
        false, // istore_2
        false, // istore_3
        true,  // lstore_0
        true,  // lstore_1
        true,  // lstore_2
        true,  // lstore_3
        false, // fstore_0
        false, // fstore_1
        false, // fstore_2
        false, // fstore_3
        true,  // dstore_0
        true,  // dstore_1
        true,  // dstore_2
        true,  // dstore_3
        false, // astore_0
        false, // astore_1
        false, // astore_2
        false, // astore_3
        false, // iastore
        true,  // lastore
        false, // fastore
        true,  // dastore
        false, // aastore
        false, // bastore
        false, // castore
        false, // sastore
        false, // pop
        true,  // pop2
        false, // dup
        false, // dup_x1
        false, // dup_x2
        true,  // dup2
        true,  // dup2_x1
        true,  // dup2_x2
        false, // swap
        false, // iadd
        true,  // ladd
        false, // fadd
        true,  // dadd
        false, // isub
        true,  // lsub
        false, // fsub
        true,  // dsub
        false, // imul
        true,  // lmul
        false, // fmul
        true,  // dmul
        false, // idiv
        true,  // ldiv
        false, // fdiv
        true,  // ddiv
        false, // irem
        true,  // lrem
        false, // frem
        true,  // drem
        false, // ineg
        true,  // lneg
        false, // fneg
        true,  // dneg
        false, // ishl
        true,  // lshl
        false, // ishr
        true,  // lshr
        false, // iushr
        true,  // lushr
        false, // iand
        true,  // land
        false, // ior
        true,  // lor
        false, // ixor
        true,  // lxor
        false, // iinc
        false, // i2l
        false, // i2f
        false, // i2d
        true,  // l2i
        true,  // l2f
        true,  // l2d
        false, // f2i
        false, // f2l
        false, // f2d
        true,  // d2i
        true,  // d2l
        true,  // d2f
        false, // i2b
        false, // i2c
        false, // i2s
        true,  // lcmp
        false, // fcmpl
        false, // fcmpg
        true,  // dcmpl
        true,  // dcmpg
        false, // ifeq
        false, // ifne
        false, // iflt
        false, // ifge
        false, // ifgt
        false, // ifle
        false, // ificmpeq
        false, // ificmpne
        false, // ificmplt
        false, // ificmpge
        false, // ificmpgt
        false, // ificmple
        false, // ifacmpeq
        false, // ifacmpne
        false, // goto
        false, // jsr
        false, // ret
        false, // tableswitch
        false, // lookupswitch
        false, // ireturn
        true,  // lreturn
        false, // freturn
        true,  // dreturn
        false, // areturn
        false, // return
        false, // getstatic
        false, // putstatic
        false, // getfield
        false, // putfield
        false, // invokevirtual
        false, // invokespecial
        false, // invokestatic
        false, // invokeinterface
        false, // invokedynamic
        false, // new
        false, // newarray
        false, // anewarray
        false, // arraylength
        false, // athrow
        false, // checkcast
        false, // instanceof
        false, // monitorenter
        false, // monitorexit
        false, // wide
        false, // multianewarray
        false, // ifnull
        false, // ifnonnull
        false, // goto_w
        false, // jsr_w
    };


    // An array containing the fixed number of entries popped from the stack,
    // for all instructions.
    private static final int[] STACK_POP_COUNTS = new int[]
    {
        0, // nop
        0, // aconst_null
        0, // iconst_m1
        0, // iconst_0
        0, // iconst_1
        0, // iconst_2
        0, // iconst_3
        0, // iconst_4
        0, // iconst_5
        0, // lconst_0
        0, // lconst_1
        0, // fconst_0
        0, // fconst_1
        0, // fconst_2
        0, // dconst_0
        0, // dconst_1
        0, // bipush
        0, // sipush
        0, // ldc
        0, // ldc_w
        0, // ldc2_w
        0, // iload
        0, // lload
        0, // fload
        0, // dload
        0, // aload
        0, // iload_0
        0, // iload_1
        0, // iload_2
        0, // iload_3
        0, // lload_0
        0, // lload_1
        0, // lload_2
        0, // lload_3
        0, // fload_0
        0, // fload_1
        0, // fload_2
        0, // fload_3
        0, // dload_0
        0, // dload_1
        0, // dload_2
        0, // dload_3
        0, // aload_0
        0, // aload_1
        0, // aload_2
        0, // aload_3
        2, // iaload
        2, // laload
        2, // faload
        2, // daload
        2, // aaload
        2, // baload
        2, // caload
        2, // saload
        1, // istore
        2, // lstore
        1, // fstore
        2, // dstore
        1, // astore
        1, // istore_0
        1, // istore_1
        1, // istore_2
        1, // istore_3
        2, // lstore_0
        2, // lstore_1
        2, // lstore_2
        2, // lstore_3
        1, // fstore_0
        1, // fstore_1
        1, // fstore_2
        1, // fstore_3
        2, // dstore_0
        2, // dstore_1
        2, // dstore_2
        2, // dstore_3
        1, // astore_0
        1, // astore_1
        1, // astore_2
        1, // astore_3
        3, // iastore
        4, // lastore
        3, // fastore
        4, // dastore
        3, // aastore
        3, // bastore
        3, // castore
        3, // sastore
        1, // pop
        2, // pop2
        1, // dup
        2, // dup_x1
        3, // dup_x2
        2, // dup2
        3, // dup2_x1
        4, // dup2_x2
        2, // swap
        2, // iadd
        4, // ladd
        2, // fadd
        4, // dadd
        2, // isub
        4, // lsub
        2, // fsub
        4, // dsub
        2, // imul
        4, // lmul
        2, // fmul
        4, // dmul
        2, // idiv
        4, // ldiv
        2, // fdiv
        4, // ddiv
        2, // irem
        4, // lrem
        2, // frem
        4, // drem
        1, // ineg
        2, // lneg
        1, // fneg
        2, // dneg
        2, // ishl
        3, // lshl
        2, // ishr
        3, // lshr
        2, // iushr
        3, // lushr
        2, // iand
        4, // land
        2, // ior
        4, // lor
        2, // ixor
        4, // lxor
        0, // iinc
        1, // i2l
        1, // i2f
        1, // i2d
        2, // l2i
        2, // l2f
        2, // l2d
        1, // f2i
        1, // f2l
        1, // f2d
        2, // d2i
        2, // d2l
        2, // d2f
        1, // i2b
        1, // i2c
        1, // i2s
        4, // lcmp
        2, // fcmpl
        2, // fcmpg
        4, // dcmpl
        4, // dcmpg
        1, // ifeq
        1, // ifne
        1, // iflt
        1, // ifge
        1, // ifgt
        1, // ifle
        2, // ificmpeq
        2, // ificmpne
        2, // ificmplt
        2, // ificmpge
        2, // ificmpgt
        2, // ificmple
        2, // ifacmpeq
        2, // ifacmpne
        0, // goto
        0, // jsr
        0, // ret
        1, // tableswitch
        1, // lookupswitch
        1, // ireturn
        2, // lreturn
        1, // freturn
        2, // dreturn
        1, // areturn
        0, // return
        0, // getstatic
        0, // putstatic
        1, // getfield
        1, // putfield
        1, // invokevirtual
        1, // invokespecial
        0, // invokestatic
        1, // invokeinterface
        0, // invokedynamic
        0, // new
        1, // newarray
        1, // anewarray
        1, // arraylength
        1, // athrow
        1, // checkcast
        1, // instanceof
        1, // monitorenter
        1, // monitorexit
        0, // wide
        0, // multianewarray
        1, // ifnull
        1, // ifnonnull
        0, // goto_w
        0, // jsr_w
    };


    // An array containing the fixed number of entries pushed onto the stack,
    // for all instructions.
    private static final int[] STACK_PUSH_COUNTS = new int[]
    {
        0, // nop
        1, // aconst_null
        1, // iconst_m1
        1, // iconst_0
        1, // iconst_1
        1, // iconst_2
        1, // iconst_3
        1, // iconst_4
        1, // iconst_5
        2, // lconst_0
        2, // lconst_1
        1, // fconst_0
        1, // fconst_1
        1, // fconst_2
        2, // dconst_0
        2, // dconst_1
        1, // bipush
        1, // sipush
        1, // ldc
        1, // ldc_w
        2, // ldc2_w
        1, // iload
        2, // lload
        1, // fload
        2, // dload
        1, // aload
        1, // iload_0
        1, // iload_1
        1, // iload_2
        1, // iload_3
        2, // lload_0
        2, // lload_1
        2, // lload_2
        2, // lload_3
        1, // fload_0
        1, // fload_1
        1, // fload_2
        1, // fload_3
        2, // dload_0
        2, // dload_1
        2, // dload_2
        2, // dload_3
        1, // aload_0
        1, // aload_1
        1, // aload_2
        1, // aload_3
        1, // iaload
        2, // laload
        1, // faload
        2, // daload
        1, // aaload
        1, // baload
        1, // caload
        1, // saload
        0, // istore
        0, // lstore
        0, // fstore
        0, // dstore
        0, // astore
        0, // istore_0
        0, // istore_1
        0, // istore_2
        0, // istore_3
        0, // lstore_0
        0, // lstore_1
        0, // lstore_2
        0, // lstore_3
        0, // fstore_0
        0, // fstore_1
        0, // fstore_2
        0, // fstore_3
        0, // dstore_0
        0, // dstore_1
        0, // dstore_2
        0, // dstore_3
        0, // astore_0
        0, // astore_1
        0, // astore_2
        0, // astore_3
        0, // iastore
        0, // lastore
        0, // fastore
        0, // dastore
        0, // aastore
        0, // bastore
        0, // castore
        0, // sastore
        0, // pop
        0, // pop2
        2, // dup
        3, // dup_x1
        4, // dup_x2
        4, // dup2
        5, // dup2_x1
        6, // dup2_x2
        2, // swap
        1, // iadd
        2, // ladd
        1, // fadd
        2, // dadd
        1, // isub
        2, // lsub
        1, // fsub
        2, // dsub
        1, // imul
        2, // lmul
        1, // fmul
        2, // dmul
        1, // idiv
        2, // ldiv
        1, // fdiv
        2, // ddiv
        1, // irem
        2, // lrem
        1, // frem
        2, // drem
        1, // ineg
        2, // lneg
        1, // fneg
        2, // dneg
        1, // ishl
        2, // lshl
        1, // ishr
        2, // lshr
        1, // iushr
        2, // lushr
        1, // iand
        2, // land
        1, // ior
        2, // lor
        1, // ixor
        2, // lxor
        0, // iinc
        2, // i2l
        1, // i2f
        2, // i2d
        1, // l2i
        1, // l2f
        2, // l2d
        1, // f2i
        2, // f2l
        2, // f2d
        1, // d2i
        2, // d2l
        1, // d2f
        1, // i2b
        1, // i2c
        1, // i2s
        1, // lcmp
        1, // fcmpl
        1, // fcmpg
        1, // dcmpl
        1, // dcmpg
        0, // ifeq
        0, // ifne
        0, // iflt
        0, // ifge
        0, // ifgt
        0, // ifle
        0, // ificmpeq
        0, // ificmpne
        0, // ificmplt
        0, // ificmpge
        0, // ificmpgt
        0, // ificmple
        0, // ifacmpeq
        0, // ifacmpne
        0, // goto
        1, // jsr
        0, // ret
        0, // tableswitch
        0, // lookupswitch
        0, // ireturn
        0, // lreturn
        0, // freturn
        0, // dreturn
        0, // areturn
        0, // return
        0, // getstatic
        0, // putstatic
        0, // getfield
        0, // putfield
        0, // invokevirtual
        0, // invokespecial
        0, // invokestatic
        0, // invokeinterface
        0, // invokedynamic
        1, // new
        1, // newarray
        1, // anewarray
        1, // arraylength
        0, // athrow
        1, // checkcast
        1, // instanceof
        0, // monitorenter
        0, // monitorexit
        0, // wide
        1, // multianewarray
        0, // ifnull
        0, // ifnonnull
        0, // goto_w
        1, // jsr_w
    };


    public byte opcode;


    /**
     * Returns the canonical opcode of this instruction, i.e. typically the
     * opcode whose extension has been removed.
     */
    public byte canonicalOpcode()
    {
        return opcode;
    }


    /**
     * Returns the actual opcode of this instruction, i.e. the opcode that is
     * stored in the bytecode.
     */
    public byte actualOpcode()
    {
        return opcode;
    }


    /**
     * Shrinks this instruction to its shortest possible form.
     * @return this instruction.
     */
    public abstract Instruction shrink();



    /**
     * Writes the Instruction at the given offset in the given code attribute.
     */
    public final void write(CodeAttribute codeAttribute, int offset)
    {
        write(codeAttribute.code, offset);
    }


    /**
     * Writes the Instruction at the given offset in the given code array.
     */
    public void write(byte[] code, int offset)
    {
        // Write the wide opcode, if necessary.
        if (isWide())
        {
            code[offset++] = InstructionConstants.OP_WIDE;
        }

        // Write the opcode.
        code[offset++] = actualOpcode();

        // Write any additional arguments.
        writeInfo(code, offset);
    }


    /**
     * Returns whether the instruction is wide, i.e. preceded by a wide opcode.
     * With the current specifications, only variable instructions can be wide.
     */
    protected boolean isWide()
    {
        return false;
    }


    /**
     * Reads the data following the instruction opcode.
     */
    protected abstract void readInfo(byte[] code, int offset);


    /**
     * Writes data following the instruction opcode.
     */
    protected abstract void writeInfo(byte[] code, int offset);


    /**
     * Returns the length in bytes of the instruction.
     */
    public abstract int length(int offset);


    /**
     * Accepts the given visitor.
     */
    public abstract void accept(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, InstructionVisitor instructionVisitor);


    /**
     * Returns a description of the instruction, at the given offset.
     */
    public String toString(int offset)
    {
        return "["+offset+"] "+ this.toString();
    }


    /**
     * Returns the name of the instruction.
     */
    public String getName()
    {
        return InstructionConstants.NAMES[opcode & 0xff];
    }


    /**
     * Returns whether the instruction may throw exceptions.
     */
    public boolean mayThrowExceptions()
    {
        return MAY_THROW_EXCEPTIONS[opcode & 0xff];
    }


    /**
     * Returns whether the instruction is a Category 2 instruction. This means
     * that it operates on long or double arguments.
     */
    public boolean isCategory2()
    {
        return IS_CATEGORY2[opcode & 0xff];
    }


    /**
     * Returns the number of entries popped from the stack during the execution
     * of the instruction.
     */
    public int stackPopCount(Clazz clazz)
    {
        return STACK_POP_COUNTS[opcode & 0xff];
    }


    /**
     * Returns the number of entries pushed onto the stack during the execution
     * of the instruction.
     */
    public int stackPushCount(Clazz clazz)
    {
        return STACK_PUSH_COUNTS[opcode & 0xff];
    }


    // Small utility methods.

    protected static int readByte(byte[] code, int offset)
    {
        return code[offset] & 0xff;
    }

    protected static int readShort(byte[] code, int offset)
    {
        return ((code[offset++] & 0xff) << 8) |
               ( code[offset  ] & 0xff      );
    }

    protected static int readInt(byte[] code, int offset)
    {
        return ( code[offset++]         << 24) |
               ((code[offset++] & 0xff) << 16) |
               ((code[offset++] & 0xff) <<  8) |
               ( code[offset  ] & 0xff       );
    }

    protected static int readValue(byte[] code, int offset, int valueSize)
    {
        switch (valueSize)
        {
            case 0: return 0;
            case 1: return readByte( code, offset);
            case 2: return readShort(code, offset);
            case 4: return readInt(  code, offset);
            default: throw new IllegalArgumentException("Unsupported value size ["+valueSize+"]");
        }
    }

    protected static int readSignedByte(byte[] code, int offset)
    {
        return code[offset];
    }

    protected static int readSignedShort(byte[] code, int offset)
    {
        return (code[offset++] <<   8) |
               (code[offset  ] & 0xff);
    }

    protected static int readSignedValue(byte[] code, int offset, int valueSize)
    {
        switch (valueSize)
        {
            case 0: return 0;
            case 1: return readSignedByte( code, offset);
            case 2: return readSignedShort(code, offset);
            case 4: return readInt(        code, offset);
            default: throw new IllegalArgumentException("Unsupported value size ["+valueSize+"]");
        }
    }

    protected static void writeByte(byte[] code, int offset, int value)
    {
        if (value > 0xff)
        {
            throw new IllegalArgumentException("Unsigned byte value larger than 0xff ["+value+"]");
        }

        code[offset] = (byte)value;
    }

    protected static void writeShort(byte[] code, int offset, int value)
    {
        if (value > 0xffff)
        {
            throw new IllegalArgumentException("Unsigned short value larger than 0xffff ["+value+"]");
        }

        code[offset++] = (byte)(value >> 8);
        code[offset  ] = (byte)(value     );
    }

    protected static void writeInt(byte[] code, int offset, int value)
    {
        code[offset++] = (byte)(value >> 24);
        code[offset++] = (byte)(value >> 16);
        code[offset++] = (byte)(value >>  8);
        code[offset  ] = (byte)(value      );
    }

    protected static void writeValue(byte[] code, int offset, int value, int valueSize)
    {
        switch (valueSize)
        {
            case 0:                                  break;
            case 1: writeByte( code, offset, value); break;
            case 2: writeShort(code, offset, value); break;
            case 4: writeInt(  code, offset, value); break;
            default: throw new IllegalArgumentException("Unsupported value size ["+valueSize+"]");
        }
    }

    protected static void writeSignedByte(byte[] code, int offset, int value)
    {
        if ((byte)value != value)
        {
            throw new IllegalArgumentException("Signed byte value out of range ["+value+"]");
        }

        code[offset] = (byte)value;
    }

    protected static void writeSignedShort(byte[] code, int offset, int value)
    {
        if ((short)value != value)
        {
            throw new IllegalArgumentException("Signed short value out of range ["+value+"]");
        }

        code[offset++] = (byte)(value >> 8);
        code[offset  ] = (byte)(value     );
    }

    protected static void writeSignedValue(byte[] code, int offset, int value, int valueSize)
    {
        switch (valueSize)
        {
            case 0:                                        break;
            case 1: writeSignedByte( code, offset, value); break;
            case 2: writeSignedShort(code, offset, value); break;
            case 4: writeInt(        code, offset, value); break;
            default: throw new IllegalArgumentException("Unsupported value size ["+valueSize+"]");
        }
    }
}
