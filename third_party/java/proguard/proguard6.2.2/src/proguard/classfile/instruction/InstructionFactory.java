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

/**
 * This class provides methods to create and reuse Instruction objects.
 *
 * @author Eric Lafortune
 */
public class InstructionFactory
{
    /**
     * Creates a new Instruction from the data in the byte array, starting
     * at the given offset.
     */
    public static Instruction create(byte[] code, int offset)
    {
        int  index  = offset;
        byte opcode = code[index++];

        boolean wide = false;
        if (opcode == InstructionConstants.OP_WIDE)
        {
            opcode = code[index++];
            wide   = true;
        }

        Instruction instruction = create(opcode, wide);

        instruction.opcode = opcode;

        instruction.readInfo(code, index);

        return instruction;
    }


    /**
     * Creates a new Instruction corresponding to the given opcode.
     */
    public static Instruction create(byte opcode)
    {
        Instruction instruction = create(opcode, false);

        instruction.opcode = opcode;

        return instruction;
    }


    /**
     * Creates a new Instruction corresponding to the given opcode.
     */
    private static Instruction create(byte opcode, boolean wide)
    {
        switch (opcode)
        {
            // Simple instructions.
            case InstructionConstants.OP_NOP:
            case InstructionConstants.OP_ACONST_NULL:
            case InstructionConstants.OP_ICONST_M1:
            case InstructionConstants.OP_ICONST_0:
            case InstructionConstants.OP_ICONST_1:
            case InstructionConstants.OP_ICONST_2:
            case InstructionConstants.OP_ICONST_3:
            case InstructionConstants.OP_ICONST_4:
            case InstructionConstants.OP_ICONST_5:
            case InstructionConstants.OP_LCONST_0:
            case InstructionConstants.OP_LCONST_1:
            case InstructionConstants.OP_FCONST_0:
            case InstructionConstants.OP_FCONST_1:
            case InstructionConstants.OP_FCONST_2:
            case InstructionConstants.OP_DCONST_0:
            case InstructionConstants.OP_DCONST_1:

            case InstructionConstants.OP_BIPUSH:
            case InstructionConstants.OP_SIPUSH:

            case InstructionConstants.OP_IALOAD:
            case InstructionConstants.OP_LALOAD:
            case InstructionConstants.OP_FALOAD:
            case InstructionConstants.OP_DALOAD:
            case InstructionConstants.OP_AALOAD:
            case InstructionConstants.OP_BALOAD:
            case InstructionConstants.OP_CALOAD:
            case InstructionConstants.OP_SALOAD:

            case InstructionConstants.OP_IASTORE:
            case InstructionConstants.OP_LASTORE:
            case InstructionConstants.OP_FASTORE:
            case InstructionConstants.OP_DASTORE:
            case InstructionConstants.OP_AASTORE:
            case InstructionConstants.OP_BASTORE:
            case InstructionConstants.OP_CASTORE:
            case InstructionConstants.OP_SASTORE:
            case InstructionConstants.OP_POP:
            case InstructionConstants.OP_POP2:
            case InstructionConstants.OP_DUP:
            case InstructionConstants.OP_DUP_X1:
            case InstructionConstants.OP_DUP_X2:
            case InstructionConstants.OP_DUP2:
            case InstructionConstants.OP_DUP2_X1:
            case InstructionConstants.OP_DUP2_X2:
            case InstructionConstants.OP_SWAP:
            case InstructionConstants.OP_IADD:
            case InstructionConstants.OP_LADD:
            case InstructionConstants.OP_FADD:
            case InstructionConstants.OP_DADD:
            case InstructionConstants.OP_ISUB:
            case InstructionConstants.OP_LSUB:
            case InstructionConstants.OP_FSUB:
            case InstructionConstants.OP_DSUB:
            case InstructionConstants.OP_IMUL:
            case InstructionConstants.OP_LMUL:
            case InstructionConstants.OP_FMUL:
            case InstructionConstants.OP_DMUL:
            case InstructionConstants.OP_IDIV:
            case InstructionConstants.OP_LDIV:
            case InstructionConstants.OP_FDIV:
            case InstructionConstants.OP_DDIV:
            case InstructionConstants.OP_IREM:
            case InstructionConstants.OP_LREM:
            case InstructionConstants.OP_FREM:
            case InstructionConstants.OP_DREM:
            case InstructionConstants.OP_INEG:
            case InstructionConstants.OP_LNEG:
            case InstructionConstants.OP_FNEG:
            case InstructionConstants.OP_DNEG:
            case InstructionConstants.OP_ISHL:
            case InstructionConstants.OP_LSHL:
            case InstructionConstants.OP_ISHR:
            case InstructionConstants.OP_LSHR:
            case InstructionConstants.OP_IUSHR:
            case InstructionConstants.OP_LUSHR:
            case InstructionConstants.OP_IAND:
            case InstructionConstants.OP_LAND:
            case InstructionConstants.OP_IOR:
            case InstructionConstants.OP_LOR:
            case InstructionConstants.OP_IXOR:
            case InstructionConstants.OP_LXOR:

            case InstructionConstants.OP_I2L:
            case InstructionConstants.OP_I2F:
            case InstructionConstants.OP_I2D:
            case InstructionConstants.OP_L2I:
            case InstructionConstants.OP_L2F:
            case InstructionConstants.OP_L2D:
            case InstructionConstants.OP_F2I:
            case InstructionConstants.OP_F2L:
            case InstructionConstants.OP_F2D:
            case InstructionConstants.OP_D2I:
            case InstructionConstants.OP_D2L:
            case InstructionConstants.OP_D2F:
            case InstructionConstants.OP_I2B:
            case InstructionConstants.OP_I2C:
            case InstructionConstants.OP_I2S:
            case InstructionConstants.OP_LCMP:
            case InstructionConstants.OP_FCMPL:
            case InstructionConstants.OP_FCMPG:
            case InstructionConstants.OP_DCMPL:
            case InstructionConstants.OP_DCMPG:

            case InstructionConstants.OP_IRETURN:
            case InstructionConstants.OP_LRETURN:
            case InstructionConstants.OP_FRETURN:
            case InstructionConstants.OP_DRETURN:
            case InstructionConstants.OP_ARETURN:
            case InstructionConstants.OP_RETURN:

            case InstructionConstants.OP_NEWARRAY:
            case InstructionConstants.OP_ARRAYLENGTH:
            case InstructionConstants.OP_ATHROW:

            case InstructionConstants.OP_MONITORENTER:
            case InstructionConstants.OP_MONITOREXIT:
                return new SimpleInstruction();

            // Instructions with a contant pool index.
            case InstructionConstants.OP_LDC:
            case InstructionConstants.OP_LDC_W:
            case InstructionConstants.OP_LDC2_W:

            case InstructionConstants.OP_GETSTATIC:
            case InstructionConstants.OP_PUTSTATIC:
            case InstructionConstants.OP_GETFIELD:
            case InstructionConstants.OP_PUTFIELD:

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
            case InstructionConstants.OP_INVOKEINTERFACE:
            case InstructionConstants.OP_INVOKEDYNAMIC:

            case InstructionConstants.OP_NEW:
            case InstructionConstants.OP_ANEWARRAY:
            case InstructionConstants.OP_CHECKCAST:
            case InstructionConstants.OP_INSTANCEOF:
            case InstructionConstants.OP_MULTIANEWARRAY:
                return new ConstantInstruction();

            // Instructions with a local variable index.
            case InstructionConstants.OP_ILOAD:
            case InstructionConstants.OP_LLOAD:
            case InstructionConstants.OP_FLOAD:
            case InstructionConstants.OP_DLOAD:
            case InstructionConstants.OP_ALOAD:
            case InstructionConstants.OP_ILOAD_0:
            case InstructionConstants.OP_ILOAD_1:
            case InstructionConstants.OP_ILOAD_2:
            case InstructionConstants.OP_ILOAD_3:
            case InstructionConstants.OP_LLOAD_0:
            case InstructionConstants.OP_LLOAD_1:
            case InstructionConstants.OP_LLOAD_2:
            case InstructionConstants.OP_LLOAD_3:
            case InstructionConstants.OP_FLOAD_0:
            case InstructionConstants.OP_FLOAD_1:
            case InstructionConstants.OP_FLOAD_2:
            case InstructionConstants.OP_FLOAD_3:
            case InstructionConstants.OP_DLOAD_0:
            case InstructionConstants.OP_DLOAD_1:
            case InstructionConstants.OP_DLOAD_2:
            case InstructionConstants.OP_DLOAD_3:
            case InstructionConstants.OP_ALOAD_0:
            case InstructionConstants.OP_ALOAD_1:
            case InstructionConstants.OP_ALOAD_2:
            case InstructionConstants.OP_ALOAD_3:

            case InstructionConstants.OP_ISTORE:
            case InstructionConstants.OP_LSTORE:
            case InstructionConstants.OP_FSTORE:
            case InstructionConstants.OP_DSTORE:
            case InstructionConstants.OP_ASTORE:
            case InstructionConstants.OP_ISTORE_0:
            case InstructionConstants.OP_ISTORE_1:
            case InstructionConstants.OP_ISTORE_2:
            case InstructionConstants.OP_ISTORE_3:
            case InstructionConstants.OP_LSTORE_0:
            case InstructionConstants.OP_LSTORE_1:
            case InstructionConstants.OP_LSTORE_2:
            case InstructionConstants.OP_LSTORE_3:
            case InstructionConstants.OP_FSTORE_0:
            case InstructionConstants.OP_FSTORE_1:
            case InstructionConstants.OP_FSTORE_2:
            case InstructionConstants.OP_FSTORE_3:
            case InstructionConstants.OP_DSTORE_0:
            case InstructionConstants.OP_DSTORE_1:
            case InstructionConstants.OP_DSTORE_2:
            case InstructionConstants.OP_DSTORE_3:
            case InstructionConstants.OP_ASTORE_0:
            case InstructionConstants.OP_ASTORE_1:
            case InstructionConstants.OP_ASTORE_2:
            case InstructionConstants.OP_ASTORE_3:

            case InstructionConstants.OP_IINC:

            case InstructionConstants.OP_RET:
                return new VariableInstruction(wide);

            // Instructions with a branch offset operand.
            case InstructionConstants.OP_IFEQ:
            case InstructionConstants.OP_IFNE:
            case InstructionConstants.OP_IFLT:
            case InstructionConstants.OP_IFGE:
            case InstructionConstants.OP_IFGT:
            case InstructionConstants.OP_IFLE:
            case InstructionConstants.OP_IFICMPEQ:
            case InstructionConstants.OP_IFICMPNE:
            case InstructionConstants.OP_IFICMPLT:
            case InstructionConstants.OP_IFICMPGE:
            case InstructionConstants.OP_IFICMPGT:
            case InstructionConstants.OP_IFICMPLE:
            case InstructionConstants.OP_IFACMPEQ:
            case InstructionConstants.OP_IFACMPNE:
            case InstructionConstants.OP_GOTO:
            case InstructionConstants.OP_JSR:

            case InstructionConstants.OP_IFNULL:
            case InstructionConstants.OP_IFNONNULL:

            case InstructionConstants.OP_GOTO_W:
            case InstructionConstants.OP_JSR_W:
                return new BranchInstruction();

            //  The tableswitch instruction.
            case InstructionConstants.OP_TABLESWITCH:
                return new TableSwitchInstruction();

            //  The lookupswitch instruction.
            case InstructionConstants.OP_LOOKUPSWITCH:
                return new LookUpSwitchInstruction();

            default:
                throw new IllegalArgumentException("Unknown instruction opcode ["+opcode+"]");
        }
    }
}
