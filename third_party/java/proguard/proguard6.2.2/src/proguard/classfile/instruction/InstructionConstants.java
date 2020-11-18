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
 * Representation of an instruction.
 *
 * @author Eric Lafortune
 */
public interface InstructionConstants
{
    public static final byte OP_NOP             = 0;
    public static final byte OP_ACONST_NULL     = 1;
    public static final byte OP_ICONST_M1       = 2;
    public static final byte OP_ICONST_0        = 3;
    public static final byte OP_ICONST_1        = 4;
    public static final byte OP_ICONST_2        = 5;
    public static final byte OP_ICONST_3        = 6;
    public static final byte OP_ICONST_4        = 7;
    public static final byte OP_ICONST_5        = 8;
    public static final byte OP_LCONST_0        = 9;
    public static final byte OP_LCONST_1        = 10;
    public static final byte OP_FCONST_0        = 11;
    public static final byte OP_FCONST_1        = 12;
    public static final byte OP_FCONST_2        = 13;
    public static final byte OP_DCONST_0        = 14;
    public static final byte OP_DCONST_1        = 15;
    public static final byte OP_BIPUSH          = 16;
    public static final byte OP_SIPUSH          = 17;
    public static final byte OP_LDC             = 18;
    public static final byte OP_LDC_W           = 19;
    public static final byte OP_LDC2_W          = 20;
    public static final byte OP_ILOAD           = 21;
    public static final byte OP_LLOAD           = 22;
    public static final byte OP_FLOAD           = 23;
    public static final byte OP_DLOAD           = 24;
    public static final byte OP_ALOAD           = 25;
    public static final byte OP_ILOAD_0         = 26;
    public static final byte OP_ILOAD_1         = 27;
    public static final byte OP_ILOAD_2         = 28;
    public static final byte OP_ILOAD_3         = 29;
    public static final byte OP_LLOAD_0         = 30;
    public static final byte OP_LLOAD_1         = 31;
    public static final byte OP_LLOAD_2         = 32;
    public static final byte OP_LLOAD_3         = 33;
    public static final byte OP_FLOAD_0         = 34;
    public static final byte OP_FLOAD_1         = 35;
    public static final byte OP_FLOAD_2         = 36;
    public static final byte OP_FLOAD_3         = 37;
    public static final byte OP_DLOAD_0         = 38;
    public static final byte OP_DLOAD_1         = 39;
    public static final byte OP_DLOAD_2         = 40;
    public static final byte OP_DLOAD_3         = 41;
    public static final byte OP_ALOAD_0         = 42;
    public static final byte OP_ALOAD_1         = 43;
    public static final byte OP_ALOAD_2         = 44;
    public static final byte OP_ALOAD_3         = 45;
    public static final byte OP_IALOAD          = 46;
    public static final byte OP_LALOAD          = 47;
    public static final byte OP_FALOAD          = 48;
    public static final byte OP_DALOAD          = 49;
    public static final byte OP_AALOAD          = 50;
    public static final byte OP_BALOAD          = 51;
    public static final byte OP_CALOAD          = 52;
    public static final byte OP_SALOAD          = 53;
    public static final byte OP_ISTORE          = 54;
    public static final byte OP_LSTORE          = 55;
    public static final byte OP_FSTORE          = 56;
    public static final byte OP_DSTORE          = 57;
    public static final byte OP_ASTORE          = 58;
    public static final byte OP_ISTORE_0        = 59;
    public static final byte OP_ISTORE_1        = 60;
    public static final byte OP_ISTORE_2        = 61;
    public static final byte OP_ISTORE_3        = 62;
    public static final byte OP_LSTORE_0        = 63;
    public static final byte OP_LSTORE_1        = 64;
    public static final byte OP_LSTORE_2        = 65;
    public static final byte OP_LSTORE_3        = 66;
    public static final byte OP_FSTORE_0        = 67;
    public static final byte OP_FSTORE_1        = 68;
    public static final byte OP_FSTORE_2        = 69;
    public static final byte OP_FSTORE_3        = 70;
    public static final byte OP_DSTORE_0        = 71;
    public static final byte OP_DSTORE_1        = 72;
    public static final byte OP_DSTORE_2        = 73;
    public static final byte OP_DSTORE_3        = 74;
    public static final byte OP_ASTORE_0        = 75;
    public static final byte OP_ASTORE_1        = 76;
    public static final byte OP_ASTORE_2        = 77;
    public static final byte OP_ASTORE_3        = 78;
    public static final byte OP_IASTORE         = 79;
    public static final byte OP_LASTORE         = 80;
    public static final byte OP_FASTORE         = 81;
    public static final byte OP_DASTORE         = 82;
    public static final byte OP_AASTORE         = 83;
    public static final byte OP_BASTORE         = 84;
    public static final byte OP_CASTORE         = 85;
    public static final byte OP_SASTORE         = 86;
    public static final byte OP_POP             = 87;
    public static final byte OP_POP2            = 88;
    public static final byte OP_DUP             = 89;
    public static final byte OP_DUP_X1          = 90;
    public static final byte OP_DUP_X2          = 91;
    public static final byte OP_DUP2            = 92;
    public static final byte OP_DUP2_X1         = 93;
    public static final byte OP_DUP2_X2         = 94;
    public static final byte OP_SWAP            = 95;
    public static final byte OP_IADD            = 96;
    public static final byte OP_LADD            = 97;
    public static final byte OP_FADD            = 98;
    public static final byte OP_DADD            = 99;
    public static final byte OP_ISUB            = 100;
    public static final byte OP_LSUB            = 101;
    public static final byte OP_FSUB            = 102;
    public static final byte OP_DSUB            = 103;
    public static final byte OP_IMUL            = 104;
    public static final byte OP_LMUL            = 105;
    public static final byte OP_FMUL            = 106;
    public static final byte OP_DMUL            = 107;
    public static final byte OP_IDIV            = 108;
    public static final byte OP_LDIV            = 109;
    public static final byte OP_FDIV            = 110;
    public static final byte OP_DDIV            = 111;
    public static final byte OP_IREM            = 112;
    public static final byte OP_LREM            = 113;
    public static final byte OP_FREM            = 114;
    public static final byte OP_DREM            = 115;
    public static final byte OP_INEG            = 116;
    public static final byte OP_LNEG            = 117;
    public static final byte OP_FNEG            = 118;
    public static final byte OP_DNEG            = 119;
    public static final byte OP_ISHL            = 120;
    public static final byte OP_LSHL            = 121;
    public static final byte OP_ISHR            = 122;
    public static final byte OP_LSHR            = 123;
    public static final byte OP_IUSHR           = 124;
    public static final byte OP_LUSHR           = 125;
    public static final byte OP_IAND            = 126;
    public static final byte OP_LAND            = 127;
    public static final byte OP_IOR             = -128;
    public static final byte OP_LOR             = -127;
    public static final byte OP_IXOR            = -126;
    public static final byte OP_LXOR            = -125;
    public static final byte OP_IINC            = -124;
    public static final byte OP_I2L             = -123;
    public static final byte OP_I2F             = -122;
    public static final byte OP_I2D             = -121;
    public static final byte OP_L2I             = -120;
    public static final byte OP_L2F             = -119;
    public static final byte OP_L2D             = -118;
    public static final byte OP_F2I             = -117;
    public static final byte OP_F2L             = -116;
    public static final byte OP_F2D             = -115;
    public static final byte OP_D2I             = -114;
    public static final byte OP_D2L             = -113;
    public static final byte OP_D2F             = -112;
    public static final byte OP_I2B             = -111;
    public static final byte OP_I2C             = -110;
    public static final byte OP_I2S             = -109;
    public static final byte OP_LCMP            = -108;
    public static final byte OP_FCMPL           = -107;
    public static final byte OP_FCMPG           = -106;
    public static final byte OP_DCMPL           = -105;
    public static final byte OP_DCMPG           = -104;
    public static final byte OP_IFEQ            = -103;
    public static final byte OP_IFNE            = -102;
    public static final byte OP_IFLT            = -101;
    public static final byte OP_IFGE            = -100;
    public static final byte OP_IFGT            = -99;
    public static final byte OP_IFLE            = -98;
    public static final byte OP_IFICMPEQ        = -97;
    public static final byte OP_IFICMPNE        = -96;
    public static final byte OP_IFICMPLT        = -95;
    public static final byte OP_IFICMPGE        = -94;
    public static final byte OP_IFICMPGT        = -93;
    public static final byte OP_IFICMPLE        = -92;
    public static final byte OP_IFACMPEQ        = -91;
    public static final byte OP_IFACMPNE        = -90;
    public static final byte OP_GOTO            = -89;
    public static final byte OP_JSR             = -88;
    public static final byte OP_RET             = -87;
    public static final byte OP_TABLESWITCH     = -86;
    public static final byte OP_LOOKUPSWITCH    = -85;
    public static final byte OP_IRETURN         = -84;
    public static final byte OP_LRETURN         = -83;
    public static final byte OP_FRETURN         = -82;
    public static final byte OP_DRETURN         = -81;
    public static final byte OP_ARETURN         = -80;
    public static final byte OP_RETURN          = -79;
    public static final byte OP_GETSTATIC       = -78;
    public static final byte OP_PUTSTATIC       = -77;
    public static final byte OP_GETFIELD        = -76;
    public static final byte OP_PUTFIELD        = -75;
    public static final byte OP_INVOKEVIRTUAL   = -74;
    public static final byte OP_INVOKESPECIAL   = -73;
    public static final byte OP_INVOKESTATIC    = -72;
    public static final byte OP_INVOKEINTERFACE = -71;
    public static final byte OP_INVOKEDYNAMIC   = -70;
    public static final byte OP_NEW             = -69;
    public static final byte OP_NEWARRAY        = -68;
    public static final byte OP_ANEWARRAY       = -67;
    public static final byte OP_ARRAYLENGTH     = -66;
    public static final byte OP_ATHROW          = -65;
    public static final byte OP_CHECKCAST       = -64;
    public static final byte OP_INSTANCEOF      = -63;
    public static final byte OP_MONITORENTER    = -62;
    public static final byte OP_MONITOREXIT     = -61;
    public static final byte OP_WIDE            = -60;
    public static final byte OP_MULTIANEWARRAY  = -59;
    public static final byte OP_IFNULL          = -58;
    public static final byte OP_IFNONNULL       = -57;
    public static final byte OP_GOTO_W          = -56;
    public static final byte OP_JSR_W           = -55;


    public static final String[] NAMES =
    {
        "nop",
        "aconst_null",
        "iconst_m1",
        "iconst_0",
        "iconst_1",
        "iconst_2",
        "iconst_3",
        "iconst_4",
        "iconst_5",
        "lconst_0",
        "lconst_1",
        "fconst_0",
        "fconst_1",
        "fconst_2",
        "dconst_0",
        "dconst_1",
        "bipush",
        "sipush",
        "ldc",
        "ldc_w",
        "ldc2_w",
        "iload",
        "lload",
        "fload",
        "dload",
        "aload",
        "iload_0",
        "iload_1",
        "iload_2",
        "iload_3",
        "lload_0",
        "lload_1",
        "lload_2",
        "lload_3",
        "fload_0",
        "fload_1",
        "fload_2",
        "fload_3",
        "dload_0",
        "dload_1",
        "dload_2",
        "dload_3",
        "aload_0",
        "aload_1",
        "aload_2",
        "aload_3",
        "iaload",
        "laload",
        "faload",
        "daload",
        "aaload",
        "baload",
        "caload",
        "saload",
        "istore",
        "lstore",
        "fstore",
        "dstore",
        "astore",
        "istore_0",
        "istore_1",
        "istore_2",
        "istore_3",
        "lstore_0",
        "lstore_1",
        "lstore_2",
        "lstore_3",
        "fstore_0",
        "fstore_1",
        "fstore_2",
        "fstore_3",
        "dstore_0",
        "dstore_1",
        "dstore_2",
        "dstore_3",
        "astore_0",
        "astore_1",
        "astore_2",
        "astore_3",
        "iastore",
        "lastore",
        "fastore",
        "dastore",
        "aastore",
        "bastore",
        "castore",
        "sastore",
        "pop",
        "pop2",
        "dup",
        "dup_x1",
        "dup_x2",
        "dup2",
        "dup2_x1",
        "dup2_x2",
        "swap",
        "iadd",
        "ladd",
        "fadd",
        "dadd",
        "isub",
        "lsub",
        "fsub",
        "dsub",
        "imul",
        "lmul",
        "fmul",
        "dmul",
        "idiv",
        "ldiv",
        "fdiv",
        "ddiv",
        "irem",
        "lrem",
        "frem",
        "drem",
        "ineg",
        "lneg",
        "fneg",
        "dneg",
        "ishl",
        "lshl",
        "ishr",
        "lshr",
        "iushr",
        "lushr",
        "iand",
        "land",
        "ior",
        "lor",
        "ixor",
        "lxor",
        "iinc",
        "i2l",
        "i2f",
        "i2d",
        "l2i",
        "l2f",
        "l2d",
        "f2i",
        "f2l",
        "f2d",
        "d2i",
        "d2l",
        "d2f",
        "i2b",
        "i2c",
        "i2s",
        "lcmp",
        "fcmpl",
        "fcmpg",
        "dcmpl",
        "dcmpg",
        "ifeq",
        "ifne",
        "iflt",
        "ifge",
        "ifgt",
        "ifle",
        "ificmpeq",
        "ificmpne",
        "ificmplt",
        "ificmpge",
        "ificmpgt",
        "ificmple",
        "ifacmpeq",
        "ifacmpne",
        "goto",
        "jsr",
        "ret",
        "tableswitch",
        "lookupswitch",
        "ireturn",
        "lreturn",
        "freturn",
        "dreturn",
        "areturn",
        "return",
        "getstatic",
        "putstatic",
        "getfield",
        "putfield",
        "invokevirtual",
        "invokespecial",
        "invokestatic",
        "invokeinterface",
        "invokedynamic",
        "new",
        "newarray",
        "anewarray",
        "arraylength",
        "athrow",
        "checkcast",
        "instanceof",
        "monitorenter",
        "monitorexit",
        "wide",
        "multianewarray",
        "ifnull",
        "ifnonnull",
        "goto_w",
        "jsr_w",
    };


    public static final byte ARRAY_T_BOOLEAN = 4;
    public static final byte ARRAY_T_CHAR    = 5;
    public static final byte ARRAY_T_FLOAT   = 6;
    public static final byte ARRAY_T_DOUBLE  = 7;
    public static final byte ARRAY_T_BYTE    = 8;
    public static final byte ARRAY_T_SHORT   = 9;
    public static final byte ARRAY_T_INT     = 10;
    public static final byte ARRAY_T_LONG    = 11;
}
