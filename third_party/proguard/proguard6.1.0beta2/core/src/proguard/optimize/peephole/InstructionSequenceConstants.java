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
package proguard.optimize.peephole;

import proguard.classfile.*;
import proguard.classfile.constant.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.visitor.ClassPrinter;

/**
 * This class contains a set of instruction sequences with their suggested
 * more compact or more efficient replacements.
 *
 * @see InstructionSequencesReplacer
 * @see InstructionSequenceReplacer
 * @author Eric Lafortune
 */
public class InstructionSequenceConstants
{
    // The arrays with constants and instructions used to be static,
    // but now they are initialized with references to classes and
    // class members, inside an instance of this class. As an added
    // benefit, they can be garbage collected after they have been used.
    public final Instruction[][][] VARIABLE_SEQUENCES;
    public final Instruction[][][] ARITHMETIC_SEQUENCES;
    public final Instruction[][][] FIELD_SEQUENCES;
    public final Instruction[][][] CAST_SEQUENCES;
    public final Instruction[][][] BRANCH_SEQUENCES;
    public final Instruction[][][] STRING_SEQUENCES;
    public final Instruction[][][] OBJECT_SEQUENCES;
    public final Instruction[][][] MATH_SEQUENCES;
    public final Instruction[][][] MATH_ANDROID_SEQUENCES;

    public final Constant[] CONSTANTS;

    // Internal short-hand constants.
    private static final String BOOLEAN        = ClassConstants.NAME_JAVA_LANG_BOOLEAN;
    private static final String BYTE           = ClassConstants.NAME_JAVA_LANG_BYTE;
    private static final String CHARACTER      = ClassConstants.NAME_JAVA_LANG_CHARACTER;
    private static final String SHORT          = ClassConstants.NAME_JAVA_LANG_SHORT;
    private static final String INTEGER        = ClassConstants.NAME_JAVA_LANG_INTEGER;
    private static final String LONG           = ClassConstants.NAME_JAVA_LANG_LONG;
    private static final String FLOAT          = ClassConstants.NAME_JAVA_LANG_FLOAT;
    private static final String DOUBLE         = ClassConstants.NAME_JAVA_LANG_DOUBLE;
    private static final String STRING         = ClassConstants.NAME_JAVA_LANG_STRING;
    private static final String STRING_BUFFER  = ClassConstants.NAME_JAVA_LANG_STRING_BUFFER;
    private static final String STRING_BUILDER = ClassConstants.NAME_JAVA_LANG_STRING_BUILDER;
    private static final String MATH           = ClassConstants.NAME_JAVA_LANG_MATH;
    private static final String FLOAT_MATH     = ClassConstants.NAME_ANDROID_UTIL_FLOAT_MATH;

    private static final int X = InstructionSequenceReplacer.X;
    private static final int Y = InstructionSequenceReplacer.Y;
    private static final int Z = InstructionSequenceReplacer.Z;

    private static final int A = InstructionSequenceReplacer.A;
    private static final int B = InstructionSequenceReplacer.B;
    private static final int C = InstructionSequenceReplacer.C;
    private static final int D = InstructionSequenceReplacer.D;

    // Replacement constants that are derived from matched variables.
    private static final int STRING_A_LENGTH  = InstructionSequenceReplacer.STRING_A_LENGTH;
    private static final int BOOLEAN_A_STRING = InstructionSequenceReplacer.BOOLEAN_A_STRING;
    private static final int CHAR_A_STRING    = InstructionSequenceReplacer.CHAR_A_STRING;
    private static final int INT_A_STRING     = InstructionSequenceReplacer.INT_A_STRING;
    private static final int LONG_A_STRING    = InstructionSequenceReplacer.LONG_A_STRING;
    private static final int FLOAT_A_STRING   = InstructionSequenceReplacer.FLOAT_A_STRING;
    private static final int DOUBLE_A_STRING  = InstructionSequenceReplacer.DOUBLE_A_STRING;
    private static final int STRING_A_STRING  = InstructionSequenceReplacer.STRING_A_STRING;
    private static final int BOOLEAN_B_STRING = InstructionSequenceReplacer.BOOLEAN_B_STRING;
    private static final int CHAR_B_STRING    = InstructionSequenceReplacer.CHAR_B_STRING;
    private static final int INT_B_STRING     = InstructionSequenceReplacer.INT_B_STRING;
    private static final int LONG_B_STRING    = InstructionSequenceReplacer.LONG_B_STRING;
    private static final int FLOAT_B_STRING   = InstructionSequenceReplacer.FLOAT_B_STRING;
    private static final int DOUBLE_B_STRING  = InstructionSequenceReplacer.DOUBLE_B_STRING;
    private static final int STRING_B_STRING  = InstructionSequenceReplacer.STRING_B_STRING;


    /**
     * Creates a new instance of InstructionSequenceConstants, with constants
     * that reference classes from the given class pools.
     */
    public InstructionSequenceConstants(ClassPool programClassPool,
                                        ClassPool libraryClassPool)
    {
        InstructionSequenceBuilder ____ =
            new InstructionSequenceBuilder(programClassPool,
                                           libraryClassPool);

        // Create fieldref constants with wildcards, for fields in class X,
        // with name Y, and the given primitive types.
        ConstantPoolEditor constantPoolEditor = ____.getConstantPoolEditor();
        final int FIELD_Z = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("Z"))), null, null));
        final int FIELD_B = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("B"))), null, null));
        final int FIELD_C = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("C"))), null, null));
        final int FIELD_S = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("S"))), null, null));
        final int FIELD_I = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("I"))), null, null));
        final int FIELD_F = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("F"))), null, null));
        final int FIELD_J = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("J"))), null, null));
        final int FIELD_D = constantPoolEditor.addConstant(new FieldrefConstant(X, constantPoolEditor.addConstant(new NameAndTypeConstant(Y, constantPoolEditor.addUtf8Constant("D"))), null, null));

        // Create methodref constants with wildcards, for methods in class X,
        // with the given names and descriptors.
        final int EQUALS        = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant(ClassConstants.METHOD_NAME_EQUALS,   ClassConstants.METHOD_TYPE_EQUALS), null, null));
        final int TO_STRING     = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant(ClassConstants.METHOD_NAME_TOSTRING, ClassConstants.METHOD_TYPE_TOSTRING), null, null));
        final int BOOLEAN_VALUE = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("booleanValue", "()Z"), null, null));
        final int BYTE_VALUE    = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("byteValue",    "()B"), null, null));
        final int CHAR_VALUE    = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("charValue",    "()C"), null, null));
        final int SHORT_VALUE   = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("shortValue",   "()S"), null, null));
        final int INT_VALUE     = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("intValue",     "()I"), null, null));
        final int FLOAT_VALUE   = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("floatValue",   "()F"), null, null));
        final int LONG_VALUE    = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("longValue",    "()J"), null, null));
        final int DOUBLE_VALUE  = constantPoolEditor.addConstant(new MethodrefConstant(X, constantPoolEditor.addNameAndTypeConstant("doubleValue",  "()D"), null, null));

        final InstructionSequenceReplacer.Label TRY_START = InstructionSequenceReplacer.label();
        final InstructionSequenceReplacer.Label TRY_END   = InstructionSequenceReplacer.label();
        final InstructionSequenceReplacer.Label CATCH_END = InstructionSequenceReplacer.label();

        final InstructionSequenceReplacer.Label CATCH_EXCEPTION = InstructionSequenceReplacer.catch_(TRY_START.offset(), TRY_END.offset(), constantPoolEditor.addClassConstant(ClassConstants.NAME_JAVA_LANG_EXCEPTION, null));

        VARIABLE_SEQUENCES = new Instruction[][][]
        {
            {   // nop = nothing
                ____.nop().__(),
            },
            {   // iload/pop = nothing
                ____.iload(X)
                    .pop().__(),
            },
            {   // lload/pop2 = nothing
                ____.lload(X)
                    .pop2().__(),
            },
            {   // fload/pop = nothing
                ____.fload(X)
                    .pop().__(),
            },
            {   // dload/pop2 = nothing
                ____.dload(X)
                    .pop2().__(),
            },
            {   // aload/pop = nothing
                ____.aload(X)
                    .pop().__(),
            },
            {   // i = i = nothing
                ____.iload(X)
                    .istore(X).__(),
            },
            {   // l = l = nothing
                ____.lload(X)
                    .lstore(X).__(),
            },
            {   // f = f = nothing
                ____.fload(X)
                    .fstore(X).__(),
            },
            {   // d = d = nothing
                ____.dload(X)
                    .dstore(X).__(),
            },
            {   // a = a = nothing
                ____.aload(X)
                    .astore(X).__(),
            },
            {   // iload/iload = iload/dup
                ____.iload(X)
                    .iload(X).__(),

                ____.iload(X)
                    .dup().__()
            },
            {   // lload/lload = lload/dup2
                ____.lload(X)
                    .lload(X).__(),

                ____.lload(X)
                    .dup2().__()
            },
            {   // fload/fload = fload/dup
                ____.fload(X)
                    .fload(X).__(),

                ____.fload(X)
                    .dup().__()
            },
            {   // dload/dload = dload/dup2
                ____.dload(X)
                    .dload(X).__(),

                ____.dload(X)
                    .dup2().__()
            },
            {   // aload/aload = aload/dup
                ____.aload(X)
                    .aload(X).__(),

                ____.aload(X)
                    .dup().__()
            },
            {   // istore/istore = pop/istore
                ____.istore(X)
                    .istore(X).__(),

                ____.pop()
                    .istore(X).__()
            },
            {   // lstore/lstore = pop2/lstore
                ____.lstore(X)
                    .lstore(X).__(),

                ____.pop2()
                    .lstore(X).__()
            },
            {   // fstore/fstore = pop/fstore
                ____.fstore(X)
                    .fstore(X).__(),

                ____.pop()
                    .fstore(X).__()
            },
            {   // dstore/dstore = pop2/dstore
                ____.dstore(X)
                    .dstore(X).__(),

                ____.pop2()
                    .dstore(X).__()
            },
            {   // astore/astore = pop/astore
                ____.astore(X)
                    .astore(X).__(),

                ____.pop()
                    .astore(X).__()
            },
            {   // istore/iload = dup/istore
                ____.istore(X)
                    .iload(X).__(),

                ____.dup()
                    .istore(X).__()
            },
            {   // lstore/lload = dup2/lstore
                ____.lstore(X)
                    .lload(X).__(),

                ____.dup2()
                    .lstore(X).__()
            },
            {   // fstore/fload = dup/fstore
                ____.fstore(X)
                    .fload(X).__(),

                ____.dup()
                    .fstore(X).__()
            },
            {   // dstore/dload = dup2/dstore
                ____.dstore(X)
                    .dload(X).__(),

                ____.dup2()
                    .dstore(X).__()
            },
            {   // astore/aload = dup/astore
                ____.astore(X)
                    .aload(X).__(),

                ____.dup()
                    .astore(X).__()
            },
            {   // iload/dup/istore = iload
                ____.iload(X)
                    .dup()
                    .istore(X).__(),

                ____.iload(X).__()
            },
            {   // lload/dup2/lstore = lload
                ____.lload(X)
                    .dup2()
                    .lstore(X).__(),

                ____.lload(X).__()
            },
            {   // fload/dup/fstore = iload
                ____.fload(X)
                    .dup()
                    .fstore(X).__(),

                ____.fload(X).__()
            },
            {   // dload/dup2/dstore = dload
                ____.dload(X)
                    .dup2()
                    .dstore(X).__(),

                ____.dload(X).__()
            },
            {   // aload/dup/astore = aload
                ____.aload(X)
                    .dup()
                    .astore(X).__(),

                ____.aload(X).__()
            },
        };

        ARITHMETIC_SEQUENCES = new Instruction[][][]
        {
            {   // c + i = i + c
                ____.iconst(A)
                    .iload(X)
                    .iadd().__(),

                ____.iload(X)
                    .iconst(A)
                    .iadd().__()
            },
            {   // b + i = i + b
                ____.bipush(A)
                    .iload(X)
                    .iadd().__(),

                ____.iload(X)
                    .bipush(A)
                    .iadd().__()
            },
            {   // s + i = i + s
                ____.sipush(A)
                    .iload(X)
                    .iadd().__(),

                ____.iload(X)
                    .sipush(A)
                    .iadd().__()
            },
            {   // c + i = i + c
                ____.ldc_(A)
                    .iload(X)
                    .iadd().__(),

                ____.iload(X)
                    .ldc_(A)
                    .iadd().__()
            },
            {   // c * i = i * c
                ____.sipush(A)
                    .iload(X)
                    .imul().__(),

                ____.iload(X)
                    .sipush(A)
                    .imul().__()
            },
            {   // b * i = i * b
                ____.bipush(A)
                    .iload(X)
                    .imul().__(),

                ____.iload(X)
                    .bipush(A)
                    .imul().__()
            },
            {   // s * i = i * s
                ____.sipush(A)
                    .iload(X)
                    .imul().__(),

                ____.iload(X)
                    .sipush(A)
                    .imul().__()
            },
            {   // c * i = i * c
                ____.ldc_(A)
                    .iload(X)
                    .imul().__(),

                ____.iload(X)
                    .ldc_(A)
                    .imul().__()
            },
            {   // c + l = l + c
                ____.lconst(A)
                    .lload(X)
                    .ladd().__(),

                ____.lload(X)
                    .lconst(A)
                    .ladd().__()
            },
            {   // c + l = l + c
                ____.ldc2_w(A)
                    .lload(X)
                    .ladd().__(),

                ____.lload(X)
                    .ldc2_w(A)
                    .ladd().__()
            },
            {   // c * l = l * c
                ____.lconst(A)
                    .lload(X)
                    .lmul().__(),

                ____.lload(X)
                    .lconst(A)
                    .lmul().__()
            },
            {   // c + f = f + c
                ____.fconst(A)
                    .fload(X)
                    .fadd().__(),

                ____.fload(X)
                    .fconst(A)
                    .fadd().__()
            },
            {   // c + f = f + c
                ____.ldc_(A)
                    .fload(X)
                    .fadd().__(),

                ____.fload(X)
                    .ldc_(A)
                    .fadd().__()
            },
            {   // c * f = f * c
                ____.fconst(A)
                    .fload(X)
                    .fmul().__(),

                ____.fload(X)
                    .fconst(A)
                    .fmul().__()
            },
            {   // c * f = f * c
                ____.ldc_(A)
                    .fload(X)
                    .lmul().__(),

                ____.fload(X)
                    .ldc_(A)
                    .lmul().__()
            },
            {   // c + d = d + c
                ____.dconst(A)
                    .dload(X)
                    .dadd().__(),

                ____.dload(X)
                    .dconst(A)
                    .dadd().__()
            },
            {   // c + d = d + c
                ____.ldc2_w(A)
                    .dload(X)
                    .dadd().__(),

                ____.dload(X)
                    .ldc2_w(A)
                    .dadd().__()
            },
            {   // c * d = d * c
                ____.dconst(A)
                    .dload(X)
                    .dmul().__(),

                ____.dload(X)
                    .dconst(A)
                    .dmul().__()
            },
            {   // c * d = d * c
                ____.ldc2_w(A)
                    .dload(X)
                    .dmul().__(),

                ____.dload(X)
                    .ldc2_w(A)
                    .dmul().__()
            },
            {   // i = i + c = i += c
                ____.iload(X)
                    .sipush(A)
                    .iadd()
                    .istore(X).__(),

                ____.iinc(X, A).__()
            },
            {   // i = i + b = i += b
                ____.iload(X)
                    .bipush(A)
                    .iadd()
                    .istore(X).__(),

                ____.iinc(X, A).__()
            },
            {   // i = i + s = i += s
                ____.iload(X)
                    .sipush(A)
                    .iadd()
                    .istore(X).__(),

                ____.iinc(X, A).__()
            },
            {   // i = i - -1 = i++
                ____.iload(X)
                    .iconst_m1()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, 1).__()
            },
            {   // i = i - 1 = i--
                ____.iload(X)
                    .iconst_1()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, -1).__()
            },
            {   // i = i - 2 = i -= 2
                ____.iload(X)
                    .iconst_2()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, -2).__()
            },
            {   // i = i - 3 = i -= 3
                ____.iload(X)
                    .iconst_3()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, -3).__()
            },
            {   // i = i - 4 = i -= 4
                ____.iload(X)
                    .iconst_4()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, -4).__()
            },
            {   // i = i - 5 = i -= 5
                ____.iload(X)
                    .iconst_5()
                    .isub()
                    .istore(X).__(),

                ____.iinc(X, -5).__()
            },
            {   // ... + 0 = ...
                ____.iconst_0()
                    .iadd().__(),
            },
            {   // ... + 0L = ...
                ____.lconst_0()
                    .ladd().__(),
            },
            // Not valid for -0.0.
//            {   // ... + 0f = ...
//                ____.fconst_0()
//                    .fadd().__(),
//
//            },
//            {   // ... + 0d = ...
//                ____.dconst_0()
//                    .dadd().__(),
//
//            },
            {   // ... - 0 = ...
                ____.iconst_0()
                    .isub().__(),
            },
            {   // ... - 0L = ...
                ____.lconst_0()
                    .lsub().__(),
            },
            {   // ... - 0f = ...
                ____.fconst_0()
                    .fsub().__(),
            },
            {   // ... - 0d = ...
                ____.dconst_0()
                    .dsub().__(),
            },
            {   // ... * -1 = -...
                ____.iconst_m1()
                    .imul().__(),

                ____.ineg().__()
            },
            {   // ... * 0 = 0
                ____.iconst_0()
                    .imul().__(),

                ____.pop()
                    .iconst_0().__()
            },
            {   // ... * 1 = ...
                ____.iconst_1()
                    .imul().__(),
            },
            {   // ... * 2 = ... << 1
                ____.iconst_2()
                    .imul().__(),

                ____.iconst_1()
                    .ishl().__()
            },
            {   // ... * 4 = ... << 2
                ____.iconst_4()
                    .imul().__(),

                ____.iconst_2()
                    .ishl().__()
            },
            {   // ... * 8 = ... << 3
                ____.bipush(8)
                    .imul().__(),

                ____.iconst_3()
                    .ishl().__()
            },
            {   // ... * 16 = ... << 4
                ____.bipush(16)
                    .imul().__(),

                ____.bipush(4)
                    .ishl().__()
            },
            {   // ... * 32 = ... << 5
                ____.bipush(32)
                    .imul().__(),

                ____.bipush(5)
                    .ishl().__()
            },
            {   // ... * 64 = ... << 6
                ____.bipush(64)
                    .imul().__(),

                ____.bipush(6)
                    .ishl().__()
            },
            {   // ... * 128 = ... << 7
                ____.sipush(128)
                    .imul().__(),

                ____.bipush(7)
                    .ishl().__()
            },
            {   // ... * 256 = ... << 8
                ____.sipush(256)
                    .imul().__(),

                ____.bipush(8)
                    .ishl().__()
            },
            {   // ... * 512 = ... << 9
                ____.sipush(512)
                    .imul().__(),

                ____.bipush(9)
                    .ishl().__()
            },
            {   // ... * 1024 = ... << 10
                ____.sipush(1024)
                    .imul().__(),

                ____.bipush(10)
                    .ishl().__()
            },
            {   // ... * 2048 = ... << 11
                ____.sipush(2048)
                    .imul().__(),

                ____.bipush(11)
                    .ishl().__()
            },
            {   // ... * 4096 = ... << 12
                ____.sipush(4096)
                    .imul().__(),

                ____.bipush(12)
                    .ishl().__()
            },
            {   // ... * 8192 = ... << 13
                ____.sipush(8192)
                    .imul().__(),

                ____.bipush(13)
                    .ishl().__()
            },
            {   // ... * 16384 = ... << 14
                ____.sipush(16384)
                    .imul().__(),

                ____.bipush(14)
                    .ishl().__()
            },
            {   // ... * 32768 = ... << 15
                ____.ldc(32768)
                    .imul().__(),

                ____.bipush(15)
                    .ishl().__()
            },
            {   // ... * 65536 = ... << 16
                ____.ldc(65536)
                    .imul().__(),

                ____.bipush(16)
                    .ishl().__()
            },
            {   // ... * 16777216 = ... << 24
                ____.ldc(16777216)
                    .imul().__(),

                ____.bipush(24)
                    .ishl().__()
            },
            {   // ... * -1L = -...
                ____.ldc2_w(-1L)
                    .lmul().__(),

                ____.lneg().__()
            },
            {   // ... * 0L = 0L
                ____.lconst_0()
                    .lmul().__(),

                ____.pop2()
                    .lconst_0().__()
            },
            {   // ... * 1L = ...
                ____.lconst_1()
                    .lmul().__(),
            },
            {   // ... * 2L = ... << 1
                ____.ldc2_w(2L)
                    .lmul().__(),

                ____.iconst_1()
                    .lshl().__()
            },
            {   // ... * 4L = ... << 2
                ____.ldc2_w(4L)
                    .lmul().__(),

                ____.iconst_2()
                    .lshl().__()
            },
            {   // ... * 8L = ... << 3
                ____.ldc2_w(8L)
                    .lmul().__(),

                ____.iconst_3()
                    .lshl().__()
            },
            {   // ... * 16L = ... << 4
                ____.ldc2_w(16L)
                    .lmul().__(),

                ____.bipush(4)
                    .lshl().__()
            },
            {   // ... * 32L = ... << 5
                ____.ldc2_w(32L)
                    .lmul().__(),

                ____.bipush(5)
                    .lshl().__()
            },
            {   // ... * 64L = ... << 6
                ____.ldc2_w(64L)
                    .lmul().__(),

                ____.bipush(6)
                    .lshl().__()
            },
            {   // ... * 128L = ... << 7
                ____.ldc2_w(128L)
                    .lmul().__(),

                ____.bipush(7)
                    .lshl().__()
            },
            {   // ... * 256L = ... << 8
                ____.ldc2_w(256L)
                    .lmul().__(),

                ____.bipush(8)
                    .lshl().__()
            },
            {   // ... * 512L = ... << 9
                ____.ldc2_w(512L)
                    .lmul().__(),

                ____.bipush(9)
                    .lshl().__()
            },
            {   // ... * 1024L = ... << 10
                ____.ldc2_w(1024L)
                    .lmul().__(),

                ____.bipush(10)
                    .lshl().__()
            },
            {   // ... * 2048L = ... << 11
                ____.ldc2_w(2048L)
                    .lmul().__(),

                ____.bipush(11)
                    .lshl().__()
            },
            {   // ... * 4096L = ... << 12
                ____.ldc2_w(4096L)
                    .lmul().__(),

                ____.bipush(12)
                    .lshl().__()
            },
            {   // ... * 8192L = ... << 13
                ____.ldc2_w(8192L)
                    .lmul().__(),

                ____.bipush(13)
                    .lshl().__()
            },
            {   // ... * 16384L = ... << 14
                ____.ldc2_w(16384L)
                    .lmul().__(),

                ____.bipush(14)
                    .lshl().__()
            },
            {   // ... * 32768L = ... << 15
                ____.ldc2_w(32768L)
                    .lmul().__(),

                ____.bipush(15)
                    .lshl().__()
            },
            {   // ... * 65536LL = ... << 16
                ____.ldc2_w(65536L)
                    .lmul().__(),

                ____.bipush(16)
                    .lshl().__()
            },
            {   // ... * 16777216L = ... << 24
                ____.ldc2_w(16777216L)
                    .lmul().__(),

                ____.bipush(24)
                    .lshl().__()
            },
            {   // ... * 4294967296L = ... << 32
                ____.ldc2_w(4294967296L)
                    .lmul().__(),

                ____.bipush(32)
                    .lshl().__()
            },
            {   // ... * -1f = -...
                ____.ldc(-1f)
                    .fmul().__(),

                ____.fneg().__()
            },
            // Not valid for -0.0 and for NaN.
//            {   // ... * 0f = 0f
//                ____.fconst_0()
//                    .fmul().__(),
//
//                ____.pop()
//                    .fconst_0().__()
//            },
            {   // ... * 1f = ...
                ____.fconst_1()
                    .fmul().__(),
            },
            {   // ... * -1d = -...
                ____.ldc2_w(-1.)
                    .dmul().__(),

                ____.dneg().__()
            },
            // Not valid for -0.0 and for NaN.
//            {   // ... * 0d = 0d
//                ____.dconst_0()
//                    .dmul().__(),
//
//                ____.pop2()
//                    .dconst_0().__()
//            },
            {   // ... * 1d = ...
                ____.dconst_1()
                    .dmul().__(),
            },
            {   // ... / -1 = -...
                ____.iconst_m1()
                    .idiv().__(),

                ____.ineg().__()
            },
            {   // ... / 1 = ...
                ____.iconst_1()
                    .idiv().__(),
            },
            // Not valid for negative values.
//            {   // ... / 2 = ... >> 1
//                ____.iconst_2()
//                    .idiv().__(),
//
//                ____.iconst_1()
//                    .ishr().__()
//            },
//            {   // ... / 4 = ... >> 2
//                ____.iconst_4()
//                    .idiv().__(),
//
//                ____.iconst_2()
//                    .ishr().__()
//            },
//            {   // ... / 8 = ... >> 3
//                ____.bipush(8)
//                    .idiv().__(),
//
//                ____.iconst_3()
//                    .ishr().__()
//            },
//            {   // ... / 16 = ... >> 4
//                ____.bipush(16)
//                    .idiv().__(),
//
//                ____.bipush(4)
//                    .ishr().__()
//            },
//            {   // ... / 32 = ... >> 5
//                ____.bipush(32)
//                    .idiv().__(),
//
//                ____.bipush(5)
//                    .ishr().__()
//            },
//            {   // ... / 64 = ... >> 6
//                ____.bipush(64)
//                    .idiv().__(),
//
//                ____.bipush(6)
//                    .ishr().__()
//            },
//            {   // ... / 128 = ... >> 7
//                ____.sipush(128)
//                    .idiv().__(),
//
//                ____.bipush(7)
//                    .ishr().__()
//            },
//            {   // ... / 256 = ... >> 8
//                ____.sipush(256)
//                    .idiv().__(),
//
//                ____.bipush(8)
//                    .ishr().__()
//            },
//            {   // ... / 512 = ... >> 9
//                ____.sipush(512)
//                    .idiv().__(),
//
//                ____.bipush(9)
//                    .ishr().__()
//            },
//            {   // ... / 1024 = ... >> 10
//                ____.sipush(1024)
//                    .idiv().__(),
//
//                ____.bipush(10)
//                    .ishr().__()
//            },
//            {   // ... / 2048 = ... >> 11
//                ____.sipush(2048)
//                    .idiv().__(),
//
//                ____.bipush(11)
//                    .ishr().__()
//            },
//            {   // ... / 4096 = ... >> 12
//                ____.sipush(4096)
//                    .idiv().__(),
//
//                ____.bipush(12)
//                    .ishr().__()
//            },
//            {   // ... / 8192 = ... >> 13
//                ____.sipush(8192)
//                    .idiv().__(),
//
//                ____.bipush(13)
//                    .ishr().__()
//            },
//            {   // ... / 16384 = ... >> 14
//                ____.sipush(16384)
//                    .idiv().__(),
//
//                ____.bipush(14)
//                    .ishr().__()
//            },
//            {   // ... / 32768 = ... >> 15
//                ____.ldc(32768)
//                    .idiv().__(),
//
//                ____.bipush(15)
//                    .ishr().__()
//            },
//            {   // ... / 65536 = ... >> 16
//                ____.ldc(65536)
//                    .idiv().__(),
//
//                ____.bipush(16)
//                    .ishr().__()
//            },
//            {   // ... / 16777216 = ... >> 24
//                ____.ldc(16777216)
//                    .idiv().__(),
//
//                ____.bipush(24)
//                    .ishr().__()
//            },
            {   // ... / -1L = -...
                ____.ldc2_w(-1L)
                    .ldiv().__(),

                ____.lneg().__()
            },
            {   // ... / 1L = ...
                ____.lconst_1()
                    .ldiv().__(),
            },
            // Not valid for negative values.
//            {   // ... / 2L = ... >> 1
//                ____.ldc2_w(2L)
//                    .ldiv().__(),
//
//                ____.iconst_1()
//                    .lshr().__()
//            },
//            {   // ... / 4L = ... >> 2
//                ____.ldc2_w(4L)
//                    .ldiv().__(),
//
//                ____.iconst_2()
//                    .lshr().__()
//            },
//            {   // ... / 8L = ... >> 3
//                ____.ldc2_w(8L)
//                    .ldiv().__(),
//
//                ____.iconst_3()
//                    .lshr().__()
//            },
//            {   // ... / 16L = ... >> 4
//                ____.ldc2_w(16L)
//                    .ldiv().__(),
//
//                ____.bipush(4)
//                    .lshr().__()
//            },
//            {   // ... / 32L = ... >> 5
//                ____.ldc2_w(32L)
//                    .ldiv().__(),
//
//                ____.bipush(5)
//                    .lshr().__()
//            },
//            {   // ... / 64L = ... >> 6
//                ____.ldc2_w(64L)
//                    .ldiv().__(),
//
//                ____.bipush(6)
//                    .lshr().__()
//            },
//            {   // ... / 128L = ... >> 7
//                ____.ldc2_w(128L)
//                    .ldiv().__(),
//
//                ____.bipush(7)
//                    .lshr().__()
//            },
//            {   // ... / 256L = ... >> 8
//                ____.ldc2_w(256L)
//                    .ldiv().__(),
//
//                ____.bipush(8)
//                    .lshr().__()
//            },
//            {   // ... / 512L = ... >> 9
//                ____.ldc2_w(512L)
//                    .ldiv().__(),
//
//                ____.bipush(9)
//                    .lshr().__()
//            },
//            {   // ... / 1024L = ... >> 10
//                ____.ldc2_w(1024L)
//                    .ldiv().__(),
//
//                ____.bipush(10)
//                    .lshr().__()
//            },
//            {   // ... / 2048L = ... >> 11
//                ____.ldc2_w(2048L)
//                    .ldiv().__(),
//
//                ____.bipush(11)
//                    .lshr().__()
//            },
//            {   // ... / 4096L = ... >> 12
//                ____.ldc2_w(4096L)
//                    .ldiv().__(),
//
//                ____.bipush(12)
//                    .lshr().__()
//            },
//            {   // ... / 8192L = ... >> 13
//                ____.ldc2_w(8192L)
//                    .ldiv().__(),
//
//                ____.bipush(13)
//                    .lshr().__()
//            },
//            {   // ... / 16384L = ... >> 14
//                ____.ldc2_w(16384L)
//                    .ldiv().__(),
//
//                ____.bipush(14)
//                    .lshr().__()
//            },
//            {   // ... / 32768L = ... >> 15
//                ____.ldc2_w(32768L)
//                    .ldiv().__(),
//
//                ____.bipush(15)
//                    .lshr().__()
//            },
//            {   // ... / 65536LL = ... >> 16
//                ____.ldc2_w(65536L)
//                    .ldiv().__(),
//
//                ____.bipush(16)
//                    .lshr().__()
//            },
//            {   // ... / 16777216L = ... >> 24
//                ____.ldc2_w(16777216L)
//                    .ldiv().__(),
//
//                ____.bipush(24)
//                    .lshr().__()
//            },
//            {   // ... / 4294967296L = ... >> 32
//                ____.ldc2_w(4294967296L)
//                    .ldiv().__(),
//
//                ____.bipush(32)
//                    .lshr().__()
//            },
            {   // ... / -1f = -...
                ____.ldc(-1f)
                    .fdiv().__(),

                ____.fneg().__()
            },
            {   // ... / 1f = ...
                ____.fconst_1()
                    .fdiv().__(),
            },
            {   // ... / -1d = -...
                ____.ldc2_w(-1.)
                    .ddiv().__(),

                ____.dneg().__()
            },
            {   // ... / 1d = ...
                ____.dconst_1()
                    .ddiv().__(),
            },
            {   // ... % 1 = 0
                ____.iconst_1()
                    .irem().__(),

                ____.pop()
                    .iconst_0().__()
            },
            // Not valid for negative values.
//            {   // ... % 2 = ... & 0x1
//                ____.iconst_2()
//                    .irem().__(),
//
//                ____.iconst_1()
//                    .iand().__()
//            },
//            {   // ... % 4 = ... & 0x3
//                ____.iconst_4()
//                    .irem().__(),
//
//                ____.iconst_3()
//                    .iand().__()
//            },
//            {   // ... % 8 = ... & 0x07
//                ____.bipush(8)
//                    .irem().__(),
//
//                ____.bipush(0x07)
//                    .iand().__()
//            },
//            {   // ... % 16 = ... & 0x0f
//                ____.bipush(16)
//                    .irem().__(),
//
//                ____.bipush(0x0f)
//                    .iand().__()
//            },
//            {   // ... % 32 = ... & 0x1f
//                ____.bipush(32)
//                    .irem().__(),
//
//                ____.bipush(0x1f)
//                    .iand().__()
//            },
//            {   // ... % 64 = ... & 0x3f
//                ____.bipush(64)
//                    .irem().__(),
//
//                ____.bipush(0x3f)
//                    .iand().__()
//            },
//            {   // ... % 128 = ... & 0x7f
//                ____.sipush(128)
//                    .irem().__(),
//
//                ____.bipush(0x7f)
//                    .iand().__()
//            },
//            {   // ... % 256 = ... & 0x00ff
//                ____.sipush(256)
//                    .irem().__(),
//
//                ____.sipush(0x00ff)
//                    .iand().__()
//            },
//            {   // ... % 512 = ... & 0x01ff
//                ____.sipush(512)
//                    .irem().__(),
//
//                ____.sipush(0x01ff)
//                    .iand().__()
//            },
//            {   // ... % 1024 = ... & 0x03ff
//                ____.sipush(1024)
//                    .irem().__(),
//
//                ____.sipush(0x03ff)
//                    .iand().__()
//            },
//            {   // ... % 2048 = ... & 0x07ff
//                ____.sipush(2048)
//                    .irem().__(),
//
//                ____.sipush(0x07ff)
//                    .iand().__()
//            },
//            {   // ... % 4096 = ... & 0x0fff
//                ____.sipush(4096)
//                    .irem().__(),
//
//                ____.sipush(0x0fff)
//                    .iand().__()
//            },
//            {   // ... % 8192 = ... & 0x1fff
//                ____.sipush(8192)
//                    .irem().__(),
//
//                ____.sipush(0x1fff)
//                    .iand().__()
//            },
//            {   // ... % 16384 = ... & 0x3fff
//                ____.sipush(16384)
//                    .irem().__(),
//
//                ____.sipush(0x3fff)
//                    .iand().__()
//            },
            {   // ... % 1L = 0L
                ____.lconst_1()
                    .lrem().__(),

                ____.pop2()
                    .lconst_0().__()
            },
//            {   // ... % 1f = 0f
//                ____.fconst_1()
//                    .frem().__(),
//
//                ____.pop()
//                    .fconst_0().__()
//            },
//            {   // ... % 1d = 0d
//                ____.dconst_1()
//                    .drem().__(),
//
//                ____.pop2()
//                    .dconst_0().__()
//            },
            {   // -(-...) = ...
                ____.ineg()
                    .ineg().__(),
            },
            {   // -(-...) = ...
                ____.lneg()
                    .lneg().__(),
            },
            {   // -(-...) = ...
                ____.fneg()
                    .fneg().__(),
            },
            {   // -(-...) = ...
                ____.dneg()
                    .dneg().__(),
            },
            {   // +(-...) = -...
                ____.ineg()
                    .iadd().__(),

                ____.isub().__()
            },
            {   // +(-...) = -...
                ____.lneg()
                    .ladd().__(),

                ____.lsub().__()
            },
            {   // +(-...) = -...
                ____.fneg()
                    .fadd().__(),

                ____.fsub().__()
            },
            {   // +(-...) = -...
                ____.dneg()
                    .dadd().__(),

                ____.dsub().__()
            },
            {   // ... << 0 = ...
                ____.iconst_0()
                    .ishl().__(),
            },
            {   // ... << 0 = ...
                ____.iconst_0()
                    .lshl().__(),
            },
            {   // ... >> 0 = ...
                ____.iconst_0()
                    .ishr().__(),
            },
            {   // ... >> 0 = ...
                ____.iconst_0()
                    .lshr().__(),
            },
            {   // ... >>> 0 = ...
                ____.iconst_0()
                    .iushr().__(),
            },
            {   // ... >>> 0 = ...
                ____.iconst_0()
                    .lushr().__(),
            },
            {   // ... & -1 = ...
                ____.iconst_m1()
                    .iand().__(),
            },
            {   // ... & 0 = 0
                ____.iconst_0()
                    .iand().__(),

                ____.pop()
                    .iconst_0().__()
            },
            {   // ... & -1L = ...
                ____.ldc2_w(-1L)
                    .land().__(),
            },
            {   // ... & 0L = 0L
                ____.lconst_0()
                    .land().__(),

                ____.pop2()
                    .lconst_0().__()
            },
            {   // ... | -1 = -1
                ____.iconst_m1()
                    .ior().__(),

                ____.pop()
                    .iconst_m1().__()
            },
            {   // ... | 0 = ...
                ____.iconst_0()
                   .ior().__(),
            },
            {   // ... | -1L = -1L
                ____.ldc2_w(-1L)
                    .land().__(),

                ____.pop2()
                    .ldc2_w(-1L).__()
            },
            {   // ... | 0L = ...
                ____.lconst_0()
                    .lor().__(),
            },
            {   // ... ^ 0 = ...
                ____.iconst_0()
                    .ixor().__(),
            },
            {   // ... ^ 0L = ...
                ____.lconst_0()
                    .lxor().__(),
            },
            {   // (... & 0x0000ff00) >> 8 = (... >> 8) & 0xff
                ____.ldc(0x0000ff00)
                    .iand()
                    .bipush(8)
                    .ishr().__(),

                ____.bipush(8)
                    .ishr()
                    .sipush(0xff)
                    .iand().__()
            },
            {   // (... & 0x0000ff00) >>> 8 = (... >>> 8) & 0xff
                ____.ldc(0x0000ff00)
                    .iand()
                    .bipush(8)
                    .iushr().__(),

                ____.bipush(8)
                    .iushr()
                    .sipush(0xff)
                    .iand().__()
            },
            {   // (... & 0x00ff0000) >> 16 = (... >> 16) & 0xff
                ____.ldc(0x00ff0000)
                    .iand()
                    .bipush(16)
                    .ishr().__(),

                ____.bipush(16)
                    .ishr()
                    .sipush(0xff)
                    .iand().__()
            },
            {   // (... & 0x00ff0000) >>> 16 = (... >>> 16) & 0xff
                ____.ldc(0x00ff0000)
                    .iand()
                    .bipush(16)
                    .iushr().__(),

                ____.bipush(16)
                    .iushr()
                    .sipush(0xff)
                    .iand().__()
            },
            {   // (... & 0xff000000) >> 24 = ... >> 24
                ____.ldc(0xff000000)
                    .iand()
                    .bipush(24)
                    .ishr().__(),

                ____.bipush(24)
                    .ishr().__()
            },
            {   // (... & 0xffff0000) >> 16 = ... >> 16
                ____.ldc(0xffff0000)
                    .iand()
                    .bipush(16)
                    .ishr().__(),

                ____.bipush(16)
                    .ishr().__()
            },
            {   // (... & 0xffff0000) >>> 16 = ... >>> 16
                ____.ldc(0xffff0000)
                    .iand()
                    .bipush(16)
                    .iushr().__(),

                ____.bipush(16)
                    .iushr().__()
            },
            {   // (... >> 24) & 0xff = ... >>> 24
                ____.bipush(24)
                    .ishr()
                    .sipush(0xff)
                    .iand().__(),

                ____.bipush(24)
                    .iushr().__()
            },
            {   // (... >>> 24) & 0xff = ... >>> 24
                ____.bipush(24)
                    .iushr()
                    .sipush(0xff)
                    .iand().__(),

                ____.bipush(24)
                    .iushr().__()
            },
            {   // (byte)(... & 0x000000ff) = (byte)...
                ____.sipush(0xff)
                    .iand()
                    .i2b().__(),

                ____.i2b().__()
            },
            {   // (char)(... & 0x0000ffff) = (char)...
                ____.ldc(0x0000ffff)
                    .iand()
                    .i2c().__(),

                ____.i2c().__()
            },
            {   // (short)(... & 0x0000ffff) = (short)...
                ____.ldc(0x0000ffff)
                    .iand()
                    .i2s().__(),

                ____.i2s().__()
            },
            // The Dalvik VM on Android 4.4 throws a VFY error or crashes if
            // the byte/short cast is removed before an array store.
//            {   // (byte)(... >> 24) = ... >> 24
//                ____.bipush(24)
//                    .ishr()
//                    .i2b().__(),
//
//                ____.bipush(24)
//                    .ishr().__()
//            },
//            {   // (byte)(... >>> 24) = ... >> 24
//                ____.bipush(24)
//                    .iushr()
//                    .i2b().__(),
//
//                ____.bipush(24)
//                    .ishr().__()
//            },
//            {   // (char)(... >> 16) = ... >>> 16
//                ____.bipush(16)
//                    .ishr()
//                    .i2c().__(),
//
//                ____.bipush(16)
//                    .iushr().__()
//            },
//            {   // (char)(... >>> 16) = ... >>> 16
//                ____.bipush(16)
//                    .iushr()
//                    .i2c().__(),
//
//                ____.bipush(16)
//                    .iushr().__()
//            },
//            {   // (short)(... >> 16) = ... >> 16
//                ____.bipush(16)
//                    .ishr()
//                    .i2s().__(),
//
//                ____.bipush(16)
//                    .ishr().__()
//            },
//            {   // (short)(... >>> 16) = ... >> 16
//                ____.bipush(16)
//                    .iushr()
//                    .i2s().__(),
//
//                ____.bipush(16)
//                    .ishr().__()
//            },
            {   // ... << 24 >> 24 = (byte)...
                ____.bipush(24)
                    .ishl()
                    .bipush(24)
                    .ishr().__(),

                ____.i2b().__()
            },
            {   // ... << 16 >>> 16 = (char)...
                ____.bipush(16)
                    .ishl()
                    .bipush(16)
                    .iushr().__(),

                ____.i2c().__()
            },
            {   // ... << 16 >> 16 = (short)...
                ____.bipush(16)
                    .ishl()
                    .bipush(16)
                    .ishr().__(),

                ____.i2s().__()
            },
            {   // ... << 32 >> 32 = (long)(int)...
                ____.bipush(32)
                    .lshl()
                    .bipush(32)
                    .lshr().__(),

                ____.l2i()
                    .i2l().__()
            },
            {   // (int)(... & 0x00000000ffffffffL) = (int)...
                ____.ldc2_w(0x00000000ffffffffL)
                    .land()
                    .l2i().__(),

                ____.l2i().__()
            },
            {   // (... & 0xffffffff00000000L) >> 32 = ... >> 32
                ____.ldc2_w(0xffffffff00000000L)
                    .land()
                    .bipush(32)
                    .lshr().__(),

                ____.bipush(32)
                    .lshr().__()
            },
            {   // (... & 0xffffffff00000000L) >>> 32 = ... >>> 32
                ____.ldc2_w(0xffffffff00000000L)
                    .land()
                    .bipush(32)
                    .lushr().__(),

                ____.bipush(32)
                    .lushr().__()
            },
            {   // ... += 0 = nothing
                ____.iinc(X, 0).__(),
            },
        };

        FIELD_SEQUENCES = new Instruction[][][]
        {
            {   // getfield/putfield = nothing
                ____.aload(X)
                    .aload(X)
                    .getfield(Y)
                    .putfield(Y).__(),
            },
//            {   // putfield_L/putfield_L = pop2_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(FIELD_J)
//                    .aload(X)
//                    // ...
//                    .putfield(FIELD_J).__(),
//
//                ____.aload(X)
//                    // ...
//                    .pop2()
//                    // ...
//                    .putfield(FIELD_J).__()
//            },
//            {   // putfield_D/putfield_D = pop2_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(FIELD_D)
//                    .aload(X)
//                    // ...
//                    .putfield(FIELD_D).__(),
//
//                ____.aload(X)
//                    // ...
//                    .pop2()
//                    // ...
//                    .putfield(FIELD_D).__()
//            },
//            {   // putfield/putfield = pop_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(Y)
//                    .aload(X)
//                    // ...
//                    .putfield(Y).__(),
//
//                ____.aload(X)
//                    // ...
//                    .pop()
//                    // ...
//                    .putfield(Y).__()
//            },
//            {   // putfield_L/getfield_L = dup2_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(FIELD_J)
//                    .aload(X)
//                    .getfield(FIELD_J).__(),
//
//                ____.aload(X)
//                    // ...
//                    .dup2_x1()
//                    .putfield(FIELD_J).__()
//            },
//            {   // putfield_D/getfield_D = dup2_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(FIELD_D)
//                    .aload(X)
//                    .getfield(FIELD_D).__(),
//
//                ____.aload(X)
//                    // ...
//                    .dup2_x1()
//                    .putfield(FIELD_D).__()
//            },
//            {   // putfield/getfield = dup_x1/putfield
//                ____.aload(X)
//                    // ...
//                    .putfield(Y)
//                    .aload(X)
//                    .getfield(Y).__(),
//
//                ____.aload(X)
//                    // ...
//                    .dup_x1()
//                    .putfield(Y).__()
//            },
            {   // getstatic/putstatic = nothing
                ____.getstatic(X)
                    .putstatic(X).__(),
            },
            {   // getstatic_L/getstatic_L = getstatic/dup2
                ____.getstatic(FIELD_J)
                    .getstatic(FIELD_J).__(),

                ____.getstatic(FIELD_J)
                    .dup2().__()
            },
            {   // getstatic_D/getstatic_D = getstatic/dup2
                ____.getstatic(FIELD_D)
                    .getstatic(FIELD_D).__(),

                ____.getstatic(FIELD_D)
                    .dup2().__()
            },
            {   // getstatic/getstatic = getstatic/dup
                ____.getstatic(X)
                    .getstatic(X).__(),

                ____.getstatic(X)
                    .dup().__()
            },
            {   // putstatic_L/putstatic_L = pop2/putstatic
                ____.putstatic(FIELD_J)
                    .putstatic(FIELD_J).__(),

                ____.pop2()
                    .putstatic(FIELD_J).__()
            },
            {   // putstatic_D/putstatic_D = pop2/putstatic
                ____.putstatic(FIELD_D)
                    .putstatic(FIELD_D).__(),

                ____.pop2()
                    .putstatic(FIELD_D).__()
            },
            {   // putstatic/putstatic = pop/putstatic
                ____.putstatic(X)
                    .putstatic(X).__(),

                ____.pop()
                    .putstatic(X).__()
            },
            {   // putstatic_L/getstatic_L = dup2/putstatic
                ____.putstatic(FIELD_J)
                    .getstatic(FIELD_J).__(),

                ____.dup2()
                    .putstatic(FIELD_J).__()
            },
            {   // putstatic_D/getstatic_D = dup2/putstatic
                ____.putstatic(FIELD_D)
                    .getstatic(FIELD_D).__(),

                ____.dup2()
                    .putstatic(FIELD_D).__()
            },
            {   // putstatic/getstatic = dup/putstatic
                ____.putstatic(X)
                    .getstatic(X).__(),

                ____.dup()
                    .putstatic(X).__()
            },
            {   // L i L: getfield_L/iload/getfield_L = iload/getfield_L/dup2_x1
                ____.aload(A)
                    .getfield(FIELD_J)
                    .iload(B)
                    .aload(A)
                    .getfield(FIELD_J).__(),

                ____.iload(B)
                    .aload(A)
                    .getfield(FIELD_J)
                    .dup2_x1().__()
            },
            {   // D i D: getfield_D/iload/getfield_D = iload/getfield_D/dup2_x1
                ____.aload(A)
                    .getfield(FIELD_D)
                    .iload(B)
                    .aload(A)
                    .getfield(FIELD_D).__(),

                ____.iload(B)
                    .aload(A)
                    .getfield(FIELD_D)
                    .dup2_x1().__()
            },
            {   // X i X (e.g. X[i] = X[.] ...): getfield/iload/getfield = iload/getfield/dup_x1
                ____.aload(A)
                    .getfield(X)
                    .iload(B)
                    .aload(A)
                    .getfield(X).__(),

                ____.iload(B)
                    .aload(A)
                    .getfield(X)
                    .dup_x1().__()
            },
            {   // L i L: getstatic_L/iload/getstatic_L = iload/getstatic_L/dup2_x1
                ____.getstatic(FIELD_J)
                    .iload(A)
                    .getstatic(FIELD_J).__(),

                ____.iload(A)
                    .getstatic(FIELD_J)
                    .dup2_x1().__()
            },
            {   // D i D: getstatic_D/iload/getstatic_D = iload/getstatic_D/dup2_x1
                ____.getstatic(FIELD_D)
                    .iload(A)
                    .getstatic(FIELD_D).__(),

                ____.iload(A)
                    .getstatic(FIELD_D)
                    .dup2_x1().__()
            },
            {   // X i X (e.g. X[i] = X[.] ...): getstatic/iload/getstatic = iload/getstatic/dup_x1
                ____.getstatic(X)
                    .iload(A)
                    .getstatic(X).__(),

                ____.iload(A)
                    .getstatic(X)
                    .dup_x1().__()
            },
            {   // X[i] j X[i] (e.g. X[i][j] = X[i][.] ...): getfield/iload/aaload/iload/getfield/iload/aaload = iload/getfield//iload/aaload/iload/dup_x1
                ____.aload(A)
                    .getfield(X)
                    .iload(B)
                    .aaload()
                    .iload(C)
                    .aload(A)
                    .getfield(X)
                    .iload(B)
                    .aaload().__(),

                ____.iload(C)
                    .aload(A)
                    .getfield(X)
                    .iload(B)
                    .aaload()
                    .dup_x1().__()
            },
            {   // X[i] j X[i] (e.g. X[i][j] = X[i][.] ...): getstatic/iload/aaload/iload/getstatic/iload/aaload = iload/getstatic//iload/aaload/iload/dup_x1
                ____.getstatic(X)
                    .iload(B)
                    .aaload()
                    .iload(C)
                    .getstatic(X)
                    .iload(B)
                    .aaload().__(),

                ____.iload(C)
                    .getstatic(X)
                    .iload(B)
                    .aaload()
                    .dup_x1().__()
            },
        };

        CAST_SEQUENCES = new Instruction[][][]
        {
            {   // (byte)(byte)... = (byte)...
                ____.i2b()
                    .i2b().__(),

                ____.i2b().__()
            },
            {   // (byte)(char)... = (byte)...
                ____.i2c()
                    .i2b().__(),

                ____.i2b().__()
            },
            {   // (byte)(short)... = (byte)...
                ____.i2s()
                    .i2b().__(),

                ____.i2b().__()
            },
            {   // (char)(char)... = (char)...
                ____.i2c()
                    .i2c().__(),

                ____.i2c().__()
            },
            {   // (char)(short)... = (char)...
                ____.i2s()
                    .i2c().__(),

                ____.i2c().__()
            },
//            {   // (short)(byte)... = (byte)...
//                ____.i2b()
//                    .i2s().__(),
//
//                ____.i2b().__()
//            },
            {   // (short)(char)... = (short)...
                ____.i2c()
                    .i2s().__(),

                ____.i2s().__()
            },
            {   // (short)(short)... = (short)...
                ____.i2s()
                    .i2s().__(),

                ____.i2s().__()
            },
            {   // (int)(long)... = ...
                ____.i2l()
                    .l2i().__(),
            },
            {   // (int)(double)... = ...
                ____.i2d()
                    .d2i().__(),
            },
            {   // (float)(double)... = (float)... for ints
                ____.i2d()
                    .d2f().__(),

                ____.i2f().__()
            },
            {   // (float)(double)... = (float)... for longs
                ____.l2d()
                    .d2f().__(),

                ____.l2f().__()
            },
            {   // (int)(double)... = (int)...
                ____.f2d()
                    .d2i().__(),

                ____.f2i().__()
            },
            {   // (long)(double)... = (long)...
                ____.f2d()
                    .d2l().__(),

                ____.f2l().__()
            },
            {   // (X)(X)... = (X)...
                ____.checkcast(X)
                    .checkcast(X).__(),

                ____.checkcast(X).__()
            },
            // Not handled correctly in all cases by VMs prior to Java 6...
//            {   // (byte)bytes[...] = bytes[...]
//                ____.baload()
//                    .i2b().__(),
//
//                ____.baload().__()
//            },
//            {   // (short)bytes[...] = bytes[...]
//                ____.baload()
//                     .i2s().__(),
//
//                ____.baload().__()
//            },
//            {   // (char)chars[...] = chars[...]
//                ____.caload()
//                    .i2c().__(),
//
//                ____.caload().__()
//            },
//            {   // (short)shorts[...] = shorts[...]
//                ____.saload()
//                    .i2s().__(),
//
//                ____.saload().__()
//            },
//            {   // bytes[...] = (byte)... = bytes[...] = ...
//                ____.i2b()
//                    .bastore().__(),
//
//                ____.bastore().__()
//            },
//            {   // chars[...] = (char)... = chars[...] = ...
//                ____.i2c()
//                    .castore().__(),
//
//                ____.castore().__()
//            },
//            {   // shorts[...] = (short)... = shorts[...] = ...
//                ____.i2s()
//                    .sastore().__(),
//
//                ____.sastore().__()
//            },
        };

        BRANCH_SEQUENCES = new Instruction[][][]
        {
            {   // goto +3 = nothing
                ____.goto_(3).__(),
            },
            {   // ifeq +3 = pop
                ____.ifeq(3).__(),

                ____.pop().__()
            },
            {   // ifne +3 = pop
                ____.ifne(3).__(),

                ____.pop().__()
            },
            {   // iflt +3 = pop
                ____.iflt(3).__(),

                ____.pop().__()
            },
            {   // ifge +3 = pop
                ____.ifge(3).__(),

                ____.pop().__()
            },
            {   // ifgt +3 = pop
                ____.ifgt(3).__(),

                ____.pop().__()
            },
            {   // ifle +3 = pop
                ____.ifle(3).__(),

                ____.pop().__()
            },
            {   // ificmpeq +3 = pop2
                ____.ificmpeq(3).__(),

                ____.pop2().__()
            },
            {   // ificmpne +3 = pop2
                ____.ificmpne(3).__(),

                ____.pop2().__()
            },
            {   // ificmplt +3 = pop2
                ____.ificmplt(3).__(),

                ____.pop2().__()
            },
            {   // ificmpge +3 = pop2
                ____.ificmpge(3).__(),

                ____.pop2().__()
            },
            {   // ificmpgt +3 = pop2
                ____.ificmpgt(3).__(),

                ____.pop2().__()
            },
            {   // ificmple +3 = pop2
                ____.ificmple(3).__(),

                ____.pop2().__()
            },
            {   // ifacmpeq +3 = pop2
                ____.ifacmpeq(3).__(),

                ____.pop2().__()
            },
            {   // ifacmpne +3 = pop2
                ____.ifacmpne(3).__(),

                ____.pop2().__()
            },
            {   // ifnull +3 = pop
                ____.ifnull(3).__(),

                ____.pop().__()
            },
            {   // ifnonnull +3 = pop
                ____.ifnonnull(3).__(),

                ____.pop().__()
            },
            {   // if (... == 0) = ifeq
                ____.iconst_0()
                    .ificmpeq(X).__(),

                ____.ifeq(X).__()
            },
            {   // if (0 == i) = iload/ifeq
                ____.iconst_0()
                    .iload(Y)
                    .ificmpeq(X).__(),

                ____.iload(Y)
                    .ifeq(X).__()
            },
            {   // if (0 == i) = getstatic/ifeq
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmpeq(X).__(),

                ____.getstatic(Y)
                    .ifeq(X).__()
            },
            {   // if (0 == i) = getfield/ifeq
                ____.iconst_0()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpeq(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifeq(X).__()
            },
            {   // if (... != 0) = ifne
                ____.iconst_0()
                    .ificmpne(X).__(),

                ____.ifne(X).__()
            },
            {   // if (0 != i) = iload/ifeq
                ____.iconst_0()
                    .iload(Y)
                    .ificmpne(X).__(),

                ____.iload(Y)
                    .ifne(X).__()
            },
            {   // if (0 != i) = getstatic/ifeq
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmpne(X).__(),

                ____.getstatic(Y)
                    .ifne(X).__()
            },
            {   // if (0 != i) = getfield/ifeq
                ____.iconst_0()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpne(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifne(X).__()
            },
            {   // if (... < 0) = iflt
                ____.iconst_0()
                    .ificmplt(X).__(),

                ____.iflt(X).__()
            },
            {   // if (... < 1) = ifle
                ____.iconst_1()
                    .ificmplt(X).__(),

                ____.ifle(X).__()
            },
            {   // if (0 > i) = iload/iflt
                ____.iconst_0()
                    .iload(Y)
                    .ificmpgt(X).__(),

                ____.iload(Y)
                    .iflt(X).__()
            },
            {   // if (1 > i) = iload/ifle
                ____.iconst_1()
                    .iload(Y)
                    .ificmpgt(X).__(),

                ____.iload(Y)
                    .ifle(X).__()
            },
            {   // if (0 > i) = getstatic/iflt
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmpgt(X).__(),

                ____.getstatic(Y)
                    .iflt(X).__()
            },
            {   // if (1 > i) = getstatic/ifle
                ____.iconst_1()
                    .getstatic(Y)
                    .ificmpgt(X).__(),

                ____.getstatic(Y)
                    .ifle(X).__()
            },
            {   // if (0 > i) = getfield/iflt
                ____.iconst_0()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpgt(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .iflt(X).__()
            },
            {   // if (1 > i) = getfield/ifle
                ____.iconst_1()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpgt(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifle(X).__()
            },
            {   // if (... >= 0) = ifge
                ____.iconst_0()
                    .ificmpge(X).__(),

                ____.ifge(X).__()
            },
            {   // if (... >= 1) = ifgt
                ____.iconst_1()
                    .ificmpge(X).__(),

                ____.ifgt(X).__()
            },
            {   // if (0 <= i) = iload/ifge
                ____.iconst_0()
                    .iload(Y)
                    .ificmple(X).__(),

                ____.iload(Y)
                    .ifge(X).__()
            },
            {   // if (1 <= i) = iload/ifgt
                ____.iconst_1()
                    .iload(Y)
                    .ificmple(X).__(),

                ____.iload(Y)
                    .ifgt(X).__()
            },
            {   // if (0 <= i) = getstatic/ifge
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmple(X).__(),

                ____.getstatic(Y)
                    .ifge(X).__()
            },
            {   // if (1 <= i) = getstatic/ifgt
                ____.iconst_1()
                    .getstatic(Y)
                    .ificmple(X).__(),

                ____.getstatic(Y)
                    .ifgt(X).__()
            },
            {   // if (0 <= i) = getfield/ifge
                ____.iconst_0()
                .aload(Y)
                .getfield(Z)
                .ificmple(X).__(),

            ____.aload(Y)
                .getfield(Z)
                .ifge(X).__()
            },
            {   // if (1 <= i) = getfield/ifgt
                ____.iconst_1()
                    .aload(Y)
                    .getfield(Z)
                    .ificmple(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifgt(X).__()
            },
            {   // if (... > 0) = ifgt
                ____.iconst_0()
                    .ificmpgt(X).__(),

                ____.ifgt(X).__()
            },
            {   // if (... > -1) = ifge
                ____.iconst_m1()
                    .ificmpgt(X).__(),

                ____.ifge(X).__()
            },
            {   // if (0 < i) = iload/ifgt
                ____.iconst_0()
                    .iload(Y)
                    .ificmplt(X).__(),

                ____.iload(Y)
                    .ifgt(X).__()
            },
            {   // if (-1 < i) = iload/ifge
                ____.iconst_m1()
                    .iload(Y)
                    .ificmplt(X).__(),

                ____.iload(Y)
                    .ifge(X).__()
            },
            {   // if (0 < i) = getstatic/ifgt
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmplt(X).__(),

                ____.getstatic(Y)
                    .ifgt(X).__()
            },
            {   // if (-1 < i) = getstatic/ifge
                ____.iconst_m1()
                    .getstatic(Y)
                    .ificmplt(X).__(),

                ____.getstatic(Y)
                    .ifge(X).__()
            },
            {   // if (0 < i) = getfield/ifgt
                ____.iconst_0()
                    .aload(Y)
                    .getfield(Z)
                    .ificmplt(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifgt(X).__()
            },
            {   // if (-1 < i) = getfield/ifge
                ____.iconst_m1()
                    .aload(Y)
                    .getfield(Z)
                    .ificmplt(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifge(X).__()
            },
            {   // if (... <= 0) = ifle
                ____.iconst_0()
                    .ificmple(X).__(),

                ____.ifle(X).__()
            },
            {   // if (... <= -1) = iflt
                ____.iconst_m1()
                    .ificmple(X).__(),

                ____.iflt(X).__()
            },
            {   // if (0 >= i) = iload/ifle
                ____.iconst_0()
                    .iload(Y)
                    .ificmpge(X).__(),

                ____.iload(Y)
                    .ifle(X).__()
            },
            {   // if (-1 >= i) = iload/iflt
                ____.iconst_m1()
                    .iload(Y)
                    .ificmpge(X).__(),

                ____.iload(Y)
                    .iflt(X).__()
            },
            {   // if (0 >= i) = getstatic/ifle
                ____.iconst_0()
                    .getstatic(Y)
                    .ificmpge(X).__(),

                ____.getstatic(Y)
                    .ifle(X).__()
            },
            {   // if (-1 >= i) = getstatic/iflt
                ____.iconst_m1()
                    .getstatic(Y)
                    .ificmpge(X).__(),

                ____.getstatic(Y)
                    .iflt(X).__()
            },
            {   // if (0 >= i) = getfield/ifle
                ____.iconst_0()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpge(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifle(X).__()
            },
            {   // if (-1 >= i) = getfield/iflt
                ____.iconst_m1()
                    .aload(Y)
                    .getfield(Z)
                    .ificmpge(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .iflt(X).__()
            },
            {   // if (... == null) = ifnull
                ____.aconst_null()
                    .ifacmpeq(X).__(),

                ____.ifnull(X).__()
            },
            {   // if (null == a) = aload/ifnull
                ____.aconst_null()
                    .aload(Y)
                    .ifacmpeq(X).__(),

                ____.aload(Y)
                    .ifnull(X).__()
            },
            {   // if (null == a) = getstatic/ifnull
                ____.aconst_null()
                    .getstatic(Y)
                    .ifacmpeq(X).__(),

                ____.getstatic(Y)
                    .ifnull(X).__()
            },
            {   // if (null == a) = getfield/ifnull
                ____.aconst_null()
                    .aload(Y)
                    .getfield(Z)
                    .ifacmpeq(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifnull(X).__()
            },
            {   // if (... != null) = ifnonnull
                ____.aconst_null()
                    .ifacmpne(X).__(),

                ____.ifnonnull(X).__()
            },
            {   // if (null != a) = aload/ifnonnull
                ____.aconst_null()
                    .aload(Y)
                    .ifacmpne(X).__(),

                ____.aload(Y)
                    .ifnonnull(X).__()
            },
            {   // if (null != a) = getstatic/ifnonnull
                ____.aconst_null()
                    .getstatic(Y)
                    .ifacmpne(X).__(),

                ____.getstatic(Y)
                    .ifnonnull(X).__()
            },
            {   // if (null != a) = getfield/ifnonnull
                ____.aconst_null()
                    .aload(Y)
                    .getfield(Z)
                    .ifacmpne(X).__(),

                ____.aload(Y)
                    .getfield(Z)
                    .ifnonnull(X).__()
            },
            {   // iconst_0/ifeq = goto
                ____.iconst_0()
                    .ifeq(X).__(),

                ____.goto_(X).__()
            },
            {   // iconst/ifeq = nothing
                ____.iconst(A)
                    .ifeq(X).__(),
            },
            {   // bipush/ifeq = nothing
                ____.bipush(A)
                    .ifeq(X).__(),
            },
            {   // sipush/ifeq = nothing
                ____.sipush(A)
                    .ifeq(X).__(),
            },
            {   // iconst_0/ifne = nothing
                ____.iconst_0()
                    .ifne(X).__(),
            },
            {   // iconst/ifne = goto
                ____.iconst(A)
                    .ifne(X).__(),

                ____.goto_(X).__()
            },
            {   // bipush/ifne = goto
                ____.bipush(A)
                    .ifne(X).__(),

                ____.goto_(X).__()
            },
            {   // sipush/ifne = goto
                ____.sipush(A)
                    .ifne(X).__(),

                ____.goto_(X).__()
            },
            {   // iconst_0/iflt = nothing
                ____.iconst_0()
                    .iflt(X).__(),
            },
            {   // iconst_0/ifge = goto
                ____.iconst_0()
                    .ifge(X).__(),

                ____.goto_(X).__()
            },
            {   // iconst_0/ifgt = nothing
                ____.iconst_0()
                    .ifgt(X).__(),
            },
            {   // iconst_0/ifle = goto
                ____.iconst_0()
                    .ifle(X).__(),

                ____.goto_(X).__()
            },
            {   // aconst_null/ifnull = goto
                ____.aconst_null()
                    .ifnull(X).__(),

                ____.goto_(X).__()
            },
            {   // aconst_null/ifnonnul = nothing
                ____.aconst_null()
                    .ifnonnull(X).__(),
            },
            {   // ifeq/goto = ifne
                ____.ifeq(6)
                    .goto_(X).__(),

                ____.ifne(X).__()
            },
            {   // ifne/goto = ifeq
                ____.ifne(6)
                    .goto_(X).__(),

                ____.ifeq(X).__()
            },
            {   // iflt/goto = ifge
                ____.iflt(6)
                    .goto_(X).__(),

                ____.ifge(X).__()
            },
            {   // ifge/goto = iflt
                ____.ifge(6)
                    .goto_(X).__(),

                ____.iflt(X).__()
            },
            {   // ifgt/goto = ifle
                ____.ifgt(6)
                    .goto_(X).__(),

                ____.ifle(X).__()
            },
            {   // ifle/goto = ifgt
                ____.ifle(6)
                    .goto_(X).__(),

                ____.ifgt(X).__()
            },
            {   // ificmpeq/goto = ificmpne
                ____.ificmpeq(6)
                    .goto_(X).__(),

                ____.ificmpne(X).__()
            },
            {   // ificmpne/goto = ificmpeq
                ____.ificmpne(6)
                    .goto_(X).__(),

                ____.ificmpeq(X).__()
            },
            {   // ificmplt/goto = ificmpge
                ____.ificmplt(6)
                    .goto_(X).__(),

                ____.ificmpge(X).__()
            },
            {   // ificmpge/goto = ificmplt
                ____.ificmpge(6)
                    .goto_(X).__(),

                ____.ificmplt(X).__()
            },
            {   // ificmpgt/goto = ificmple
                ____.ificmpgt(6)
                    .goto_(X).__(),

                ____.ificmple(X).__()
            },
            {   // ificmple/goto = ificmpgt
                ____.ificmple(6)
                    .goto_(X).__(),

                ____.ificmpgt(X).__()
            },
            {   // ifacmpeq/goto = ifacmpne
                ____.ifacmpeq(6)
                    .goto_(X).__(),

                ____.ifacmpne(X).__()
            },
            {   // ifacmpne/goto = ifacmpeq
                ____.ifacmpne(6)
                    .goto_(X).__(),

                ____.ifacmpeq(X).__()
            },
            {   // ifnull/goto = ifnonnull
                ____.ifnull(6)
                    .goto_(X).__(),

                ____.ifnonnull(X).__()
            },
            {   // ifnonnull/goto = ifnull
                ____.ifnonnull(6)
                    .goto_(X).__(),

                ____.ifnull(X).__()
            },
//            {   // switch (...) { default: ... } = pop/goto ...
//                ____.tableswitch(A, X, Y, 0, new int[0]).__(),
//
//                ____.pop()
//                    .goto_(A).__()
//            },
//            {   // switch (...) { default: ... } = pop/goto ...
//                ____.lookupswitch(A, 0, new int[0], new int[0]).__(),
//
//                ____.pop()
//                    .goto_(A).__()
//            },
            {   // switch (...) { case/case/default: ... } = switch (...) { case/default: ... }
                ____.lookupswitch(A, new int[] { X, Y }, new int[] { A, B }).__(),

                ____.lookupswitch(A, new int[] { Y }, new int[] { B }).__()
            },
            {   // switch (...) { case/case/default: ... } = switch (...) { case/default: ... }
                ____.lookupswitch(B, new int[] { X, Y }, new int[] { A, B }).__(),

                ____.lookupswitch(B, new int[] { X }, new int[] { A }).__()
            },
            {   // switch (...) { case/case/case/default: ... } = switch (...) { case/case/default: ... }
                ____.lookupswitch(A, new int[] { X, Y, Z }, new int[] { A, B, C }).__(),

                ____.lookupswitch(A, new int[] { Y, Z }, new int[] { B, C }).__()
            },
            {   // switch (...) { case/case/case/default: ... } = switch (...) { case/case/default: ... }
                ____.lookupswitch(B, new int[] { X, Y, Z }, new int[] { A, B, C }).__(),

                ____.lookupswitch(B, new int[] { X, Z }, new int[] { A, C }).__()
            },
            {   // switch (...) { case/case/case/default: ... } = switch (...) { case/case/default: ... }
                ____.lookupswitch(C, new int[] { X, Y, Z }, new int[] { A, B, C }).__(),

                ____.lookupswitch(C, new int[] { X, Y }, new int[] { A, B }).__()
            },
//            {   // switch (...) { case ...: ...  default:  ... }
//                // = if (... == ...) ... else ...
//                ____.tableswitch(A, X, Y, 1, new int[] { B }).__(),
//
//                ____.sipush(X)
//                    .ificmpne(A)
//                    .goto_(B).__()
//            },
//            {   // switch (...) { case ...: ...  default:  ... }
//                // = if (... == ...) ... else ...
//                ____.lookupswitch(A, 1, new int[] { X }, new int[] { B }).__(),
//
//                ____.sipush(X)
//                    .ificmpne(A)
//                    .goto_(B).__()
//            }
        };

        OBJECT_SEQUENCES = new Instruction[][][]
        {
            {   // "...".equals("...") = X.class.equals(X.class) = true (ignoring class loader)
                ____.ldc_(A)
                    .ldc_(A)
                    .invokevirtual(EQUALS).__(),

                ____.iconst_1().__()
            },
            {   // ....equals(dup) = true (discarding any NullPointerException)
                ____.dup()
                    .invokevirtual(EQUALS).__(),

                ____.pop()
                    .iconst_1().__()
            },
            {   // object.equals(object) = true (ignoring implementation and discarding any NullPointerException)
                ____.aload(A)
                    .aload(A)
                    .invokevirtual(EQUALS).__(),

                ____.iconst_1().__()
            },
            {   // object.equals(object) = true (ignoring implementation and discarding any NullPointerException)
                ____.getstatic(A)
                    .getstatic(A)
                    .invokevirtual(EQUALS).__(),

                ____.iconst_1().__()
            },
            {   // object.equals(object) = true (ignoring implementation and discarding any NullPointerException)
                ____.aload(A)
                    .getfield(B)
                    .aload(A)
                    .getfield(B)
                    .invokevirtual(EQUALS).__(),

                ____.iconst_1().__()
            },
            {   // Boolean.valueOf(false) = Boolean.FALSE
                ____.iconst_0()
                    .invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;").__(),

                ____.getstatic(BOOLEAN, "FALSE", "Ljava/lang/Boolean;").__()
            },
            {   // Boolean.valueOf(true) = Boolean.TRUE
                ____.iconst_1()
                    .invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;").__(),

                ____.getstatic(BOOLEAN, "TRUE", "Ljava/lang/Boolean;").__()
            },
            {   // new Boolean(false) = Boolean.FALSE (ignoring identity)
                ____.new_(BOOLEAN)
                    .dup()
                    .iconst_0()
                    .invokespecial(BOOLEAN, "<init>", "(Z)V").__(),

                ____.getstatic(BOOLEAN, "FALSE", "Ljava/lang/Boolean;").__()
            },
            {   // new Boolean(true) = Boolean.TRUE (ignoring identity)
                ____.new_(BOOLEAN)
                    .dup()
                    .iconst_1()
                    .invokespecial(BOOLEAN, "<init>", "(Z)V").__(),

                ____.getstatic(BOOLEAN, "TRUE", "Ljava/lang/Boolean;").__()
            },
            {   // new Boolean(v) = Boolean.valueof(v) (ignoring identity)
                ____.new_(BOOLEAN)
                    .dup()
                    .iload(A)
                    .invokespecial(BOOLEAN, "<init>", "(Z)V").__(),

                ____.iload(A)
                    .invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;").__()
            },
            {   // new Boolean(s) = Boolean.valueof(s) (ignoring identity)
                ____.new_(BOOLEAN)
                    .dup()
                    .getstatic(FIELD_Z)
                    .invokespecial(BOOLEAN, "<init>", "(Z)V").__(),

                ____.getstatic(FIELD_Z)
                    .invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;").__()
            },
            {   // new Boolean(v.f) = Boolean.valueof(v.f) (ignoring identity)
                ____.new_(BOOLEAN)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_Z)
                    .invokespecial(BOOLEAN, "<init>", "(Z)V").__(),

                ____.aload(A)
                    .getfield(FIELD_Z)
                    .invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;").__()
            },
            {   // Boolean.FALSE.booleanValue() = false
                ____.getstatic(BOOLEAN, "FALSE", "Ljava/lang/Boolean;")
                    .invokevirtual(BOOLEAN_VALUE).__(),

                ____.iconst_0().__()
            },
            {   // Boolean.TRUE.booleanValue() = true
                ____.getstatic(BOOLEAN, "TRUE", "Ljava/lang/Boolean;")
                    .invokevirtual(BOOLEAN_VALUE).__(),

                ____.iconst_1().__()
            },
            {   // Boolean.valueOf(...).booleanValue() = nothing
                ____.invokestatic(BOOLEAN, "valueOf", "(Z)Ljava/lang/Boolean;")
                    .invokevirtual(BOOLEAN_VALUE).__(),
            },
            {   // new Byte(B) = Byte.valueof(B) (ignoring identity)
                ____.new_(BYTE)
                    .dup()
                    .iconst(A)
                    .invokespecial(BYTE, "<init>", "(B)V").__(),

                ____.iconst(A)
                    .invokestatic(BYTE, "valueOf", "(B)Ljava/lang/Byte;").__()
            },
            {   // new Byte(v) = Byte.valueof(v) (ignoring identity)
                ____.new_(BYTE)
                    .dup()
                    .iload(A)
                    .invokespecial(BYTE, "<init>", "(B)V").__(),

                ____.iload(A)
                    .invokestatic(BYTE, "valueOf", "(B)Ljava/lang/Byte;").__()
            },
            {   // new Byte(s) = Byte.valueof(s) (ignoring identity)
                ____.new_(BYTE)
                    .dup()
                    .getstatic(FIELD_B)
                    .invokespecial(BYTE, "<init>", "(B)V").__(),

                ____.getstatic(FIELD_B)
                    .invokestatic(BYTE, "valueOf", "(B)Ljava/lang/Byte;").__()
            },
            {   // new Byte(v.f) = Byte.valueof(v.f) (ignoring identity)
                ____.new_(BYTE)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_B)
                    .invokespecial(BYTE, "<init>", "(B)V").__(),

                ____.aload(A)
                    .getfield(FIELD_B)
                    .invokestatic(BYTE, "valueOf", "(B)Ljava/lang/Byte;").__()
            },
            {   // Byte.valueOf(...).byteValue() = nothing
                ____.invokestatic(BYTE, "valueOf", "(B)Ljava/lang/Byte;")
                    .invokevirtual(BYTE_VALUE).__(),
            },
            {   // new Character(C) = Character.valueof(C) (ignoring identity)
                ____.new_(CHARACTER)
                    .dup()
                    .iconst(A)
                    .invokespecial(CHARACTER, "<init>", "(C)V").__(),

                ____.iconst(A)
                    .invokestatic(CHARACTER, "valueOf", "(C)Ljava/lang/Character;").__()
            },
            {   // new Character(v) = Character.valueof(v) (ignoring identity)
                ____.new_(CHARACTER)
                    .dup()
                    .iload(A)
                    .invokespecial(CHARACTER, "<init>", "(C)V").__(),

                ____.iload(A)
                    .invokestatic(CHARACTER, "valueOf", "(C)Ljava/lang/Character;").__()
            },
            {   // new Character(s) = Character.valueof(s) (ignoring identity)
                ____.new_(CHARACTER)
                    .dup()
                    .getstatic(FIELD_C)
                    .invokespecial(CHARACTER, "<init>", "(C)V").__(),

                ____.getstatic(FIELD_C)
                    .invokestatic(CHARACTER, "valueOf", "(C)Ljava/lang/Character;").__()
            },
            {   // new Character(v.f) = Character.valueof(v.f) (ignoring identity)
                ____.new_(CHARACTER)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_C)
                    .invokespecial(CHARACTER, "<init>", "(C)V").__(),

                ____.aload(A)
                    .getfield(FIELD_C)
                    .invokestatic(CHARACTER, "valueOf", "(C)Ljava/lang/Character;").__()
            },
            {   // Character.valueOf(...).charValue() = nothing
                ____.invokestatic(CHARACTER, "valueOf", "(C)Ljava/lang/Character;")
                    .invokevirtual(CHAR_VALUE).__(),
            },
            {   // new Short(S) = Short.valueof(S) (ignoring identity)
                ____.new_(SHORT)
                    .dup()
                    .iconst(A)
                    .invokespecial(SHORT, "<init>", "(S)V").__(),

                ____.iconst(A)
                    .invokestatic(SHORT, "valueOf", "(S)Ljava/lang/Short;").__()
            },
            {   // new Short(v) = Short.valueof(v) (ignoring identity)
                ____.new_(SHORT)
                    .dup()
                    .iload(A)
                    .invokespecial(SHORT, "<init>", "(S)V").__(),

                ____.iload(A)
                    .invokestatic(SHORT, "valueOf", "(S)Ljava/lang/Short;").__()
            },
            {   // new Short(s) = Short.valueof(s) (ignoring identity)
                ____.new_(SHORT)
                    .dup()
                    .getstatic(FIELD_S)
                    .invokespecial(SHORT, "<init>", "(S)V").__(),

                ____.getstatic(FIELD_S)
                    .invokestatic(SHORT, "valueOf", "(S)Ljava/lang/Short;").__()
            },
            {   // new Short(v.f) = Short.valueof(v.f) (ignoring identity)
                ____.new_(SHORT)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_S)
                    .invokespecial(SHORT, "<init>", "(S)V").__(),

                ____.aload(A)
                    .getfield(FIELD_S)
                    .invokestatic(SHORT, "valueOf", "(S)Ljava/lang/Short;").__()
            },
            {   // Short.valueOf(...).shortValue() = nothing
                ____.invokestatic(SHORT, "valueOf", "(S)Ljava/lang/Short;")
                    .invokevirtual(SHORT_VALUE).__(),
            },
            {   // new Integer(I) = Integer.valueof(I) (ignoring identity)
                ____.new_(INTEGER)
                    .dup()
                    .iconst(A)
                    .invokespecial(INTEGER, "<init>", "(I)V").__(),

                ____.iconst(A)
                    .invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;").__()
            },
            {   // new Integer(I) = Integer.valueof(I) (ignoring identity)
                ____.new_(INTEGER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(INTEGER, "<init>", "(I)V").__(),

                ____.ldc_(A)
                    .invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;").__()
            },
            {   // new Integer(v) = Integer.valueof(v) (ignoring identity)
                ____.new_(INTEGER)
                    .dup()
                    .iload(A)
                    .invokespecial(INTEGER, "<init>", "(I)V").__(),

                ____.iload(A)
                    .invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;").__()
            },
            {   // new Integer(c) = Integer.valueof(c) (ignoring identity)
                ____.new_(INTEGER)
                    .dup()
                    .getstatic(FIELD_I)
                    .invokespecial(INTEGER, "<init>", "(I)V").__(),

                ____.getstatic(FIELD_I)
                    .invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;").__()
            },
            {   // new Integer(v.f) = Integer.valueof(v.f) (ignoring identity)
                ____.new_(INTEGER)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_I)
                    .invokespecial(INTEGER, "<init>", "(I)V").__(),

                ____.aload(A)
                    .getfield(FIELD_I)
                    .invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;").__()
            },
            {   // Integer.valueOf(...).intValue() = nothing
                ____.invokestatic(INTEGER, "valueOf", "(I)Ljava/lang/Integer;")
                    .invokevirtual(INT_VALUE).__(),
            },
            {   // new Float(F) = Float.valueof(F) (ignoring identity)
                ____.new_(FLOAT)
                    .dup()
                    .fconst(A)
                    .invokespecial(FLOAT, "<init>", "(F)V").__(),

                ____.fconst(A)
                    .invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;").__()
            },
            {   // new Float(F) = Float.valueof(F) (ignoring identity)
                ____.new_(FLOAT)
                    .dup()
                    .ldc_(A)
                    .invokespecial(FLOAT, "<init>", "(F)V").__(),

                ____.ldc_(A)
                    .invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;").__()
            },
            {   // new Float(v) = Float.valueof(v) (ignoring identity)
                ____.new_(FLOAT)
                    .dup()
                    .fload(A)
                    .invokespecial(FLOAT, "<init>", "(F)V").__(),

                ____.fload(A)
                    .invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;").__()
            },
            {   // new Float(s) = Float.valueof(s) (ignoring identity)
                ____.new_(FLOAT)
                    .dup()
                    .getstatic(FIELD_F)
                    .invokespecial(FLOAT, "<init>", "(F)V").__(),

                ____.getstatic(FIELD_F)
                    .invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;").__()
            },
            {   // new Float(v.f) = Float.valueof(v.f) (ignoring identity)
                ____.new_(FLOAT)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_F)
                    .invokespecial(FLOAT, "<init>", "(F)V").__(),

                ____.aload(A)
                    .getfield(FIELD_F)
                    .invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;").__()
            },
            {   // Float.valueOf(...).floatValue() = nothing
                ____.invokestatic(FLOAT, "valueOf", "(F)Ljava/lang/Float;")
                    .invokevirtual(FLOAT_VALUE).__(),
            },
            {   // new Long(J) = Long.valueof(J) (ignoring identity)
                ____.new_(LONG)
                    .dup()
                    .lconst(A)
                    .invokespecial(LONG, "<init>", "(J)V").__(),

                ____.lconst(A)
                    .invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;").__()
            },
            {   // new Long(J) = Long.valueof(J) (ignoring identity)
                ____.new_(LONG)
                    .dup()
                    .ldc2_w(A)
                    .invokespecial(LONG, "<init>", "(J)V").__(),

                ____.ldc2_w(A)
                    .invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;").__()
            },
            {   // new Long(v) = Long.valueof(v) (ignoring identity)
                ____.new_(LONG)
                    .dup()
                    .iload(A)
                    .invokespecial(LONG, "<init>", "(J)V").__(),

                ____.iload(A)
                    .invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;").__()
            },
            {   // new Long(s) = Long.valueof(s) (ignoring identity)
                ____.new_(LONG)
                    .dup()
                    .getstatic(FIELD_J)
                    .invokespecial(LONG, "<init>", "(J)V").__(),

                ____.getstatic(FIELD_J)
                    .invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;").__()
            },
            {   // new Long(v.f) = Long.valueof(v.f) (ignoring identity)
                ____.new_(LONG)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_J)
                    .invokespecial(LONG, "<init>", "(J)V").__(),

                ____.aload(A)
                    .getfield(FIELD_J)
                    .invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;").__()
            },
            {   // Long.valueOf(...).longValue() = nothing
                ____.invokestatic(LONG, "valueOf", "(J)Ljava/lang/Long;")
                    .invokevirtual(LONG_VALUE).__(),
            },
            {   // new Double(D) = Double.valueof(D) (ignoring identity)
                ____.new_(DOUBLE)
                    .dup()
                    .dconst(A)
                    .invokespecial(DOUBLE, "<init>", "(D)V").__(),

                ____.dconst(A)
                    .invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;").__()
            },
            {   // new Double(D) = Double.valueof(D) (ignoring identity)
                ____.new_(DOUBLE)
                    .dup()
                    .ldc2_w(A)
                    .invokespecial(DOUBLE, "<init>", "(D)V").__(),

                ____.ldc2_w(A)
                    .invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;").__()
            },
            {   // new Double(v) = Double.valueof(v) (ignoring identity)
                ____.new_(DOUBLE)
                    .dup()
                    .dload(A)
                    .invokespecial(DOUBLE, "<init>", "(D)V").__(),

                ____.dload(A)
                    .invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;").__()
            },
            {   // new Double(s) = Double.valueof(s) (ignoring identity)
                ____.new_(DOUBLE)
                    .dup()
                    .getstatic(FIELD_D)
                    .invokespecial(DOUBLE, "<init>", "(D)V").__(),

                ____.getstatic(FIELD_D)
                    .invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;").__()
            },
            {   // new Double(v.f) = Double.valueof(v.f) (ignoring identity)
                ____.new_(DOUBLE)
                    .dup()
                    .aload(A)
                    .getfield(FIELD_D)
                    .invokespecial(DOUBLE, "<init>", "(D)V").__(),

                ____.aload(A)
                    .getfield(FIELD_D)
                    .invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;").__()
            },
            {   // Double.valueOf(...).doubleValue() = nothing
                ____.invokestatic(DOUBLE, "valueOf", "(D)Ljava/lang/Double;")
                    .invokevirtual(DOUBLE_VALUE).__(),
            },
        };

        STRING_SEQUENCES = new Instruction[][][]
        {
            {   // "...".equals("...") = true
                ____.ldc_(A)
                    .ldc_(A)
                    .invokevirtual(STRING, "equals", "(Ljava/lang/Object;)Z").__(),

                ____.iconst_1().__()
            },
            {   // "...".length() = ...
                ____.ldc_(A)
                    .invokevirtual(STRING, "length", "()I").__(),

                ____.sipush(STRING_A_LENGTH).__()
            },
            {   // String.valueOf(Z) = "....
                ____.iconst(A)
                    .invokestatic(STRING, "valueOf", "(Z)Ljava/lang/String;").__(),

                ____.ldc_(BOOLEAN_A_STRING).__()
            },
            {   // String.valueOf(C) = "...."
                ____.iconst(A)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;").__(),

                ____.ldc_(CHAR_A_STRING).__()
            },
            {   // String.valueOf(Cc) = "...."
                ____.ldc_(A)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;").__(),

                ____.ldc_(CHAR_A_STRING).__()
            },
            {   // String.valueOf(I) = "...."
                ____.iconst(A)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;").__(),

                ____.ldc_(INT_A_STRING).__()
            },
            {   // String.valueOf(Ic) = "...."
                ____.ldc_(A)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;").__(),

                ____.ldc_(INT_A_STRING).__()
            },
            {   // String.valueOf(J) = "...."
                ____.lconst(A)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;").__(),

                ____.ldc_(LONG_A_STRING).__()
            },
            {   // String.valueOf(Jc) = "...."
                ____.ldc2_w(A)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;").__(),

                ____.ldc_(LONG_A_STRING).__()
            },
            {   // String.valueOf(F) = "...."
                ____.fconst(A)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;").__(),

                ____.ldc_(FLOAT_A_STRING).__()
            },
            {   // String.valueOf(Fc) = "...."
                ____.ldc_(A)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;").__(),

                ____.ldc_(FLOAT_A_STRING).__()
            },
            {   // String.valueOf(D) = "...."
                ____.dconst(A)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;").__(),

                ____.ldc_(DOUBLE_A_STRING).__()
            },
            {   // String.valueOf(Dc) = "...."
                ____.ldc2_w(A)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;").__(),

                ____.ldc_(DOUBLE_A_STRING).__()
            },
            {   // "...".concat("...") = "......"
                ____.ldc_(A)
                    .ldc_(B)
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__(),

                ____.ldc_(STRING_A_STRING | STRING_B_STRING).__(),
            },

            {   // new StringBuffer("...").toString() = "..." (ignoring identity)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A).__()
            },
            {   // new StringBuffer(string).toString() = string (ignoring identity and discarding any NullPointerException)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .aload(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A).__()
            },
            {   // new StringBuffer("...").length() = length
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(STRING_BUFFER, "length", "()I").__(),

                ____.sipush(STRING_A_LENGTH).__()
            },
            {   // new StringBuffer() (without dup) = nothing
                ____.new_(STRING_BUFFER)
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "()V").__(),
            },
            {   // new StringBuffer("...") (without dup) = nothing
                ____.new_(STRING_BUFFER)
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__(),
            },
            {   // new StringBuffer()/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .pop().__(),
            },
            {   // new StringBuffer("...")/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(z)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(c)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(i)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(l)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .lload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(f)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .fload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(d)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .dload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // new StringBuffer("...").append(s)/pop = nothing
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .pop().__(),
            },
            {   // StringBuffer#toString()/pop = pop
                ____.invokevirtual(TO_STRING)
                    .pop().__(),

                ____.pop().__()
            },
            {   // StringBuffer#append("") = nothing
                ____.ldc("")
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__(),
            },
            {   // new StringBuffer().append(Z) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(BOOLEAN_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(C) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(CHAR_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(Cc) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(CHAR_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(I) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(INT_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(Ic) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(INT_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(J) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .lconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(LONG_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(Jc) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc2_w(A)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(LONG_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(F) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .fconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(FLOAT_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(Fc) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(FLOAT_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(D) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .dconst(A)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(DOUBLE_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append(Dc) = new StringBuffer("....")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc2_w(A)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(DOUBLE_A_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer().append("...") = new StringBuffer("...")
                ____.invokespecial(STRING_BUFFER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Z) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | BOOLEAN_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(C) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Cc) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(I) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Ic) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(J) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .lconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Jc) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(F) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .fconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Fc) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(D) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .dconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(Dc) = new StringBuffer("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append("...") = new StringBuffer("......")
                ____.ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | STRING_B_STRING)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuffer("...").append(z).toString() = "...".concat(String.valueOf(z))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(Z)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(c).toString() = "...".concat(String.valueOf(c))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(i).toString() = "...".concat(String.valueOf(i))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(l).toString() = "...".concat(String.valueOf(l))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .lload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .lload(B)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(f).toString() = "...".concat(String.valueOf(f))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .fload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .fload(B)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(d).toString() = "...".concat(String.valueOf(d))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .dload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .dload(B)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(string).toString() = "...".concat(String.valueOf(string))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .aload(B)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer("...").append(object).toString() = "...".concat(String.valueOf(object))
                ____.new_(STRING_BUFFER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUFFER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/Object;)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .aload(B)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // StringBuffer#append("...").append(Z) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | BOOLEAN_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(C) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(Cc) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(I) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .iconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(Ic) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(J) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .lconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(Jc) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(F) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .fconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(Fc) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(D) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .dconst(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append(Dc) = StringBuffer#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // StringBuffer#append("...").append("...") = StringBuffer#append("......")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__(),

                ____.ldc_(STRING_A_STRING | STRING_B_STRING)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;").__()
            },
            {   // new StringBuffer().append(z).toString() = String.valueOf(z)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Z)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(Z)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(c).toString() = String.valueOf(c)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(C)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(i).toString() = String.valueOf(i)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(I)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(j).toString() = String.valueOf(j)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .lload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(J)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.lload(A)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(f).toString() = String.valueOf(f)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .fload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(F)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.fload(A)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(d).toString() = String.valueOf(d)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .dload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(D)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.dload(A)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(string).toString() = String.valueOf(string)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .aload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;").__()
            },
            {   // new StringBuffer().append(object).toString() = String.valueOf(object)
                ____.new_(STRING_BUFFER)
                    .dup()
                    .invokespecial(STRING_BUFFER, "<init>", "()V")
                    .aload(A)
                    .invokevirtual(STRING_BUFFER, "append", "(Ljava/lang/Object;)Ljava/lang/StringBuffer;")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;").__()
            },

            {   // new StringBuilder("...").toString() = "..." (ignoring identity)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A).__()
            },
            {   // new StringBuilder(string).toString() = string (ignoring identity and discarding any NullPointerException)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .aload(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A).__()
            },
            {   // new StringBuilder("...").length() = length
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .invokevirtual(STRING_BUILDER, "length", "()I").__(),

                ____.sipush(STRING_A_LENGTH).__()
            },
            {   // new StringBuilder() (without dup) = nothing
                ____.new_(STRING_BUILDER)
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "()V").__(),
            },
            {   // new StringBuilder("...") (without dup) = nothing
                ____.new_(STRING_BUILDER)
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__(),
            },
            {   // new StringBuilder()/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .pop().__(),
            },
            {   // new StringBuilder("...")/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(z)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(c)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(i)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(l)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .lload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(f)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .fload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(d)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .dload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // new StringBuilder("...").append(s)/pop = nothing
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .pop().__(),
            },
            {   // StringBuilder#toString()/pop = pop
                ____.invokevirtual(TO_STRING)
                    .pop().__(),

                ____.pop().__()
            },
            {   // StringBuilder#append("") = nothing
                ____.ldc("")
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__(),
            },
            {   // new StringBuilder().append(Z) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(BOOLEAN_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(C) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(CHAR_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(Cc) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(CHAR_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(I) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(INT_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(Ic) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(INT_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(J) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .lconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(LONG_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(Jc) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc2_w(A)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(LONG_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(F) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .fconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(FLOAT_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(Fc) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(FLOAT_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(D) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .dconst(A)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(DOUBLE_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append(Dc) = new StringBuilder("....")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc2_w(A)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(DOUBLE_A_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder().append("...") = new StringBuilder("...")
                ____.invokespecial(STRING_BUILDER, "<init>", "()V")
                    .ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Z) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | BOOLEAN_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(C) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Cc) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(I) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Ic) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(J) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .lconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Jc) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(F) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .fconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Fc) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(D) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .dconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(Dc) = new StringBuilder("....")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append("...") = new StringBuilder("......")
                ____.ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | STRING_B_STRING)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V").__()
            },
            {   // new StringBuilder("...").append(z).toString() = "...".concat(String.valueOf(z))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(Z)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(c).toString() = "...".concat(String.valueOf(c))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(i).toString() = "...".concat(String.valueOf(i))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .iload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .iload(B)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(l).toString() = "...".concat(String.valueOf(l))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .lload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .lload(B)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(f).toString() = "...".concat(String.valueOf(f))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .fload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .fload(B)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(d).toString() = "...".concat(String.valueOf(d))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .dload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .dload(B)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(string).toString() = "...".concat(String.valueOf(string))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .aload(B)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder("...").append(object).toString() = "...".concat(String.valueOf(object))
                ____.new_(STRING_BUILDER)
                    .dup()
                    .ldc_(A)
                    .invokespecial(STRING_BUILDER, "<init>", "(Ljava/lang/String;)V")
                    .aload(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/Object;)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.ldc_(A)
                    .aload(B)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;")
                    .invokevirtual(STRING, "concat", "(Ljava/lang/String;)Ljava/lang/String;").__()
            },
            {   // StringBuilder#append("...").append(Z) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | BOOLEAN_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(C) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(Cc) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | CHAR_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(I) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .iconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(Ic) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | INT_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(J) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .lconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(Jc) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | LONG_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(F) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .fconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(Fc) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | FLOAT_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(D) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .dconst(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append(Dc) = StringBuilder#append("....")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc2_w(B)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | DOUBLE_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // StringBuilder#append("...").append("...") = StringBuilder#append("......")
                ____.ldc_(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .ldc_(B)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__(),

                ____.ldc_(STRING_A_STRING | STRING_B_STRING)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;").__()
            },
            {   // new StringBuilder().append(z).toString() = String.valueOf(z)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Z)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(Z)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(c).toString() = String.valueOf(c)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(C)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(C)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(i).toString() = String.valueOf(i)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .iload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(I)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.iload(A)
                    .invokestatic(STRING, "valueOf", "(I)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(j).toString() = String.valueOf(j)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .lload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(J)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.lload(A)
                    .invokestatic(STRING, "valueOf", "(J)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(f).toString() = String.valueOf(f)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .fload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(F)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.fload(A)
                    .invokestatic(STRING, "valueOf", "(F)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(d).toString() = String.valueOf(d)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .dload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(D)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.dload(A)
                    .invokestatic(STRING, "valueOf", "(D)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(string).toString() = String.valueOf(string)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .aload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;").__()
            },
            {   // new StringBuilder().append(object).toString() = String.valueOf(object)
                ____.new_(STRING_BUILDER)
                    .dup()
                    .invokespecial(STRING_BUILDER, "<init>", "()V")
                    .aload(A)
                    .invokevirtual(STRING_BUILDER, "append", "(Ljava/lang/Object;)Ljava/lang/StringBuilder;")
                    .invokevirtual(TO_STRING).__(),

                ____.aload(A)
                    .invokestatic(STRING, "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;").__()
            },
        };

        MATH_SEQUENCES = new Instruction[][][]
        {
            {   // (float)Math.abs((double)...) = Math.abs(...)
                ____.f2d()
                    .invokestatic(MATH, "abs", "(D)D")
                    .d2f().__(),

                ____.invokestatic(MATH, "abs", "(F)F").__()
            },
            {   // (float)Math.abs(...) = Math.abs((float)...)
                ____.invokestatic(MATH, "abs", "(D)D")
                    .d2f().__(),

                ____.d2f()
                    .invokestatic(MATH, "abs", "(F)F").__()
            },
            {   // (int)Math.floor((double)...) = ...
                ____.i2d()
                    .invokestatic(MATH, "floor", "(D)D")
                    .d2i().__(),
            },
            {   // (int)Math.ceil((double)...) = ...
                ____.i2d()
                    .invokestatic(MATH, "ceil", "(D)D")
                    .d2i().__(),
            },
            {   // (float)Math.min((double)..., 0.0) = Math.min(..., 0f)
                ____.f2d()
                    .dconst_0()
                    .invokestatic(MATH, "min", "(DD)D")
                    .d2f().__(),

                ____.fconst_0()
                    .invokestatic(MATH, "min", "(FF)F").__()
            },
            {   // (float)Math.min(..., 0.0) = Math.min((float)..., 0f) (assuming in float range)
                ____.dconst_0()
                    .invokestatic(MATH, "min", "(DD)D")
                    .d2f().__(),

                ____.d2f()
                    .fconst_0()
                    .invokestatic(MATH, "min", "(FF)F").__()
            },
            {   // (float)Math.max((double)..., 0.0) = Math.max(..., 0f)
                ____.f2d()
                    .dconst_0()
                    .invokestatic(MATH, "max", "(DD)D")
                    .d2f().__(),

                ____.fconst_0()
                    .invokestatic(MATH, "max", "(FF)F").__()
            },
            {   // (float)Math.max(..., 0.0) = Math.max((float)..., 0f) (assuming in float range)
                ____.dconst_0()
                    .invokestatic(MATH, "max", "(DD)D")
                    .d2f().__(),

                ____.d2f()
                    .fconst_0()
                    .invokestatic(MATH, "max", "(FF)F").__()
            },
        };

        MATH_ANDROID_SEQUENCES = new Instruction[][][]
        {
            // As of API level 22, FloatMath has been deprecated, as the
            // equivalent methods in Math are faster on Android versions
            // with a JIT. We therefore now convert from FloatMath to Math.

            {   // FloatMath.sqrt((float)...) = (float)Math.sqrt(...)
                ____.d2f()
                    .invokestatic(FLOAT_MATH, "sqrt", "(F)F").__(),

                ____.invokestatic(MATH, "sqrt", "(D)D")
                    .d2f().__()
            },
            {   // FloatMath.sqrt(...) = (float)Math.sqrt((double)...)
                ____.invokestatic(FLOAT_MATH, "sqrt", "(F)F").__(),

                ____.f2d()
                    .invokestatic(MATH, "sqrt", "(D)D")
                    .d2f().__()
            },
            {   // FloatMath.cos((float)...) = (float)Math.cos(...)
                ____.d2f()
                    .invokestatic(FLOAT_MATH, "cos", "(F)F").__(),

                ____.invokestatic(MATH, "cos", "(D)D")
                    .d2f().__()
            },
            {   // FloatMath.cos(...) = (float)Math.cos((double)...)
                ____.invokestatic(FLOAT_MATH, "cos", "(F)F").__(),

                ____.f2d()
                    .invokestatic(MATH, "cos", "(D)D")
                    .d2f().__()
            },
            {   // FloatMath.sin((float)...) = (float)Math.sin(...)
                ____.d2f()
                    .invokestatic(FLOAT_MATH, "sin", "(F)F").__(),

                ____.invokestatic(MATH, "sin", "(D)D")
                    .d2f().__()
            },
            {   // FloatMath.sin(...) = (float)Math.sin((double)...)
                ____.invokestatic(FLOAT_MATH, "sin", "(F)F").__(),

                ____.f2d()
                    .invokestatic(MATH, "sin", "(D)D")
                    .d2f().__()
            },
            {   //  FloatMath.floor((float)...) = (float)Math.floor(...)
                ____.d2f()
                    .invokestatic(FLOAT_MATH, "floor", "(F)F").__(),

                ____.invokestatic(MATH, "floor", "(D)D")
                    .d2f().__()
            },
            {   //  FloatMath.floor(...) = (float)Math.floor((double)...)
                ____.invokestatic(FLOAT_MATH, "floor", "(F)F").__(),

                ____.f2d()
                    .invokestatic(MATH, "floor", "(D)D")
                    .d2f().__()
            },
            {   //  FloatMath.ceil((float)...) = (float)Math.ceil(...)
                ____.d2f()
                    .invokestatic(FLOAT_MATH, "ceil", "(F)F").__(),

                ____.invokestatic(MATH, "ceil", "(D)D")
                    .d2f().__()
            },
            {   //  FloatMath.ceil(...) = (float)Math.ceil((double)...)
                ____.invokestatic(FLOAT_MATH, "ceil", "(F)F").__(),

                ____.f2d()
                    .invokestatic(MATH, "ceil", "(D)D")
                    .d2f().__()
            },
        };

        CONSTANTS = ____.constants();
    }


    /**
     * Prints out the instruction sequences.
     */
    public static void main(String[] args)
    {
        InstructionSequenceConstants instructionSequenceConstants =
            new InstructionSequenceConstants(new ClassPool(),
                                             new ClassPool());

        Instruction[][][][] sets = new Instruction[][][][]
        {
            instructionSequenceConstants.VARIABLE_SEQUENCES,
            instructionSequenceConstants.ARITHMETIC_SEQUENCES,
            instructionSequenceConstants.FIELD_SEQUENCES,
            instructionSequenceConstants.CAST_SEQUENCES,
            instructionSequenceConstants.BRANCH_SEQUENCES,
            instructionSequenceConstants.STRING_SEQUENCES,
            instructionSequenceConstants.OBJECT_SEQUENCES,
            instructionSequenceConstants.MATH_SEQUENCES,
            instructionSequenceConstants.MATH_ANDROID_SEQUENCES,
        };

        ProgramClass clazz = new ProgramClass();
        clazz.constantPool = instructionSequenceConstants.CONSTANTS;

        for (int setIndex = 0; setIndex < sets.length; setIndex++)
        {
            Instruction[][][] sequencePairs = sets[setIndex];

            for (int sequencePairIndex = 0; sequencePairIndex < sequencePairs.length; sequencePairIndex++)
            {
                Instruction[][] sequencePair = sequencePairs[sequencePairIndex];

                // Print out the pattern instructions.
                Instruction[] sequence = sequencePair[0];
                for (int index = 0; index < sequence.length; index++)
                {
                    Instruction instruction = sequence[index];
                    try
                    {
                        instruction.accept(clazz, null, null, index, new ClassPrinter());
                    }
                    catch (Exception e) {}
                }

                // Are there any replacement instructions?
                if (sequencePair.length < 2)
                {
                    System.out.println("=> delete");
                }
                else
                {
                    System.out.println("=>");

                    // Print out the replacement instructions.
                    sequence = sequencePair[1];
                    for (int index = 0; index < sequence.length; index++)
                    {
                        Instruction instruction = sequence[index];
                        try
                        {
                            instruction.accept(clazz, null, null, index, new ClassPrinter());
                        }
                        catch (Exception e) {}
                    }
                }
                System.out.println();
            }
        }
    }
}
