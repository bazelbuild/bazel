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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.Constant;
import proguard.classfile.instruction.*;
import proguard.classfile.util.*;

import static proguard.classfile.ClassConstants.*;

/**
 * This AttributeVisitor accumulates instructions and exceptions, and then
 * copies them into code attributes that it visits.
 *
 * @see CodeAttributeComposer
 *
 * @author Eric Lafortune
 */
public class CompactCodeAttributeComposer
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final ConstantPoolEditor    constantPoolEditor;
    private final CodeAttributeComposer codeAttributeComposer;


    /**
     * Creates a new CompactCodeAttributeComposer that doesn't allow external
     * branch targets or exception handlers and that automatically shrinks
     * instructions.
     */
    public CompactCodeAttributeComposer(ProgramClass targetClass)
    {
        this(targetClass, false, false, true);
    }


    /**
     * Creates a new CompactCodeAttributeComposer.
     * @param allowExternalBranchTargets     specifies whether branch targets
     *                                       can lie outside the code fragment
     *                                       of the branch instructions.
     * @param allowExternalExceptionHandlers specifies whether exception
     *                                       handlers can lie outside the code
     *                                       fragment in which exceptions are
     *                                       defined.
     * @param shrinkInstructions             specifies whether instructions
     *                                       should automatically be shrunk
     *                                       before being written.
     */
    public CompactCodeAttributeComposer(ProgramClass targetClass,
                                        boolean      allowExternalBranchTargets,
                                        boolean      allowExternalExceptionHandlers,
                                        boolean      shrinkInstructions)
    {
        constantPoolEditor =
            new ConstantPoolEditor(targetClass);

        codeAttributeComposer =
            new CodeAttributeComposer(allowExternalBranchTargets,
                                      allowExternalExceptionHandlers,
                                      shrinkInstructions);
    }


    /**
     * Starts a new code definition.
     */
    public CompactCodeAttributeComposer reset()
    {
        codeAttributeComposer.reset();

        return this;
    }


    /**
     * Starts a new code fragment. Branch instructions that are added are
     * assumed to be relative within such code fragments.
     * @param maximumCodeFragmentLength the maximum length of the code that will
     *                                  be added as part of this fragment (more
     *                                  precisely, the maximum old instruction
     *                                  offset or label that is specified, plus
     *                                  one).
     */
    public CompactCodeAttributeComposer beginCodeFragment(int maximumCodeFragmentLength)
    {
        codeAttributeComposer.beginCodeFragment(maximumCodeFragmentLength);

        return this;
    }


    /**
     * Appends the given instruction with the given old offset.
     * Branch instructions must fit, for instance by enabling automatic
     * shrinking of instructions.
     * @param oldInstructionOffset the old offset of the instruction, to which
     *                             branches and other references in the current
     *                             code fragment are pointing.
     * @param instruction          the instruction to be appended.
     */
    public CompactCodeAttributeComposer appendInstruction(int         oldInstructionOffset,
                                                          Instruction instruction)
    {
        codeAttributeComposer.appendInstruction(oldInstructionOffset, instruction);

        return this;
    }


    /**
     * Appends the given label with the given old offset.
     * @param oldInstructionOffset the old offset of the label, to which
     *                             branches and other references in the current
     *                             code fragment are pointing.
     */
    public CompactCodeAttributeComposer appendLabel(int oldInstructionOffset)
    {
        codeAttributeComposer.appendLabel(oldInstructionOffset);

        return this;
    }


    /**
     * Appends the given instruction without defined offsets.
     * @param instructions the instructions to be appended.
     */
    public CompactCodeAttributeComposer appendInstructions(Instruction[] instructions)
    {
        codeAttributeComposer.appendInstructions(instructions);

        return this;
    }


    /**
     * Appends the given instruction without a defined offset.
     * Branch instructions should have a label, to allow computing the
     * new relative offset.
     * Branch instructions must fit, for instance by enabling automatic
     * shrinking of instructions.
     * @param instruction the instruction to be appended.
     */
    public CompactCodeAttributeComposer appendInstruction(Instruction instruction)
    {
        codeAttributeComposer.appendInstruction(instruction);

        return this;
    }


    /**
     * Appends the given exception to the exception table.
     * @param exceptionInfo the exception to be appended.
     */
    public CompactCodeAttributeComposer appendException(ExceptionInfo exceptionInfo)
    {
        codeAttributeComposer.appendException(exceptionInfo);

        return this;
    }


    /**
     * Inserts the given line number at the appropriate position in the line
     * number table.
     * @param lineNumberInfo the line number to be inserted.
     * @return the index where the line number was actually inserted.
     */
    public int insertLineNumber(LineNumberInfo lineNumberInfo)
    {
        return codeAttributeComposer.insertLineNumber(lineNumberInfo);
    }


    /**
     * Inserts the given line number at the appropriate position in the line
     * number table.
     * @param minimumIndex   the minimum index where the line number may be
     *                       inserted.
     * @param lineNumberInfo the line number to be inserted.
     * @return the index where the line number was inserted.
     */
    public int insertLineNumber(int minimumIndex, LineNumberInfo lineNumberInfo)
    {
        return codeAttributeComposer.insertLineNumber(minimumIndex, lineNumberInfo);
    }


    /**
     * Appends the given line number to the line number table.
     * @param lineNumberInfo the line number to be appended.
     */
    public CompactCodeAttributeComposer appendLineNumber(LineNumberInfo lineNumberInfo)
    {
        codeAttributeComposer.appendLineNumber(lineNumberInfo);

        return this;
    }


    /**
     * Wraps up the current code fragment, continuing with the previous one on
     * the stack.
     */
    public CompactCodeAttributeComposer endCodeFragment()
    {
        codeAttributeComposer.endCodeFragment();

        return this;
    }


    // Methods corresponding to the bytecode opcodes.

    public CompactCodeAttributeComposer nop()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_NOP));
    }

    public CompactCodeAttributeComposer aconst_null()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ACONST_NULL));
    }

    public CompactCodeAttributeComposer iconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_0, constant));
    }

    public CompactCodeAttributeComposer iconst_m1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_M1));
    }

    public CompactCodeAttributeComposer iconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_0));
    }

    public CompactCodeAttributeComposer iconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_1));
    }

    public CompactCodeAttributeComposer iconst_2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_2));
    }

    public CompactCodeAttributeComposer iconst_3()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_3));
    }

    public CompactCodeAttributeComposer iconst_4()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_4));
    }

    public CompactCodeAttributeComposer iconst_5()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_5));
    }

    public CompactCodeAttributeComposer lconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_0, constant));
    }

    public CompactCodeAttributeComposer lconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_0));
    }

    public CompactCodeAttributeComposer lconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_1));
    }

    public CompactCodeAttributeComposer fconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_0, constant));
    }

    public CompactCodeAttributeComposer fconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_0));
    }

    public CompactCodeAttributeComposer fconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_1));
    }

    public CompactCodeAttributeComposer fconst_2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_2));
    }

    public CompactCodeAttributeComposer dconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_0, constant));
    }

    public CompactCodeAttributeComposer dconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_0));
    }

    public CompactCodeAttributeComposer dconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_1));
    }

    public CompactCodeAttributeComposer bipush(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BIPUSH, constant));
    }

    public CompactCodeAttributeComposer sipush(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SIPUSH, constant));
    }

    public CompactCodeAttributeComposer ldc(int value)
    {
        return ldc_(constantPoolEditor.addIntegerConstant(value));
    }

    public CompactCodeAttributeComposer ldc(float value)
    {
        return ldc_(constantPoolEditor.addFloatConstant(value));
    }

    public CompactCodeAttributeComposer ldc(String string)
    {
        return ldc(string, null, null);
    }

    public CompactCodeAttributeComposer ldc(Object primitiveArray)
    {
        return ldc_(constantPoolEditor.addPrimitiveArrayConstant(primitiveArray));
    }

    public CompactCodeAttributeComposer ldc(String string, Clazz referencedClass, Method referencedMember)
    {
        return ldc_(constantPoolEditor.addStringConstant(string, referencedClass, referencedMember));
    }

    public CompactCodeAttributeComposer ldc(String className, Clazz referencedClass)
    {
        return ldc_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer ldc_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC, constantIndex));
    }

    public CompactCodeAttributeComposer ldc_w(int value)
    {
        return ldc_w_(constantPoolEditor.addIntegerConstant(value));
    }

    public CompactCodeAttributeComposer ldc_w(float value)
    {
        return ldc_w_(constantPoolEditor.addFloatConstant(value));
    }

    public CompactCodeAttributeComposer ldc_w(String string)
    {
        return ldc_w(string, null, null);
    }

    public CompactCodeAttributeComposer ldc_w(String string, Clazz referencedClass, Method referencedMember)
    {
        return ldc_w_(constantPoolEditor.addStringConstant(string, referencedClass, referencedMember));
    }

    public CompactCodeAttributeComposer ldc_w(String className, Clazz referencedClass)
    {
        return ldc_w_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer ldc_w_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC_W, constantIndex));
    }

    public CompactCodeAttributeComposer ldc2_w(long value)
    {
        return ldc2_w(constantPoolEditor.addLongConstant(value));
    }

    public CompactCodeAttributeComposer ldc2_w(double value)
    {
        return ldc2_w(constantPoolEditor.addDoubleConstant(value));
    }

    public CompactCodeAttributeComposer ldc2_w(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC2_W, constantIndex));
    }

    public CompactCodeAttributeComposer iload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD, variableIndex));
    }

    public CompactCodeAttributeComposer lload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD, variableIndex));
    }

    public CompactCodeAttributeComposer fload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD, variableIndex));
    }

    public CompactCodeAttributeComposer dload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD, variableIndex));
    }

    public CompactCodeAttributeComposer aload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD, variableIndex));
    }

    public CompactCodeAttributeComposer iload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_0));
    }

    public CompactCodeAttributeComposer iload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_1));
    }

    public CompactCodeAttributeComposer iload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_2));
    }

    public CompactCodeAttributeComposer iload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_3));
    }

    public CompactCodeAttributeComposer lload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_0));
    }

    public CompactCodeAttributeComposer lload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_1));
    }

    public CompactCodeAttributeComposer lload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_2));
    }

    public CompactCodeAttributeComposer lload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_3));
    }

    public CompactCodeAttributeComposer fload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_0));
    }

    public CompactCodeAttributeComposer fload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_1));
    }

    public CompactCodeAttributeComposer fload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_2));
    }

    public CompactCodeAttributeComposer fload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_3));
    }

    public CompactCodeAttributeComposer dload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_0));
    }

    public CompactCodeAttributeComposer dload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_1));
    }

    public CompactCodeAttributeComposer dload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_2));
    }

    public CompactCodeAttributeComposer dload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_3));
    }

    public CompactCodeAttributeComposer aload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_0));
    }

    public CompactCodeAttributeComposer aload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_1));
    }

    public CompactCodeAttributeComposer aload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_2));
    }

    public CompactCodeAttributeComposer aload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_3));
    }

    public CompactCodeAttributeComposer iaload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IALOAD));
    }

    public CompactCodeAttributeComposer laload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LALOAD));
    }

    public CompactCodeAttributeComposer faload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FALOAD));
    }

    public CompactCodeAttributeComposer daload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DALOAD));
    }

    public CompactCodeAttributeComposer aaload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_AALOAD));
    }

    public CompactCodeAttributeComposer baload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BALOAD));
    }

    public CompactCodeAttributeComposer caload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_CALOAD));
    }

    public CompactCodeAttributeComposer saload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SALOAD));
    }

    public CompactCodeAttributeComposer istore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE, variableIndex));
    }

    public CompactCodeAttributeComposer lstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE, variableIndex));
    }

    public CompactCodeAttributeComposer fstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE, variableIndex));
    }

    public CompactCodeAttributeComposer dstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE, variableIndex));
    }

    public CompactCodeAttributeComposer astore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE, variableIndex));
    }

    public CompactCodeAttributeComposer istore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_0));
    }

    public CompactCodeAttributeComposer istore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_1));
    }

    public CompactCodeAttributeComposer istore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_2));
    }

    public CompactCodeAttributeComposer istore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_3));
    }

    public CompactCodeAttributeComposer lstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_0));
    }

    public CompactCodeAttributeComposer lstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_1));
    }

    public CompactCodeAttributeComposer lstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_2));
    }

    public CompactCodeAttributeComposer lstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_3));
    }

    public CompactCodeAttributeComposer fstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_0));
    }

    public CompactCodeAttributeComposer fstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_1));
    }

    public CompactCodeAttributeComposer fstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_2));
    }

    public CompactCodeAttributeComposer fstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_3));
    }

    public CompactCodeAttributeComposer dstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_0));
    }

    public CompactCodeAttributeComposer dstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_1));
    }

    public CompactCodeAttributeComposer dstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_2));
    }

    public CompactCodeAttributeComposer dstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_3));
    }

    public CompactCodeAttributeComposer astore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_0));
    }

    public CompactCodeAttributeComposer astore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_1));
    }

    public CompactCodeAttributeComposer astore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_2));
    }

    public CompactCodeAttributeComposer astore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_3));
    }

    public CompactCodeAttributeComposer iastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IASTORE));
    }

    public CompactCodeAttributeComposer lastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LASTORE));
    }

    public CompactCodeAttributeComposer fastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FASTORE));
    }

    public CompactCodeAttributeComposer dastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DASTORE));
    }

    public CompactCodeAttributeComposer aastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_AASTORE));
    }

    public CompactCodeAttributeComposer bastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BASTORE));
    }

    public CompactCodeAttributeComposer castore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_CASTORE));
    }

    public CompactCodeAttributeComposer sastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SASTORE));
    }

    public CompactCodeAttributeComposer pop()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_POP));
    }

    public CompactCodeAttributeComposer pop2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_POP2));
    }

    public CompactCodeAttributeComposer dup()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP));
    }

    public CompactCodeAttributeComposer dup_x1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP_X1));
    }

    public CompactCodeAttributeComposer dup_x2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP_X2));
    }

    public CompactCodeAttributeComposer dup2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2));
    }

    public CompactCodeAttributeComposer dup2_x1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2_X1));
    }

    public CompactCodeAttributeComposer dup2_x2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2_X2));
    }

    public CompactCodeAttributeComposer swap()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SWAP));
    }

    public CompactCodeAttributeComposer iadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IADD));
    }

    public CompactCodeAttributeComposer ladd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LADD));
    }

    public CompactCodeAttributeComposer fadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FADD));
    }

    public CompactCodeAttributeComposer dadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DADD));
    }

    public CompactCodeAttributeComposer isub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISUB));
    }

    public CompactCodeAttributeComposer lsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSUB));
    }

    public CompactCodeAttributeComposer fsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FSUB));
    }

    public CompactCodeAttributeComposer dsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DSUB));
    }

    public CompactCodeAttributeComposer imul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IMUL));
    }

    public CompactCodeAttributeComposer lmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LMUL));
    }

    public CompactCodeAttributeComposer fmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FMUL));
    }

    public CompactCodeAttributeComposer dmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DMUL));
    }

    public CompactCodeAttributeComposer idiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IDIV));
    }

    public CompactCodeAttributeComposer ldiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LDIV));
    }

    public CompactCodeAttributeComposer fdiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FDIV));
    }

    public CompactCodeAttributeComposer ddiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DDIV));
    }

    public CompactCodeAttributeComposer irem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IREM));
    }

    public CompactCodeAttributeComposer lrem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LREM));
    }

    public CompactCodeAttributeComposer frem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FREM));
    }

    public CompactCodeAttributeComposer drem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DREM));
    }

    public CompactCodeAttributeComposer ineg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_INEG));
    }

    public CompactCodeAttributeComposer lneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LNEG));
    }

    public CompactCodeAttributeComposer fneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FNEG));
    }

    public CompactCodeAttributeComposer dneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DNEG));
    }

    public CompactCodeAttributeComposer ishl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISHL));
    }

    public CompactCodeAttributeComposer lshl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSHL));
    }

    public CompactCodeAttributeComposer ishr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISHR));
    }

    public CompactCodeAttributeComposer lshr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSHR));
    }

    public CompactCodeAttributeComposer iushr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IUSHR));
    }

    public CompactCodeAttributeComposer lushr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LUSHR));
    }

    public CompactCodeAttributeComposer iand()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IAND));
    }

    public CompactCodeAttributeComposer land()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LAND));
    }

    public CompactCodeAttributeComposer ior()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IOR));
    }

    public CompactCodeAttributeComposer lor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LOR));
    }

    public CompactCodeAttributeComposer ixor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IXOR));
    }

    public CompactCodeAttributeComposer lxor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LXOR));
    }

    public CompactCodeAttributeComposer iinc(int variableIndex,
                                               int constant)
    {
        return add(new VariableInstruction(InstructionConstants.OP_IINC, variableIndex, constant));
    }

    public CompactCodeAttributeComposer i2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2L));
    }

    public CompactCodeAttributeComposer i2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2F));
    }

    public CompactCodeAttributeComposer i2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2D));
    }

    public CompactCodeAttributeComposer l2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2I));
    }

    public CompactCodeAttributeComposer l2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2F));
    }

    public CompactCodeAttributeComposer l2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2D));
    }

    public CompactCodeAttributeComposer f2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2I));
    }

    public CompactCodeAttributeComposer f2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2L));
    }

    public CompactCodeAttributeComposer f2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2D));
    }

    public CompactCodeAttributeComposer d2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2I));
    }

    public CompactCodeAttributeComposer d2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2L));
    }

    public CompactCodeAttributeComposer d2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2F));
    }

    public CompactCodeAttributeComposer i2b()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2B));
    }

    public CompactCodeAttributeComposer i2c()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2C));
    }

    public CompactCodeAttributeComposer i2s()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2S));
    }

    public CompactCodeAttributeComposer lcmp()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCMP));
    }

    public CompactCodeAttributeComposer fcmpl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCMPL));
    }

    public CompactCodeAttributeComposer fcmpg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCMPG));
    }

    public CompactCodeAttributeComposer dcmpl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCMPL));
    }

    public CompactCodeAttributeComposer dcmpg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCMPG));
    }

    public CompactCodeAttributeComposer ifeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFEQ, branchOffset));
    }

    public CompactCodeAttributeComposer ifne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNE, branchOffset));
    }

    public CompactCodeAttributeComposer iflt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFLT, branchOffset));
    }

    public CompactCodeAttributeComposer ifge(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFGE, branchOffset));
    }

    public CompactCodeAttributeComposer ifgt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFGT, branchOffset));
    }

    public CompactCodeAttributeComposer ifle(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFLE, branchOffset));
    }

    public CompactCodeAttributeComposer ificmpeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPEQ, branchOffset));
    }

    public CompactCodeAttributeComposer ificmpne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPNE, branchOffset));
    }

    public CompactCodeAttributeComposer ificmplt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPLT, branchOffset));
    }

    public CompactCodeAttributeComposer ificmpge(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPGE, branchOffset));
    }

    public CompactCodeAttributeComposer ificmpgt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPGT, branchOffset));
    }

    public CompactCodeAttributeComposer ificmple(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPLE, branchOffset));
    }

    public CompactCodeAttributeComposer ifacmpeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFACMPEQ, branchOffset));
    }

    public CompactCodeAttributeComposer ifacmpne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFACMPNE, branchOffset));
    }

    public CompactCodeAttributeComposer goto_(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_GOTO, branchOffset));
    }

    public CompactCodeAttributeComposer jsr(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_JSR, branchOffset));
    }

    public CompactCodeAttributeComposer ret(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_RET, variableIndex));
    }

    public CompactCodeAttributeComposer tableswitch(int   defaultOffset,
                                                      int   lowCase,
                                                      int   highCase,
                                                      int[] jumpOffsets)
    {
        return add(new TableSwitchInstruction(InstructionConstants.OP_TABLESWITCH,
                                              defaultOffset,
                                              lowCase,
                                              highCase,
                                              jumpOffsets));
    }

    public CompactCodeAttributeComposer lookupswitch(int  defaultOffset,
                                                       int[] cases,
                                                       int[] jumpOffsets)
    {
        return add(new LookUpSwitchInstruction(InstructionConstants.OP_LOOKUPSWITCH,
                                               defaultOffset,
                                               cases,
                                               jumpOffsets));
    }

    public CompactCodeAttributeComposer ireturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IRETURN));
    }

    public CompactCodeAttributeComposer lreturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LRETURN));
    }

    public CompactCodeAttributeComposer freturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FRETURN));
    }

    public CompactCodeAttributeComposer dreturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DRETURN));
    }

    public CompactCodeAttributeComposer areturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ARETURN));
    }

    public CompactCodeAttributeComposer return_()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_RETURN));
    }

    public CompactCodeAttributeComposer getstatic(Clazz  referencedClass,
                                                  Member referencedMember)
    {
        return getstatic(referencedClass.getName(),
                         referencedMember.getName(referencedClass),
                         referencedMember.getDescriptor(referencedClass),
                         referencedClass,
                         referencedMember);
    }

    public CompactCodeAttributeComposer getstatic(String className,
                                                  String name,
                                                  String descriptor)
    {
        return getstatic(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer getstatic(String className,
                                                  String name,
                                                  String descriptor,
                                                  Clazz  referencedClass,
                                                  Member referencedMember)
    {
        return getstatic(constantPoolEditor.addFieldrefConstant(className,
                                                                name,
                                                                descriptor,
                                                                referencedClass,
                                                                referencedMember));
    }

    public CompactCodeAttributeComposer getstatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_GETSTATIC, constantIndex));
    }

    public CompactCodeAttributeComposer putstatic(Clazz  referencedClass,
                                                  Member referencedMember)
    {
        return putstatic(referencedClass.getName(),
                         referencedMember.getName(referencedClass),
                         referencedMember.getDescriptor(referencedClass),
                         referencedClass,
                         referencedMember);
    }

    public CompactCodeAttributeComposer putstatic(String className,
                                                  String name,
                                                  String descriptor)
    {
        return putstatic(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer putstatic(String className,
                                                  String name,
                                                  String descriptor,
                                                  Clazz  referencedClass,
                                                  Member referencedMember)
    {
        return putstatic(constantPoolEditor.addFieldrefConstant(className,
                                                                name,
                                                                descriptor,
                                                                referencedClass,
                                                                referencedMember));
    }

    public CompactCodeAttributeComposer putstatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_PUTSTATIC, constantIndex));
    }

    public CompactCodeAttributeComposer getfield(String className,
                                                 String name,
                                                 String descriptor)
    {
        return getfield(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer getfield(String className,
                                                   String name,
                                                   String descriptor,
                                                   Clazz  referencedClass,
                                                   Member referencedMember)
    {
        return getfield(constantPoolEditor.addFieldrefConstant(className,
                                                               name,
                                                               descriptor,
                                                               referencedClass,
                                                               referencedMember));
    }

    public CompactCodeAttributeComposer getfield(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_GETFIELD, constantIndex));
    }

    public CompactCodeAttributeComposer putfield(String className,
                                                 String name,
                                                 String descriptor)
    {
        return putfield(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer putfield(String className,
                                                 String name,
                                                 String descriptor,
                                                 Clazz  referencedClass,
                                                 Member referencedMember)
    {
        return putfield(constantPoolEditor.addFieldrefConstant(className,
                                                               name,
                                                               descriptor,
                                                               referencedClass,
                                                               referencedMember));
    }

    public CompactCodeAttributeComposer putfield(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_PUTFIELD, constantIndex));
    }

    public CompactCodeAttributeComposer invokevirtual(String className,
                                                      String name,
                                                      String descriptor)
    {
        return invokevirtual(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer invokevirtual(Clazz  referencedClass,
                                                      Member referencedMember)
    {
        return invokevirtual(referencedClass.getName(),
                             referencedMember.getName(referencedClass),
                             referencedMember.getDescriptor(referencedClass),
                             referencedClass,
                             referencedMember);
    }

    public CompactCodeAttributeComposer invokevirtual(String className,
                                                      String name,
                                                      String descriptor,
                                                      Clazz  referencedClass,
                                                      Member referencedMember)
    {
        return invokevirtual(constantPoolEditor.addMethodrefConstant(className,
                                                                     name,
                                                                     descriptor,
                                                                     referencedClass,
                                                                     referencedMember));
    }

    public CompactCodeAttributeComposer invokevirtual(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, constantIndex));
    }

    public CompactCodeAttributeComposer invokespecial(String className,
                                                      String name,
                                                      String descriptor)
    {
        return invokespecial(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer invokespecial(String className,
                                                      String name,
                                                      String descriptor,
                                                      Clazz  referencedClass,
                                                      Member referencedMember)
    {
        return invokespecial(constantPoolEditor.addMethodrefConstant(className,
                                                                     name,
                                                                     descriptor,
                                                                     referencedClass,
                                                                     referencedMember));
    }

    public CompactCodeAttributeComposer invokespecial(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, constantIndex));
    }

    public CompactCodeAttributeComposer invokestatic(String className,
                                                     String name,
                                                     String descriptor)
    {
        return invokestatic(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer invokestatic(Clazz  referencedClass,
                                                     Member referencedMember)
    {
        return invokestatic(referencedClass.getName(),
                            referencedMember.getName(referencedClass),
                            referencedMember.getDescriptor(referencedClass),
                            referencedClass,
                            referencedMember);
    }

    public CompactCodeAttributeComposer invokestatic(String className,
                                                     String name,
                                                     String descriptor,
                                                     Clazz  referencedClass,
                                                     Member referencedMember)
    {
        return invokestatic(constantPoolEditor.addMethodrefConstant(className,
                                                                    name,
                                                                    descriptor,
                                                                    referencedClass,
                                                                    referencedMember));
    }

    public CompactCodeAttributeComposer invokestaticinterface(String className,
                                                              String name,
                                                              String descriptor)
    {
        return invokestaticinterface(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer invokestaticinterface(Clazz  referencedClass,
                                                              Member referencedMember)
    {
        return invokestaticinterface(referencedClass.getName(),
                                     referencedMember.getName(referencedClass),
                                     referencedMember.getDescriptor(referencedClass),
                                     referencedClass,
                                     referencedMember);
    }

    public CompactCodeAttributeComposer invokestaticinterface(String className,
                                                              String name,
                                                              String descriptor,
                                                              Clazz  referencedClass,
                                                              Member referencedMember)
    {
        return invokestatic(constantPoolEditor.addInterfaceMethodrefConstant(className,
                                                                             name,
                                                                             descriptor,
                                                                             referencedClass,
                                                                             referencedMember));
    }

    public CompactCodeAttributeComposer invokestatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, constantIndex));
    }

    public CompactCodeAttributeComposer invokeinterface(String className,
                                                          String name,
                                                          String descriptor)
    {
        return invokeinterface(className, name, descriptor, null, null);
    }

    public CompactCodeAttributeComposer invokeinterface(String className,
                                                        String name,
                                                        String descriptor,
                                                        Clazz  referencedClass,
                                                        Member referencedMember)
    {
        int invokeinterfaceConstant =
            (ClassUtil.internalMethodParameterSize(descriptor, false)) << 8;

        return invokeinterface(constantPoolEditor.addInterfaceMethodrefConstant(className,
                                                                                name,
                                                                                descriptor,
                                                                                referencedClass,
                                                                                referencedMember),
                               invokeinterfaceConstant);
    }

    public CompactCodeAttributeComposer invokeinterface(int constantIndex,
                                                        int constant)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEINTERFACE, constantIndex, constant));
    }

    public CompactCodeAttributeComposer invokedynamic(int     bootStrapMethodIndex,
                                                      String  name,
                                                      String  descriptor,
                                                      Clazz[] referencedClasses)
    {
        return invokedynamic(constantPoolEditor.addInvokeDynamicConstant(bootStrapMethodIndex,
                                                                         name,
                                                                         descriptor,
                                                                         referencedClasses));
    }

    public CompactCodeAttributeComposer invokedynamic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEDYNAMIC, constantIndex));
    }

    public CompactCodeAttributeComposer new_(String className)
    {
        return new_(className, null);
    }

    public CompactCodeAttributeComposer new_(String className, Clazz referencedClass)
    {
        return new_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer new_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_NEW, constantIndex));
    }

    public CompactCodeAttributeComposer newarray(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_NEWARRAY, constant));
    }

    public CompactCodeAttributeComposer anewarray(String className, Clazz referencedClass)
    {
        return anewarray(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer anewarray(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, constantIndex));
    }

    public CompactCodeAttributeComposer arraylength()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ARRAYLENGTH));
    }

    public CompactCodeAttributeComposer athrow()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ATHROW));
    }

    public CompactCodeAttributeComposer checkcast(String className)
    {
        return checkcast(className, null);
    }

    public CompactCodeAttributeComposer checkcast(String className, Clazz referencedClass)
    {
        return checkcast(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer checkcast(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_CHECKCAST, constantIndex));
    }

    public CompactCodeAttributeComposer instanceof_(String className, Clazz referencedClass)
    {
        return instanceof_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer instanceof_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INSTANCEOF, constantIndex));
    }

    public CompactCodeAttributeComposer monitorenter()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_MONITORENTER));
    }

    public CompactCodeAttributeComposer monitorexit()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_MONITOREXIT));
    }

    public CompactCodeAttributeComposer wide()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_WIDE));
    }

    public CompactCodeAttributeComposer multianewarray(String className, Clazz referencedClass)
    {
        return multianewarray(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public CompactCodeAttributeComposer multianewarray(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_MULTIANEWARRAY, constantIndex));
    }

    public CompactCodeAttributeComposer ifnull(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNULL, branchOffset));
    }

    public CompactCodeAttributeComposer ifnonnull(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNONNULL, branchOffset));
    }

    public CompactCodeAttributeComposer goto_w(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_GOTO_W, branchOffset));
    }

    public CompactCodeAttributeComposer jsr_w(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_JSR_W, branchOffset));
    }


    // Additional convenience methods.

    /**
     * Pushes the given primitive value on the stack.
     *
     * Operand stack:
     * ... -> ..., value
     *
     * @param primitive the primitive value to be pushed - should never be null.
     * @param internalType      the internal type of the primitive ('Z','B','I',...)
     */
    public CompactCodeAttributeComposer pushPrimitive(Object primitive,
                                                      char   internalType)
    {
        switch (internalType)
        {
            case TYPE_BOOLEAN: return ((Boolean)primitive).booleanValue() ? iconst_1() : iconst_0();
            case TYPE_BYTE:    return bipush((Byte)primitive);
            case TYPE_CHAR:    return ldc(((Character)primitive).charValue());
            case TYPE_SHORT:   return sipush((Short)primitive);
            case TYPE_INT:     return ldc(((Integer)primitive).intValue());
            case TYPE_LONG:    return ldc2_w((Long)primitive);
            case TYPE_FLOAT:   return ldc(((Float)primitive).floatValue());
            case TYPE_DOUBLE:  return ldc2_w((Double)primitive);
            default: throw new IllegalArgumentException(primitive.toString());
        }
    }


    /**
     * Pushes the given primitive int on the stack in the most efficient way
     * (as an iconst, bipush, sipush, or ldc instruction).
     *
     * @param value the int value to be pushed.
     */
    public CompactCodeAttributeComposer pushInt(int value)
    {
        return
            value >= -1 &&
            value <= 5            ? iconst(value) :
            value == (byte)value  ? bipush(value) :
            value == (short)value ? sipush(value) :
                                    ldc(value);
    }


    /**
     * Pushes the given primitive float on the stack in the most efficient way
     * (as an fconst or ldc instruction).
     *
     * @param value the int value to be pushed.
     */
    public CompactCodeAttributeComposer pushFloat(float value)
    {
        return
            value == 0f ||
            value == 1f ? fconst((int)value) :
                          ldc(value);
    }


    /**
     * Pushes the given primitive long on the stack in the most efficient way
     * (as an lconst or ldc instruction).
     *
     * @param value the int value to be pushed.
     */
    public CompactCodeAttributeComposer pushLong(long value)
    {
        return
            value == 0L ||
            value == 1L ? lconst((int)value) :
                          ldc2_w(value);
    }


    /**
     * Pushes the given primitive double on the stack in the most efficient way
     * (as a dconst or ldc instruction).
     *
     * @param value the int value to be pushed.
     */
    public CompactCodeAttributeComposer pushDouble(double value)
    {
        return
            value == 0. ||
            value == 1. ? dconst((int)value) :
                          ldc2_w(value);
    }


    /**
     * Pushes a new array on the stack.
     *
     * Operand stack:
     * ... -> ..., array
     *
     * @param elementTypeOrClassName the array element type (or class name in case of objects).
     * @param size                   the size of the array to be created.
     */
    public CompactCodeAttributeComposer pushNewArray(String elementTypeOrClassName,
                                                     int    size)
    {
        // Create new array.
        pushInt(size);

        return ClassUtil.isInternalPrimitiveType(elementTypeOrClassName) ?
            newarray(InstructionUtil.arrayTypeFromInternalType(elementTypeOrClassName.charAt(0))) :
            anewarray(elementTypeOrClassName, null);
    }


    /**
     * Loads the given variable onto the stack.
     *
     * Operand stack:
     * ... -> ..., value
     *
     * @param variableIndex the index of the variable to be loaded.
     * @param internalType  the type of the variable to be loaded.
     */
    public CompactCodeAttributeComposer load(int    variableIndex,
                                             String internalType)
    {
        return load(variableIndex, internalType.charAt(0));
    }


    /**
     * Loads the given variable of primitive type onto the stack.
     *
     * Operand stack:
     * ... -> ..., value
     *
     * @param variableIndex the index of the variable to be loaded.
     * @param internalType  the primitive type of the variable to be loaded.
     */
    public CompactCodeAttributeComposer load(int  variableIndex,
                                             char internalType)
    {
        switch (internalType)
        {
            case TYPE_BOOLEAN:
            case TYPE_BYTE:
            case TYPE_CHAR:
            case TYPE_SHORT:
            case TYPE_INT:    return iload(variableIndex);
            case TYPE_LONG:   return lload(variableIndex);
            case TYPE_FLOAT:  return fload(variableIndex);
            case TYPE_DOUBLE: return dload(variableIndex);
            default:          return aload(variableIndex);
        }
    }


    /**
     * Stores the value on top of the stack in the variable with given index.
     *
     * Operand stsack:
     * ..., value -> ...
     *
     * @param variableIndex the index of the variable where to store the
     *                      value.
     * @param internalType  the type of the value to be stored.
     */
    public CompactCodeAttributeComposer store(int    variableIndex,
                                              String internalType)
    {
        return store(variableIndex, internalType.charAt(0));
    }


    /**
     * Stores the primitve value on top of the stack in the variable with given
     * index.
     *
     * Operand stack:
     * ..., value -> ...
     *
     * @param variableIndex the index of the variable where to store the
     *                      value.
     * @param internalType  the primitive type of the value to be stored.
     */
    public CompactCodeAttributeComposer store(int  variableIndex,
                                              char internalType)
    {
        switch (internalType)
        {
            case TYPE_BOOLEAN:
            case TYPE_BYTE:
            case TYPE_CHAR:
            case TYPE_SHORT:
            case TYPE_INT:    return istore(variableIndex);
            case TYPE_LONG:   return lstore(variableIndex);
            case TYPE_FLOAT:  return fstore(variableIndex);
            case TYPE_DOUBLE: return dstore(variableIndex);
            default:          return astore(variableIndex);
        }
    }


    /**
     * Stores an element to an array.
     *
     * Operand stack:
     * ..., array, index, value -> ...
     *
     * @param elementType the type of the value to be stored.
     */
    public CompactCodeAttributeComposer storeToArray(String elementType)
    {
        switch (elementType.charAt(0))
        {
            case TYPE_BOOLEAN:
            case TYPE_BYTE:   return bastore();
            case TYPE_CHAR:   return castore();
            case TYPE_SHORT:  return sastore();
            case TYPE_INT:    return iastore();
            case TYPE_LONG:   return lastore();
            case TYPE_FLOAT:  return fastore();
            case TYPE_DOUBLE: return dastore();
            default:          return aastore();
        }
    }


    /**
     * Appends the proper return statement for the given internal type.
     *
     * @param internalType the return type.
     */
    public CompactCodeAttributeComposer return_(String internalType)
    {
        switch (internalType.charAt(0))
        {
            case TYPE_BOOLEAN:
            case TYPE_BYTE:
            case TYPE_CHAR:
            case TYPE_SHORT:
            case TYPE_INT:    return ireturn();
            case TYPE_LONG:   return lreturn();
            case TYPE_FLOAT:  return freturn();
            case TYPE_DOUBLE: return dreturn();
            case TYPE_VOID:   return return_();
            default:          return areturn();
        }
    }


    /**
     * Appends instructions to print out the given message and the top int on
     * the stack.
     */
    public CompactCodeAttributeComposer appendPrintIntegerInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintIntegerInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message and the top int on
     * the stack as a hexadecimal value.
     */
    public CompactCodeAttributeComposer appendPrintIntegerHexInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintIntegerHexInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message and the top long on
     * the stack.
     */
    public CompactCodeAttributeComposer appendPrintLongInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintLongInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message and the top String on
     * the stack.
     */
    public CompactCodeAttributeComposer appendPrintStringInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintStringInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message and the top Object on
     * the stack.
     */
    public CompactCodeAttributeComposer appendPrintObjectInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintObjectInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message and the stack trace
     * of the top Throwable on the stack.
     */
    public CompactCodeAttributeComposer appendPrintStackTraceInstructions(String message)
    {
        appendPrintInstructions(message);
        appendPrintStackTraceInstructions();
        return this;
    }

    /**
     * Appends instructions to print out the given message.
     */
    public CompactCodeAttributeComposer appendPrintInstructions(String message)
    {
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        ldc(message);
        invokevirtual("java/io/PrintStream", "println", "(Ljava/lang/String;)V");
        return this;
    }

    /**
     * Appends instructions to print out the top int on the stack.
     */
    public CompactCodeAttributeComposer appendPrintIntegerInstructions()
    {
        dup();
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        swap();
        invokevirtual("java/io/PrintStream", "println", "(I)V");
        return this;
    }

    /**
     * Appends instructions to print out the top integer on the stack as a
     * hexadecimal value.
     */
    public CompactCodeAttributeComposer appendPrintIntegerHexInstructions()
    {
        dup();
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        swap();
        invokestatic("java/lang/Integer", "toHexString", "(I)Ljava/lang/String;");
        invokevirtual("java/io/PrintStream", "println", "(Ljava/lang/String;)V");
        return this;
    }

    /**
     * Appends instructions to print out the top long on the stack.
     */
    public CompactCodeAttributeComposer appendPrintLongInstructions()
    {
        dup2();
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        dup_x2();
        pop();
        invokevirtual("java/io/PrintStream", "println", "(J)V");
        return this;
    }

    /**
     * Appends instructions to print out the top String on the stack.
     */
    public CompactCodeAttributeComposer appendPrintStringInstructions()
    {
        dup();
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        swap();
        invokevirtual("java/io/PrintStream", "println", "(Ljava/lang/String;)V");
        return this;
    }

    /**
     * Appends instructions to print out the top Object on the stack.
     */
    public CompactCodeAttributeComposer appendPrintObjectInstructions()
    {
        dup();
        getstatic("java/lang/System", "err", "Ljava/io/PrintStream;");
        swap();
        invokevirtual("java/io/PrintStream", "println", "(Ljava/lang/Object;)V");
        return this;
    }

    /**
     * Appends instructions to print out the stack trace of the top Throwable
     * on the stack.
     */
    public CompactCodeAttributeComposer appendPrintStackTraceInstructions()
    {
        dup();
        invokevirtual("java/lang/Throwable", "printStackTrace", "()V");
        return this;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttributeComposer.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Small utility methods.

    /**
     * Adds the given instruction, shrinking it if necessary.
     */
    private CompactCodeAttributeComposer add(Instruction instruction)
    {
        codeAttributeComposer.appendInstruction(instruction);

        return this;
    }


    public static void main(String[] args)
    {
        ProgramClass targetClass = new ProgramClass(0, 0, new Constant[32], 0, 0, 0);

        CompactCodeAttributeComposer composer = new CompactCodeAttributeComposer(targetClass);

        composer.beginCodeFragment(4);
        composer.appendInstruction(0, new SimpleInstruction(InstructionConstants.OP_ICONST_0));
        composer.appendInstruction(1, new VariableInstruction(InstructionConstants.OP_ISTORE, 0));
        composer.appendInstruction(2, new BranchInstruction(InstructionConstants.OP_GOTO, 1));

        composer.beginCodeFragment(4);
        composer.appendInstruction(0, new VariableInstruction(InstructionConstants.OP_IINC, 0, 1));
        composer.appendInstruction(1, new VariableInstruction(InstructionConstants.OP_ILOAD, 0));
        composer.appendInstruction(2, new SimpleInstruction(InstructionConstants.OP_ICONST_5));
        composer.appendInstruction(3, new BranchInstruction(InstructionConstants.OP_IFICMPLT, -3));
        composer.endCodeFragment();

        composer.appendInstruction(3, new SimpleInstruction(InstructionConstants.OP_RETURN));
        composer.endCodeFragment();
    }
}
