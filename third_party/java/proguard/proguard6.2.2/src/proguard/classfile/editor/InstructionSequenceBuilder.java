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
import proguard.classfile.constant.Constant;
import proguard.classfile.instruction.*;
import proguard.classfile.util.ClassUtil;
import proguard.optimize.peephole.InstructionSequenceReplacer;

import java.util.*;

/**
 * This utility class allows to construct sequences of instructions and
 * their constants.
 *
 * @author Eric Lafortune
 */
public class InstructionSequenceBuilder
{
    private final ConstantPoolEditor constantPoolEditor;

    private final List<Instruction> instructions = new ArrayList<Instruction>(256);


    /**
     * Creates a new InstructionSequenceBuilder.
     */
    public InstructionSequenceBuilder()
    {
        this(null, null);
    }


    /**
     * Creates a new InstructionSequenceBuilder that automatically initializes
     * class references and class member references in new constants.
     * @param programClassPool the program class pool from which new
     *                         constants can be initialized.
     * @param libraryClassPool the library class pool from which new
     *                         constants can be initialized.
     */
    public InstructionSequenceBuilder(ClassPool programClassPool,
                                      ClassPool libraryClassPool)
    {
        this(new MyDummyClass(),
             programClassPool,
             libraryClassPool);
    }


    /**
     * Creates a new InstructionSequenceBuilder.
     * @param targetClass the target class for the instruction
     *                    constants.
     */
    public InstructionSequenceBuilder(ProgramClass targetClass)
    {
        this(new ConstantPoolEditor(targetClass));
    }


    /**
     * Creates a new InstructionSequenceBuilder that automatically initializes
     * class references and class member references in new constants.
     * @param targetClass      the target class for the instruction
     *                         constants.
     * @param programClassPool the program class pool from which new
     *                         constants can be initialized.
     * @param libraryClassPool the library class pool from which new
     *                         constants can be initialized.
     */
    public InstructionSequenceBuilder(ProgramClass targetClass,
                                      ClassPool    programClassPool,
                                      ClassPool    libraryClassPool)
    {
        this(new ConstantPoolEditor(targetClass, programClassPool, libraryClassPool));
    }


    /**
     * Creates a new InstructionSequenceBuilder.
     * @param constantPoolEditor the editor to use for creating any constants
     *                           for the instructions.
     */
    public InstructionSequenceBuilder(ConstantPoolEditor constantPoolEditor)
    {
         this.constantPoolEditor = constantPoolEditor;
    }


    /**
     * Returns the ConstantPoolEditor used by this builder to
     * create constants.
     *
     * @return the ConstantPoolEditor used by this builder to
     * create constants.
     */
    public ConstantPoolEditor getConstantPoolEditor()
    {
        return constantPoolEditor;
    }


    /**
     * Short for {@link #appendInstruction(Instruction)}.
     *
     * @see InstructionSequenceReplacer#label()
     */
    public InstructionSequenceBuilder label(Instruction instruction)
    {
        return appendInstruction(instruction);
    }


    /**
     * Short for {@link #appendInstruction(Instruction)}.
     *
     * @see InstructionSequenceReplacer#catch_(int,int,int)
     */
    public InstructionSequenceBuilder catch_(Instruction instruction)
    {
        return appendInstruction(instruction);
    }


    /**
     * Appends the given instruction.
     * @param instruction the instruction to be appended.
     * @return the builder itself.
     */
    public InstructionSequenceBuilder appendInstruction(Instruction instruction)
    {
        return add(instruction);
    }


    /**
     * Appends the given instructions.
     * @param instructions the instructions to be appended.
     * @return the builder itself.
     */
    public InstructionSequenceBuilder appendInstructions(Instruction[] instructions)
    {
        for (Instruction instruction : instructions)
        {
            add(instruction);
        }

        return this;
    }


    /**
     * Short for {@link #instructions()}.
     */
    public Instruction[] __()
    {
        return instructions();
    }


    /**
     * Returns the accumulated sequence of instructions
     * and resets the sequence in the builder.
     */
    public Instruction[] instructions()
    {
        Instruction[] instructions_array = new Instruction[instructions.size()];
        instructions.toArray(instructions_array);

        instructions.clear();

        return instructions_array;
    }


    /**
     * Returns the accumulated set of constants
     * and resets the set in the builder.
     */
    public Constant[] constants()
    {
        ProgramClass targetClass = constantPoolEditor.getTargetClass();

        Constant[] constantPool = new Constant[targetClass.u2constantPoolCount];
        System.arraycopy(targetClass.constantPool,
                         0,
                         constantPool,
                         0,
                         targetClass.u2constantPoolCount);

        targetClass.u2constantPoolCount = 0;

        return constantPool;
    }


    // Methods corresponding to the bytecode opcodes.

    public InstructionSequenceBuilder nop()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_NOP));
    }

    public InstructionSequenceBuilder aconst_null()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ACONST_NULL));
    }

    public InstructionSequenceBuilder iconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_0, constant));
    }

    public InstructionSequenceBuilder iconst_m1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_M1));
    }

    public InstructionSequenceBuilder iconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_0));
    }

    public InstructionSequenceBuilder iconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_1));
    }

    public InstructionSequenceBuilder iconst_2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_2));
    }

    public InstructionSequenceBuilder iconst_3()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_3));
    }

    public InstructionSequenceBuilder iconst_4()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_4));
    }

    public InstructionSequenceBuilder iconst_5()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ICONST_5));
    }

    public InstructionSequenceBuilder lconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_0, constant));
    }

    public InstructionSequenceBuilder lconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_0));
    }

    public InstructionSequenceBuilder lconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCONST_1));
    }

    public InstructionSequenceBuilder fconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_0, constant));
    }

    public InstructionSequenceBuilder fconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_0));
    }

    public InstructionSequenceBuilder fconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_1));
    }

    public InstructionSequenceBuilder fconst_2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCONST_2));
    }

    public InstructionSequenceBuilder dconst(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_0, constant));
    }

    public InstructionSequenceBuilder dconst_0()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_0));
    }

    public InstructionSequenceBuilder dconst_1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCONST_1));
    }

    public InstructionSequenceBuilder bipush(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BIPUSH, constant));
    }

    public InstructionSequenceBuilder sipush(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SIPUSH, constant));
    }

    public InstructionSequenceBuilder ldc(int value)
    {
        return ldc_(constantPoolEditor.addIntegerConstant(value));
    }

    public InstructionSequenceBuilder ldc(float value)
    {
        return ldc_(constantPoolEditor.addFloatConstant(value));
    }

    public InstructionSequenceBuilder ldc(Object primitiveArray)
    {
        return ldc_(constantPoolEditor.addPrimitiveArrayConstant(primitiveArray));
    }

    public InstructionSequenceBuilder ldc(String string)
    {
        return ldc(string, null, null);
    }

    public InstructionSequenceBuilder ldc(String string, Clazz referencedClass, Method referencedMember)
    {
        return ldc_(constantPoolEditor.addStringConstant(string, referencedClass, referencedMember));
    }

    public InstructionSequenceBuilder ldc(Clazz clazz)
    {
        return ldc(clazz.getName(), clazz);
    }

    public InstructionSequenceBuilder ldc(String className, Clazz referencedClass)
    {
        return ldc_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder ldc_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC, constantIndex));
    }

    public InstructionSequenceBuilder ldc_w(int value)
    {
        return ldc_w_(constantPoolEditor.addIntegerConstant(value));
    }

    public InstructionSequenceBuilder ldc_w(float value)
    {
        // If we can shrink the instruction, we may not need to create a constant.
        return ldc_w_(constantPoolEditor.addFloatConstant(value));
    }

    public InstructionSequenceBuilder ldc_w(String string)
    {
        return ldc_w(string, null, null);
    }

    public InstructionSequenceBuilder ldc_w(String string, Clazz referencedClass, Method referencedMember)
    {
        return ldc_w_(constantPoolEditor.addStringConstant(string, referencedClass, referencedMember));
    }

    public InstructionSequenceBuilder ldc_w(String className, Clazz referencedClass)
    {
        return ldc_w_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder ldc_w_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC_W, constantIndex));
    }

    public InstructionSequenceBuilder ldc2_w(long value)
    {
        // If we can shrink the instruction, we may not need to create a constant.
        return ldc2_w(constantPoolEditor.addLongConstant(value));
    }

    public InstructionSequenceBuilder ldc2_w(double value)
    {
        // If we can shrink the instruction, we may not need to create a constant.
        return ldc2_w(constantPoolEditor.addDoubleConstant(value));
    }

    public InstructionSequenceBuilder ldc2_w(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_LDC2_W, constantIndex));
    }

    public InstructionSequenceBuilder iload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD, variableIndex));
    }

    public InstructionSequenceBuilder lload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD, variableIndex));
    }

    public InstructionSequenceBuilder fload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD, variableIndex));
    }

    public InstructionSequenceBuilder dload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD, variableIndex));
    }

    public InstructionSequenceBuilder aload(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD, variableIndex));
    }

    public InstructionSequenceBuilder iload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_0));
    }

    public InstructionSequenceBuilder iload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_1));
    }

    public InstructionSequenceBuilder iload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_2));
    }

    public InstructionSequenceBuilder iload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ILOAD_3));
    }

    public InstructionSequenceBuilder lload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_0));
    }

    public InstructionSequenceBuilder lload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_1));
    }

    public InstructionSequenceBuilder lload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_2));
    }

    public InstructionSequenceBuilder lload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LLOAD_3));
    }

    public InstructionSequenceBuilder fload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_0));
    }

    public InstructionSequenceBuilder fload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_1));
    }

    public InstructionSequenceBuilder fload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_2));
    }

    public InstructionSequenceBuilder fload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FLOAD_3));
    }

    public InstructionSequenceBuilder dload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_0));
    }

    public InstructionSequenceBuilder dload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_1));
    }

    public InstructionSequenceBuilder dload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_2));
    }

    public InstructionSequenceBuilder dload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DLOAD_3));
    }

    public InstructionSequenceBuilder aload_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_0));
    }

    public InstructionSequenceBuilder aload_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_1));
    }

    public InstructionSequenceBuilder aload_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_2));
    }

    public InstructionSequenceBuilder aload_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ALOAD_3));
    }

    public InstructionSequenceBuilder iaload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IALOAD));
    }

    public InstructionSequenceBuilder laload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LALOAD));
    }

    public InstructionSequenceBuilder faload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FALOAD));
    }

    public InstructionSequenceBuilder daload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DALOAD));
    }

    public InstructionSequenceBuilder aaload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_AALOAD));
    }

    public InstructionSequenceBuilder baload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BALOAD));
    }

    public InstructionSequenceBuilder caload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_CALOAD));
    }

    public InstructionSequenceBuilder saload()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SALOAD));
    }

    public InstructionSequenceBuilder istore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE, variableIndex));
    }

    public InstructionSequenceBuilder lstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE, variableIndex));
    }

    public InstructionSequenceBuilder fstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE, variableIndex));
    }

    public InstructionSequenceBuilder dstore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE, variableIndex));
    }

    public InstructionSequenceBuilder astore(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE, variableIndex));
    }

    public InstructionSequenceBuilder istore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_0));
    }

    public InstructionSequenceBuilder istore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_1));
    }

    public InstructionSequenceBuilder istore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_2));
    }

    public InstructionSequenceBuilder istore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ISTORE_3));
    }

    public InstructionSequenceBuilder lstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_0));
    }

    public InstructionSequenceBuilder lstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_1));
    }

    public InstructionSequenceBuilder lstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_2));
    }

    public InstructionSequenceBuilder lstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_LSTORE_3));
    }

    public InstructionSequenceBuilder fstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_0));
    }

    public InstructionSequenceBuilder fstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_1));
    }

    public InstructionSequenceBuilder fstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_2));
    }

    public InstructionSequenceBuilder fstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_FSTORE_3));
    }

    public InstructionSequenceBuilder dstore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_0));
    }

    public InstructionSequenceBuilder dstore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_1));
    }

    public InstructionSequenceBuilder dstore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_2));
    }

    public InstructionSequenceBuilder dstore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_DSTORE_3));
    }

    public InstructionSequenceBuilder astore_0()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_0));
    }

    public InstructionSequenceBuilder astore_1()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_1));
    }

    public InstructionSequenceBuilder astore_2()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_2));
    }

    public InstructionSequenceBuilder astore_3()
    {
        return add(new VariableInstruction(InstructionConstants.OP_ASTORE_3));
    }

    public InstructionSequenceBuilder iastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IASTORE));
    }

    public InstructionSequenceBuilder lastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LASTORE));
    }

    public InstructionSequenceBuilder fastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FASTORE));
    }

    public InstructionSequenceBuilder dastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DASTORE));
    }

    public InstructionSequenceBuilder aastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_AASTORE));
    }

    public InstructionSequenceBuilder bastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_BASTORE));
    }

    public InstructionSequenceBuilder castore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_CASTORE));
    }

    public InstructionSequenceBuilder sastore()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SASTORE));
    }

    public InstructionSequenceBuilder pop()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_POP));
    }

    public InstructionSequenceBuilder pop2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_POP2));
    }

    public InstructionSequenceBuilder dup()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP));
    }

    public InstructionSequenceBuilder dup_x1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP_X1));
    }

    public InstructionSequenceBuilder dup_x2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP_X2));
    }

    public InstructionSequenceBuilder dup2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2));
    }

    public InstructionSequenceBuilder dup2_x1()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2_X1));
    }

    public InstructionSequenceBuilder dup2_x2()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DUP2_X2));
    }

    public InstructionSequenceBuilder swap()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_SWAP));
    }

    public InstructionSequenceBuilder iadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IADD));
    }

    public InstructionSequenceBuilder ladd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LADD));
    }

    public InstructionSequenceBuilder fadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FADD));
    }

    public InstructionSequenceBuilder dadd()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DADD));
    }

    public InstructionSequenceBuilder isub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISUB));
    }

    public InstructionSequenceBuilder lsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSUB));
    }

    public InstructionSequenceBuilder fsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FSUB));
    }

    public InstructionSequenceBuilder dsub()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DSUB));
    }

    public InstructionSequenceBuilder imul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IMUL));
    }

    public InstructionSequenceBuilder lmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LMUL));
    }

    public InstructionSequenceBuilder fmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FMUL));
    }

    public InstructionSequenceBuilder dmul()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DMUL));
    }

    public InstructionSequenceBuilder idiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IDIV));
    }

    public InstructionSequenceBuilder ldiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LDIV));
    }

    public InstructionSequenceBuilder fdiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FDIV));
    }

    public InstructionSequenceBuilder ddiv()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DDIV));
    }

    public InstructionSequenceBuilder irem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IREM));
    }

    public InstructionSequenceBuilder lrem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LREM));
    }

    public InstructionSequenceBuilder frem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FREM));
    }

    public InstructionSequenceBuilder drem()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DREM));
    }

    public InstructionSequenceBuilder ineg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_INEG));
    }

    public InstructionSequenceBuilder lneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LNEG));
    }

    public InstructionSequenceBuilder fneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FNEG));
    }

    public InstructionSequenceBuilder dneg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DNEG));
    }

    public InstructionSequenceBuilder ishl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISHL));
    }

    public InstructionSequenceBuilder lshl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSHL));
    }

    public InstructionSequenceBuilder ishr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ISHR));
    }

    public InstructionSequenceBuilder lshr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LSHR));
    }

    public InstructionSequenceBuilder iushr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IUSHR));
    }

    public InstructionSequenceBuilder lushr()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LUSHR));
    }

    public InstructionSequenceBuilder iand()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IAND));
    }

    public InstructionSequenceBuilder land()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LAND));
    }

    public InstructionSequenceBuilder ior()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IOR));
    }

    public InstructionSequenceBuilder lor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LOR));
    }

    public InstructionSequenceBuilder ixor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IXOR));
    }

    public InstructionSequenceBuilder lxor()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LXOR));
    }

    public InstructionSequenceBuilder iinc(int variableIndex,
                                           int constant)
    {
        return add(new VariableInstruction(InstructionConstants.OP_IINC, variableIndex, constant));
    }

    public InstructionSequenceBuilder i2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2L));
    }

    public InstructionSequenceBuilder i2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2F));
    }

    public InstructionSequenceBuilder i2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2D));
    }

    public InstructionSequenceBuilder l2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2I));
    }

    public InstructionSequenceBuilder l2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2F));
    }

    public InstructionSequenceBuilder l2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_L2D));
    }

    public InstructionSequenceBuilder f2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2I));
    }

    public InstructionSequenceBuilder f2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2L));
    }

    public InstructionSequenceBuilder f2d()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_F2D));
    }

    public InstructionSequenceBuilder d2i()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2I));
    }

    public InstructionSequenceBuilder d2l()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2L));
    }

    public InstructionSequenceBuilder d2f()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_D2F));
    }

    public InstructionSequenceBuilder i2b()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2B));
    }

    public InstructionSequenceBuilder i2c()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2C));
    }

    public InstructionSequenceBuilder i2s()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_I2S));
    }

    public InstructionSequenceBuilder lcmp()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LCMP));
    }

    public InstructionSequenceBuilder fcmpl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCMPL));
    }

    public InstructionSequenceBuilder fcmpg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FCMPG));
    }

    public InstructionSequenceBuilder dcmpl()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCMPL));
    }

    public InstructionSequenceBuilder dcmpg()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DCMPG));
    }

    public InstructionSequenceBuilder ifeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFEQ, branchOffset));
    }

    public InstructionSequenceBuilder ifne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNE, branchOffset));
    }

    public InstructionSequenceBuilder iflt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFLT, branchOffset));
    }

    public InstructionSequenceBuilder ifge(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFGE, branchOffset));
    }

    public InstructionSequenceBuilder ifgt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFGT, branchOffset));
    }

    public InstructionSequenceBuilder ifle(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFLE, branchOffset));
    }

    public InstructionSequenceBuilder ificmpeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPEQ, branchOffset));
    }

    public InstructionSequenceBuilder ificmpne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPNE, branchOffset));
    }

    public InstructionSequenceBuilder ificmplt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPLT, branchOffset));
    }

    public InstructionSequenceBuilder ificmpge(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPGE, branchOffset));
    }

    public InstructionSequenceBuilder ificmpgt(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPGT, branchOffset));
    }

    public InstructionSequenceBuilder ificmple(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFICMPLE, branchOffset));
    }

    public InstructionSequenceBuilder ifacmpeq(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFACMPEQ, branchOffset));
    }

    public InstructionSequenceBuilder ifacmpne(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFACMPNE, branchOffset));
    }

    public InstructionSequenceBuilder goto_(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_GOTO, branchOffset));
    }

    public InstructionSequenceBuilder jsr(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_JSR, branchOffset));
    }

    public InstructionSequenceBuilder ret(int variableIndex)
    {
        return add(new VariableInstruction(InstructionConstants.OP_RET, variableIndex));
    }

    public InstructionSequenceBuilder tableswitch(int   defaultOffset,
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

    public InstructionSequenceBuilder lookupswitch(int  defaultOffset,
                                                    int[] cases,
                                                    int[] jumpOffsets)
    {
        return add(new LookUpSwitchInstruction(InstructionConstants.OP_LOOKUPSWITCH,
                                                     defaultOffset,
                                                     cases,
                                                     jumpOffsets));
    }

    public InstructionSequenceBuilder ireturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_IRETURN));
    }

    public InstructionSequenceBuilder lreturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_LRETURN));
    }

    public InstructionSequenceBuilder freturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_FRETURN));
    }

    public InstructionSequenceBuilder dreturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_DRETURN));
    }

    public InstructionSequenceBuilder areturn()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ARETURN));
    }

    public InstructionSequenceBuilder return_()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_RETURN));
    }

    public InstructionSequenceBuilder getstatic(Clazz  referencedClass,
                                                Member referencedMember)
    {
        return getstatic(referencedClass.getName(),
                         referencedMember.getName(referencedClass),
                         referencedMember.getDescriptor(referencedClass),
                         referencedClass,
                         referencedMember);
    }

    public InstructionSequenceBuilder getstatic(String className,
                                                String name,
                                                String descriptor)
    {
        return getstatic(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder getstatic(String className,
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

    public InstructionSequenceBuilder getstatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_GETSTATIC, constantIndex));
    }

    public InstructionSequenceBuilder putstatic(Clazz  referencedClass,
                                                Member referencedMember)
    {
        return putstatic(referencedClass.getName(),
                         referencedMember.getName(referencedClass),
                         referencedMember.getDescriptor(referencedClass),
                         referencedClass,
                         referencedMember);
    }

    public InstructionSequenceBuilder putstatic(String className,
                                                String name,
                                                String descriptor)
    {
        return putstatic(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder putstatic(String className,
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

    public InstructionSequenceBuilder putstatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_PUTSTATIC, constantIndex));
    }

    public InstructionSequenceBuilder getfield(Clazz  referencedClass,
                                               Member referencedMember)
    {
        return getfield(referencedClass.getName(),
                        referencedMember.getName(referencedClass),
                        referencedMember.getDescriptor(referencedClass),
                        referencedClass,
                        referencedMember);
    }

    public InstructionSequenceBuilder getfield(String className,
                                               String name,
                                               String descriptor)
    {
        return getfield(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder getfield(String className,
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

    public InstructionSequenceBuilder getfield(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_GETFIELD, constantIndex));
    }

    public InstructionSequenceBuilder putfield(Clazz  referencedClass,
                                               Member referencedMember)
    {
        return putfield(referencedClass.getName(),
                        referencedMember.getName(referencedClass),
                        referencedMember.getDescriptor(referencedClass),
                        referencedClass,
                        referencedMember);
    }

    public InstructionSequenceBuilder putfield(String className,
                                               String name,
                                               String descriptor)
    {
        return putfield(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder putfield(String className,
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

    public InstructionSequenceBuilder putfield(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_PUTFIELD, constantIndex));
    }

    public InstructionSequenceBuilder invokevirtual(Clazz  referencedClass,
                                                    Member referencedMember)
    {
        return invokevirtual(referencedClass.getName(),
                             referencedMember.getName(referencedClass),
                             referencedMember.getDescriptor(referencedClass),
                             referencedClass,
                             referencedMember);
    }

    public InstructionSequenceBuilder invokevirtual(String className,
                                                    String name,
                                                    String descriptor)
    {
        return invokevirtual(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokevirtual(int    classIndex,
                                                    String name,
                                                    String descriptor)
    {
        return invokevirtual(constantPoolEditor.addMethodrefConstant(classIndex,
                                                                     name,
                                                                     descriptor,
                                                                     null,
                                                                     null));
    }

    public InstructionSequenceBuilder invokevirtual(String className,
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

    public InstructionSequenceBuilder invokevirtual(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, constantIndex));
    }

    public InstructionSequenceBuilder invokespecial(String className,
                                                    String name,
                                                    String descriptor)
    {
        return invokespecial(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokespecial(Clazz  referencedClass,
                                                    Member referencedMember)
    {
        return invokespecial(referencedClass.getName(),
                             referencedMember.getName(referencedClass),
                             referencedMember.getDescriptor(referencedClass),
                             referencedClass,
                             referencedMember);
    }

    public InstructionSequenceBuilder invokespecial(String className,
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

    public InstructionSequenceBuilder invokespecial_interface(String className,
                                                              String name,
                                                    String descriptor)
    {
        return invokespecial_interface(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokespecial_interface(Clazz  referencedClass,
                                                              Member referencedMember)
    {
        return invokespecial_interface(referencedClass.getName(),
                             referencedMember.getName(referencedClass),
                             referencedMember.getDescriptor(referencedClass),
                             referencedClass,
                             referencedMember);
    }

    public InstructionSequenceBuilder invokespecial_interface(String className,
                                                    String name,
                                                    String descriptor,
                                                    Clazz  referencedClass,
                                                    Member referencedMember)
    {
        return invokespecial(constantPoolEditor.addInterfaceMethodrefConstant(className,
                                                                              name,
                                                                              descriptor,
                                                                              referencedClass,
                                                                              referencedMember));
    }

    public InstructionSequenceBuilder invokespecial(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKESPECIAL, constantIndex));
    }

    public InstructionSequenceBuilder invokestatic(Clazz  referencedClass,
                                                   Member referencedMember)
    {
        return invokestatic(referencedClass.getName(),
                            referencedMember.getName(referencedClass),
                            referencedMember.getDescriptor(referencedClass),
                            referencedClass,
                            referencedMember);
    }

    public InstructionSequenceBuilder invokestatic(String className,
                                                   String name,
                                                   String descriptor)
    {
        return invokestatic(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokestatic(String className,
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

    public InstructionSequenceBuilder invokestatic_interface(String className,
                                                             String name,
                                                             String descriptor)
    {
        return invokestatic_interface(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokestatic_interface(String className,
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

    public InstructionSequenceBuilder invokestatic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, constantIndex));
    }

    public InstructionSequenceBuilder invokeinterface(Clazz  referencedClass,
                                                      Member referencedMember)
    {
        return invokeinterface(referencedClass.getName(),
                               referencedMember.getName(referencedClass),
                               referencedMember.getDescriptor(referencedClass),
                               referencedClass,
                               referencedMember);
    }

    public InstructionSequenceBuilder invokeinterface(String className,
                                                      String name,
                                                      String descriptor)
    {
        return invokeinterface(className, name, descriptor, null, null);
    }

    public InstructionSequenceBuilder invokeinterface(String className,
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

    public InstructionSequenceBuilder invokeinterface(int constantIndex, int constant)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEINTERFACE, constantIndex, constant));
    }

    public InstructionSequenceBuilder invokedynamic(int    bootStrapMethodIndex,
                                                    String name,
                                                    String descriptor)
    {
        return invokedynamic(bootStrapMethodIndex, name, descriptor, null);
    }

    public InstructionSequenceBuilder invokedynamic(int     bootStrapMethodIndex,
                                                    String  name,
                                                    String  descriptor,
                                                    Clazz[] referencedClasses)
    {
        return invokedynamic(constantPoolEditor.addInvokeDynamicConstant(bootStrapMethodIndex,
                                                                         name,
                                                                         descriptor,
                                                                         referencedClasses));
    }

    public InstructionSequenceBuilder invokedynamic(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INVOKEDYNAMIC, constantIndex));
    }

    public InstructionSequenceBuilder new_(String className)
    {
        return new_(className, null);
    }

    public InstructionSequenceBuilder new_(String className, Clazz referencedClass)
    {
        return new_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder new_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_NEW, constantIndex));
    }

    public InstructionSequenceBuilder newarray(int constant)
    {
        return add(new SimpleInstruction(InstructionConstants.OP_NEWARRAY, constant));
    }

    public InstructionSequenceBuilder anewarray(String className, Clazz referencedClass)
    {
        return anewarray(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder anewarray(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, constantIndex));
    }

    public InstructionSequenceBuilder arraylength()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ARRAYLENGTH));
    }

    public InstructionSequenceBuilder athrow()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_ATHROW));
    }

    public InstructionSequenceBuilder checkcast(String className)
    {
        return checkcast(className, null);
    }

    public InstructionSequenceBuilder checkcast(String className, Clazz referencedClass)
    {
        return checkcast(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder checkcast(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_CHECKCAST, constantIndex));
    }

    public InstructionSequenceBuilder instanceof_(String className, Clazz referencedClass)
    {
        return instanceof_(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder instanceof_(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_INSTANCEOF, constantIndex));
    }

    public InstructionSequenceBuilder monitorenter()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_MONITORENTER));
    }

    public InstructionSequenceBuilder monitorexit()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_MONITOREXIT));
    }

    public InstructionSequenceBuilder wide()
    {
        return add(new SimpleInstruction(InstructionConstants.OP_WIDE));
    }

    public InstructionSequenceBuilder multianewarray(String className, Clazz referencedClass)
    {
        return multianewarray(constantPoolEditor.addClassConstant(className, referencedClass));
    }

    public InstructionSequenceBuilder multianewarray(int constantIndex)
    {
        return add(new ConstantInstruction(InstructionConstants.OP_MULTIANEWARRAY, constantIndex));
    }

    public InstructionSequenceBuilder ifnull(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNULL, branchOffset));
    }

    public InstructionSequenceBuilder ifnonnull(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_IFNONNULL, branchOffset));
    }

    public InstructionSequenceBuilder goto_w(int branchOffset)
    {
        return add(new BranchInstruction(InstructionConstants.OP_GOTO_W, branchOffset));
    }

    public InstructionSequenceBuilder jsr_w(int branchOffset)
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
     * @param value        the primitive value to be pushed - should never be null.
     * @param type the internal type of the primitive ('Z','B','I',...)
     */
    public InstructionSequenceBuilder pushPrimitive(Object value,
                                                    char   type)
    {
        switch (type)
        {
            case ClassConstants.TYPE_BOOLEAN: return ((Boolean)value).booleanValue() ? iconst_1() : iconst_0();
            case ClassConstants.TYPE_BYTE:
            case ClassConstants.TYPE_SHORT:
            case ClassConstants.TYPE_INT:     return pushInt(((Number)value).intValue());
            case ClassConstants.TYPE_CHAR:    return ldc(((Character)value).charValue());
            case ClassConstants.TYPE_LONG:    return ldc2_w((Long)value);
            case ClassConstants.TYPE_FLOAT:   return ldc(((Float)value).floatValue());
            case ClassConstants.TYPE_DOUBLE:  return ldc2_w((Double)value);
            default: throw new IllegalArgumentException("" + type);
        }
    }


    /**
     * Pushes the given primitive int on the stack in the most efficient way
     * (as an iconst, bipush, sipush, or ldc instruction).
     *
     * @param value the int value to be pushed.
     */
    public InstructionSequenceBuilder pushInt(int value)
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
    public InstructionSequenceBuilder pushFloat(float value)
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
    public InstructionSequenceBuilder pushLong(long value)
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
    public InstructionSequenceBuilder pushDouble(double value)
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
     * @param type the array element type (or class name in case of objects).
     * @param size the size of the array to be created.
     */
    public InstructionSequenceBuilder pushNewArray(String type,
                                                   int    size)
    {
        // Create new array.
        pushInt(size);

        return ClassUtil.isInternalPrimitiveType(type) ?
            newarray(InstructionUtil.arrayTypeFromInternalType(type.charAt(0))) :
            anewarray(type, null);
    }


    /**
     * Loads the given variable onto the stack.
     *
     * Operand stack:
     * ... -> ..., value
     *
     * @param variableIndex the index of the variable to be loaded.
     * @param type  the type of the variable to be loaded.
     */
    public InstructionSequenceBuilder load(int    variableIndex,
                                           String type)
    {
        return load(variableIndex, type.charAt(0));
    }


    /**
     * Loads the given variable of primitive type onto the stack.
     *
     * Operand stack:
     * ... -> ..., value
     *
     * @param variableIndex the index of the variable to be loaded.
     * @param type          the type of the variable to be loaded.
     */
    public InstructionSequenceBuilder load(int  variableIndex,
                                           char type)
    {
        switch (type)
        {
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:
            case ClassConstants.TYPE_CHAR:
            case ClassConstants.TYPE_SHORT:
            case ClassConstants.TYPE_INT:    return iload(variableIndex);
            case ClassConstants.TYPE_LONG:   return lload(variableIndex);
            case ClassConstants.TYPE_FLOAT:  return fload(variableIndex);
            case ClassConstants.TYPE_DOUBLE: return dload(variableIndex);
            default:                         return aload(variableIndex);
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
     * @param type          the type of the value to be stored.
     */
    public InstructionSequenceBuilder store(int    variableIndex,
                                            String type)
    {
        return store(variableIndex, type.charAt(0));
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
     * @param type          the type of the value to be stored.
     */
    public InstructionSequenceBuilder store(int  variableIndex,
                                            char type)
    {
        switch (type)
        {
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:
            case ClassConstants.TYPE_CHAR:
            case ClassConstants.TYPE_SHORT:
            case ClassConstants.TYPE_INT:    return istore(variableIndex);
            case ClassConstants.TYPE_LONG:   return lstore(variableIndex);
            case ClassConstants.TYPE_FLOAT:  return fstore(variableIndex);
            case ClassConstants.TYPE_DOUBLE: return dstore(variableIndex);
            default:                         return astore(variableIndex);
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
    public InstructionSequenceBuilder storeToArray(String elementType)
    {
        // Store element on stack in array.
        switch (elementType.charAt(0))
        {
            case ClassConstants.TYPE_BOOLEAN:
            case ClassConstants.TYPE_BYTE:   return bastore();
            case ClassConstants.TYPE_CHAR:   return castore();
            case ClassConstants.TYPE_SHORT:  return sastore();
            case ClassConstants.TYPE_INT:    return iastore();
            case ClassConstants.TYPE_LONG:   return lastore();
            case ClassConstants.TYPE_FLOAT:  return fastore();
            case ClassConstants.TYPE_DOUBLE: return dastore();
            default:                         return aastore();
        }
    }


    // Small utility methods.

    /**
     * Adds the given instruction, shrinking it if necessary.
     */
    private InstructionSequenceBuilder add(Instruction instruction)
    {
        instructions.add(instruction);

        return this;
    }


    /**
     * This main method tests the class.
     */
    public static void main(String[] args)
    {
        InstructionSequenceBuilder builder = new InstructionSequenceBuilder();

        Instruction[] instructions = builder
            .iconst_2()
            .istore_0()
            .iinc(0, 2)
            .iload_0()
            .ldc(12)
            .iadd()
            .putstatic("com/example/SomeClass", "someField", "I", null, null)
            .instructions();

        Constant[] constants = builder.constants();

        for (Instruction instruction : instructions)
        {
            System.out.println(instruction);
        }

        System.out.println();

        for (int index = 0; index < constants.length; index++)
        {
            System.out.println("#"+index+": " + constants[index]);
        }
    }


    /**
     * This ProgramClass is a dummy container for a constant pool, with a null name.
     */
    private static class MyDummyClass
    extends              ProgramClass
    {
        public MyDummyClass()
        {
            super(ClassConstants.CLASS_VERSION_1_0, 1, new Constant[256], 0, 0, 0);
        }


        // Overriding methods for Claaz.

        public String getName()
        {
            return null;
        }
    }
}
