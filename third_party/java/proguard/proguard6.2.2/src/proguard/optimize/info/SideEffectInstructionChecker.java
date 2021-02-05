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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This class can tell whether an instruction has any side effects. Return
 * instructions and array store instructions can be included or not.
 *
 * @see ReadWriteFieldMarker
 * @see SideEffectClassMarker
 * @see NoSideEffectMethodMarker
 * @see SideEffectMethodMarker
 * @author Eric Lafortune
 */
public class SideEffectInstructionChecker
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor,
             MemberVisitor
{
    public static final boolean OPTIMIZE_CONSERVATIVELY = System.getProperty("optimize.conservatively") != null;


    private final boolean includeReturnInstructions;
    private final boolean includeArrayStoreInstructions;

    // Parameters and return values for the visitor methods.
    private boolean writingField;
    private Clazz   referencingClass;
    private boolean hasSideEffects;


    /**
     * Creates a new SideEffectInstructionChecker
     * @param includeReturnInstructions     specifies whether return
     *                                      instructions count as side
     *                                      effects.
     * @param includeArrayStoreInstructions specifies whether storing values
     *                                      in arrays counts as side effects.
     */
    public SideEffectInstructionChecker(boolean includeReturnInstructions,
                                        boolean includeArrayStoreInstructions)
    {
        this.includeReturnInstructions     = includeReturnInstructions;
        this.includeArrayStoreInstructions = includeArrayStoreInstructions;
    }


    public boolean hasSideEffects(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        hasSideEffects = false;

        instruction.accept(clazz, method, codeAttribute, offset, this);

        return hasSideEffects;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        byte opcode = simpleInstruction.opcode;

        // Check for instructions that might cause side effects.
        switch (opcode)
        {
            case InstructionConstants.OP_IDIV:
            case InstructionConstants.OP_LDIV:
            case InstructionConstants.OP_IREM:
            case InstructionConstants.OP_LREM:
            case InstructionConstants.OP_FDIV:
            case InstructionConstants.OP_FREM:
            case InstructionConstants.OP_DDIV:
            case InstructionConstants.OP_DREM:
            case InstructionConstants.OP_IALOAD:
            case InstructionConstants.OP_LALOAD:
            case InstructionConstants.OP_FALOAD:
            case InstructionConstants.OP_DALOAD:
            case InstructionConstants.OP_AALOAD:
            case InstructionConstants.OP_BALOAD:
            case InstructionConstants.OP_CALOAD:
            case InstructionConstants.OP_SALOAD:
            case InstructionConstants.OP_NEWARRAY:
            case InstructionConstants.OP_ARRAYLENGTH:
                // These instructions strictly taken may cause a side effect
                // (ArithmeticException, NullPointerException,
                // ArrayIndexOutOfBoundsException, NegativeArraySizeException).
                hasSideEffects = OPTIMIZE_CONSERVATIVELY;
                break;

            case InstructionConstants.OP_IASTORE:
            case InstructionConstants.OP_LASTORE:
            case InstructionConstants.OP_FASTORE:
            case InstructionConstants.OP_DASTORE:
            case InstructionConstants.OP_AASTORE:
            case InstructionConstants.OP_BASTORE:
            case InstructionConstants.OP_CASTORE:
            case InstructionConstants.OP_SASTORE:
                // These instructions may cause a side effect.
                hasSideEffects = includeArrayStoreInstructions;
                break;

            case InstructionConstants.OP_ATHROW :
            case InstructionConstants.OP_MONITORENTER:
            case InstructionConstants.OP_MONITOREXIT:
                // These instructions always cause a side effect.
                hasSideEffects = true;
                break;

            case InstructionConstants.OP_IRETURN:
            case InstructionConstants.OP_LRETURN:
            case InstructionConstants.OP_FRETURN:
            case InstructionConstants.OP_DRETURN:
            case InstructionConstants.OP_ARETURN:
            case InstructionConstants.OP_RETURN:
                // These instructions may have a side effect.
                hasSideEffects = includeReturnInstructions;
                break;
        }
    }


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        byte opcode = variableInstruction.opcode;

        // Check for instructions that might cause side effects.
        switch (opcode)
        {
            case InstructionConstants.OP_RET:
                // This instruction may have a side effect.
                hasSideEffects = includeReturnInstructions;
                break;
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        byte opcode = constantInstruction.opcode;

        // Check for instructions that might cause side effects.
        switch (opcode)
        {
            case InstructionConstants.OP_GETSTATIC:
                // Check if accessing the field might cause any side effects.
                writingField = false;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_PUTSTATIC:
                // Check if accessing the field might cause any side effects.
                writingField = true;
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_GETFIELD:
                if (OPTIMIZE_CONSERVATIVELY)
                {
                    // These instructions strictly taken may cause a side effect
                    // (NullPointerException).
                    hasSideEffects = true;
                }
                else
                {
                    // Check if the field is write-only or volatile.
                    writingField = false;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                }
                break;

            case InstructionConstants.OP_PUTFIELD:
                if (OPTIMIZE_CONSERVATIVELY)
                {
                    // These instructions strictly taken may cause a side effect
                    // (NullPointerException).
                    hasSideEffects = true;
                }
                else
                {
                    // Check if the field is write-only or volatile.
                    writingField = true;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                }
                break;

            case InstructionConstants.OP_INVOKESPECIAL:
            case InstructionConstants.OP_INVOKESTATIC:
                // Check if the invoked method is causing any side effects.
                clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                break;

            case InstructionConstants.OP_INVOKEVIRTUAL:
            case InstructionConstants.OP_INVOKEINTERFACE:
            case InstructionConstants.OP_INVOKEDYNAMIC:
                if (OPTIMIZE_CONSERVATIVELY)
                {
                    // These instructions strictly taken may cause a side effect
                    // (NullPointerException).
                    hasSideEffects = true;
                }
                else
                {
                    // Check if the invoked method is causing any side effects.
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                }
                break;

            case InstructionConstants.OP_ANEWARRAY:
            case InstructionConstants.OP_MULTIANEWARRAY:
            case InstructionConstants.OP_CHECKCAST:
                // This instructions strictly taken may cause a side effect
                // (ClassCastException, NegativeArraySizeException).
                hasSideEffects = OPTIMIZE_CONSERVATIVELY;
                break;
        }
    }


    public void visitBranchInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, BranchInstruction branchInstruction)
    {
        byte opcode = branchInstruction.opcode;

        // Check for instructions that might cause side effects.
        switch (opcode)
        {
            case InstructionConstants.OP_JSR:
            case InstructionConstants.OP_JSR_W:
                hasSideEffects = includeReturnInstructions;
                break;
        }
    }


    // Implementations for ConstantVisitor.

    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        // We'll have to assume invoking an unknown method has side effects.
        hasSideEffects = true;
    }


    public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
    {
        // Pass the referencing class.
        referencingClass = clazz;

        // We'll have to assume accessing an unknown field has side effects.
        hasSideEffects = true;

        // Check the referenced field, if known.
        fieldrefConstant.referencedMemberAccept(this);
    }


    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
    {
        // Pass the referencing class.
        referencingClass = clazz;

        // We'll have to assume invoking an unknown method has side effects.
        hasSideEffects = true;

        // Check the referenced method, if known.
        refConstant.referencedMemberAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        hasSideEffects =
            (writingField && ReadWriteFieldMarker.isRead(programField))        ||
            (programField.getAccessFlags() & ClassConstants.ACC_VOLATILE) != 0 ||
            SideEffectClassChecker.mayHaveSideEffects(referencingClass,
                                                      programClass,
                                                      programField);
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Note that side effects already include synchronization of some
        // implementation of the method.
        hasSideEffects =
            SideEffectMethodMarker.hasSideEffects(programMethod) ||
            SideEffectClassChecker.mayHaveSideEffects(referencingClass,
                                                      programClass,
                                                      programMethod);
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        hasSideEffects = true;
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        hasSideEffects =
            !NoSideEffectMethodMarker.hasNoSideEffects(libraryMethod);
    }
}
