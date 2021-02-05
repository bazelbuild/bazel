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
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This class can tell whether an instruction might throw exceptions.
 *
 * @author Eric Lafortune
 */
public class ExceptionInstructionChecker
extends      SimplifiedVisitor
implements   InstructionVisitor
//             ConstantVisitor,
//             MemberVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    public  static       boolean DEBUG = System.getProperty("eic") != null;
    //*/


    // A return value for the visitor methods.
    private boolean mayThrowExceptions;


    /**
     * Returns whether the specified method may throw exceptions.
     */
    public boolean mayThrowExceptions(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute)
    {
        return mayThrowExceptions(clazz,
                                  method,
                                  codeAttribute,
                                  0,
                                  codeAttribute.u4codeLength);
    }


    /**
     * Returns whether the specified block of code may throw exceptions.
     */
    public boolean mayThrowExceptions(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute,
                                      int           startOffset,
                                      int           endOffset)
    {
        if (DEBUG)
        {
            System.out.println("ExceptionInstructionChecker.mayThrowExceptions ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]: "+startOffset+" -> "+endOffset);
        }

        return firstExceptionThrowingInstructionOffset(clazz,
                                                       method,
                                                       codeAttribute,
                                                       startOffset,
                                                       endOffset) < endOffset;
    }


    /**
     * Returns the offset of the first instruction in the specified block of
     * code that may throw exceptions, or the end offset if there is none.
     */
    public int firstExceptionThrowingInstructionOffset(Clazz         clazz,
                                                       Method        method,
                                                       CodeAttribute codeAttribute,
                                                       int           startOffset,
                                                       int           endOffset)
    {
        if (DEBUG)
        {
            System.out.println("ExceptionInstructionChecker.firstExceptionThrowingInstructionOffset ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]: "+startOffset+" -> "+endOffset);
        }

        byte[] code = codeAttribute.code;

        // Go over all instructions.
        int offset = startOffset;
        while (offset < endOffset)
        {
            // Get the current instruction.
            Instruction instruction = InstructionFactory.create(code, offset);

            // Check if it may be throwing exceptions.
            if (mayThrowExceptions(clazz,
                                   method,
                                   codeAttribute,
                                   offset,
                                   instruction))
            {
                if (DEBUG)
                {
                    System.out.println("  "+instruction.toString(offset));
                }

                return offset;
            }

            // Go to the next instruction.
            offset += instruction.length(offset);
        }

        return endOffset;
    }


    /**
     * Returns the offset after the last instruction in the specified block of
     * code that may throw exceptions, or the start offset if there is none.
     */
    public int lastExceptionThrowingInstructionOffset(Clazz         clazz,
                                                       Method        method,
                                                       CodeAttribute codeAttribute,
                                                       int           startOffset,
                                                       int           endOffset)
    {
        if (DEBUG)
        {
            System.out.println("ExceptionInstructionChecker.lastExceptionThrowingInstructionOffset ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]: "+startOffset+" -> "+endOffset);
        }

        byte[] code = codeAttribute.code;

        int lastOffset = startOffset;

        // Go over all instructions.
        int offset = startOffset;
        while (offset < endOffset)
        {
            // Get the current instruction.
            Instruction instruction = InstructionFactory.create(code, offset);

            // Check if it may be throwing exceptions.
            if (mayThrowExceptions(clazz,
                                   method,
                                   codeAttribute,
                                   offset,
                                   instruction))
            {
                if (DEBUG)
                {
                    System.out.println("  "+instruction.toString(offset));
                }

                // Go to the next instruction.
                offset += instruction.length(offset);

                lastOffset = offset;
            }
            else
            {
                // Go to the next instruction.
                offset += instruction.length(offset);
            }
        }

        return lastOffset;
    }


    /**
     * Returns whether the specified instruction may throw exceptions.
     */
    public boolean mayThrowExceptions(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute,
                                      int           offset)
    {
        Instruction instruction = InstructionFactory.create(codeAttribute.code, offset);

        return mayThrowExceptions(clazz,
                                  method,
                                  codeAttribute,
                                  offset,
                                  instruction);
    }


    /**
     * Returns whether the given instruction may throw exceptions.
     */
    public boolean mayThrowExceptions(Clazz         clazz,
                                      Method        method,
                                      CodeAttribute codeAttribute,
                                      int           offset,
                                      Instruction   instruction)
    {
        return instruction.mayThrowExceptions();

//        mayThrowExceptions = false;
//
//        instruction.accept(clazz, method,  codeAttribute, offset, this);
//
//        return mayThrowExceptions;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
    {
        // Check for instructions that may throw exceptions.
        // Note that monitorexit can not sensibly throw exceptions, except the
        // broken and deprecated asynchronous ThreadDeath. Removing the
        // artificial infinite looping exception blocks that recent compilers
        // add does not strictly follow the JVM specs, but it does have the
        // additional benefit of avoiding a bug in the JVM in JDK 1.1.
        switch (simpleInstruction.opcode)
        {
            case InstructionConstants.OP_IDIV:
            case InstructionConstants.OP_LDIV:
            case InstructionConstants.OP_IREM:
            case InstructionConstants.OP_LREM:
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
            case InstructionConstants.OP_NEWARRAY:
            case InstructionConstants.OP_ARRAYLENGTH:
            case InstructionConstants.OP_ATHROW:
            case InstructionConstants.OP_MONITORENTER:
                // These instructions may throw exceptions.
                mayThrowExceptions = true;
        }
    }


    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        // Check for instructions that may throw exceptions.
        switch (constantInstruction.opcode)
        {
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
                // These instructions may throw exceptions.
                mayThrowExceptions = true;

//          case InstructionConstants.OP_INVOKEVIRTUAL:
//          case InstructionConstants.OP_INVOKESPECIAL:
//          case InstructionConstants.OP_INVOKESTATIC:
//          case InstructionConstants.OP_INVOKEINTERFACE:
//            // Check if the invoking the method may throw an exception.
//            clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
        }
    }


//    // Implementations for ConstantVisitor.
//
//    public void visitAnyMethodrefConstant(Clazz clazz, RefConstant refConstant)
//    {
//        Member referencedMember = refConstant.referencedMember;
//
//        // Do we have a reference to the method?
//        if (referencedMember == null)
//        {
//            // We'll have to assume invoking the unknown method may throw an
//            // an exception.
//            mayThrowExceptions = true;
//        }
//        else
//        {
//            // First check the referenced method itself.
//            refConstant.referencedMemberAccept(this);
//
//            // If the result isn't conclusive, check down the hierarchy.
//            if (!mayThrowExceptions)
//            {
//                Clazz  referencedClass  = refConstant.referencedClass;
//                Method referencedMethod = (Method)referencedMember;
//
//                // Check all other implementations of the method in the class
//                // hierarchy.
//                referencedClass.methodImplementationsAccept(referencedMethod,
//                                                            false,
//                                                            false,
//                                                            true,
//                                                            true,
//                                                            this);
//            }
//        }
//    }
//
//
//    // Implementations for MemberVisitor.
//
//    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
//    {
//        mayThrowExceptions = mayThrowExceptions ||
//                             ExceptionMethodMarker.mayThrowExceptions(programMethod);
//    }
//
//
//    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
//    {
//        mayThrowExceptions = mayThrowExceptions ||
//                             !NoExceptionMethodMarker.doesntThrowExceptions(libraryMethod);
//    }
}
