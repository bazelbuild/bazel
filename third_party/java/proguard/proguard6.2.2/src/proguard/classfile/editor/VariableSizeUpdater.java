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
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;

/**
 * This AttributeVisitor computes and updates the maximum local variable frame
 * size of the code attributes that it visits. It also cleans up the local
 * variable tables.
 *
 * @author Eric Lafortune
 */
public class VariableSizeUpdater
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/


    private VariableCleaner variableCleaner = new VariableCleaner();


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // The minimum variable size is determined by the arguments.
        codeAttribute.u2maxLocals =
            ClassUtil.internalMethodParameterSize(method.getDescriptor(clazz),
                                                  method.getAccessFlags());

        if (DEBUG)
        {
            System.out.println("VariableSizeUpdater: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
            System.out.println("  Max locals: "+codeAttribute.u2maxLocals+" <- parameters");
        }

        // Go over all instructions.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Remove the unused variables of the attributes.
        variableCleaner.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    public void visitVariableInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, VariableInstruction variableInstruction)
    {
        int variableSize = variableInstruction.variableIndex + 1;
        if (variableInstruction.isCategory2())
        {
            variableSize++;
        }

        if (codeAttribute.u2maxLocals < variableSize)
        {
            codeAttribute.u2maxLocals = variableSize;

            if (DEBUG)
            {
                System.out.println("  Max locals: "+codeAttribute.u2maxLocals+" <- "+variableInstruction.toString(offset));
            }
        }
    }
}
