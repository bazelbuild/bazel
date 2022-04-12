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
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

import java.util.Arrays;


/**
 * This ClassVisitor removes NameAndType constant pool entries that are not
 * used.
 *
 * @author Eric Lafortune
 */
public class NameAndTypeShrinker
extends      SimplifiedVisitor
implements   ClassVisitor,
             ConstantVisitor,
             AttributeVisitor
{
    // A visitor info flag to indicate the NameAndType constant pool entry is being used.
    private static final Object USED = new Object();

    private       int[]                constantIndexMap;
    private final ConstantPoolRemapper constantPoolRemapper = new ConstantPoolRemapper();


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Mark the NameAndType entries referenced by all other constant pool
        // entries.
        programClass.constantPoolEntriesAccept(this);

        // Mark the NameAndType entries referenced by all EnclosingMethod
        // attributes.
        programClass.attributesAccept(this);

        // Shift the used constant pool entries together, filling out the
        // index map.
        int newConstantPoolCount =
            shrinkConstantPool(programClass.constantPool,
                               programClass.u2constantPoolCount);

        // Remap the references to the constant pool if it has shrunk.
        if (newConstantPoolCount < programClass.u2constantPoolCount)
        {
            programClass.u2constantPoolCount = newConstantPoolCount;

            // Remap all constant pool references.
            constantPoolRemapper.setConstantIndexMap(constantIndexMap);
            constantPoolRemapper.visitProgramClass(programClass);
        }
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}


    public void visitInvokeDynamicConstant(Clazz clazz, InvokeDynamicConstant invokeDynamicConstant)
    {
        markNameAndTypeConstant(clazz, invokeDynamicConstant.u2nameAndTypeIndex);
    }


    public void visitAnyRefConstant(Clazz clazz, RefConstant refConstant)
    {
        markNameAndTypeConstant(clazz, refConstant.u2nameAndTypeIndex);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        if (enclosingMethodAttribute.u2nameAndTypeIndex != 0)
        {
            markNameAndTypeConstant(clazz, enclosingMethodAttribute.u2nameAndTypeIndex);
        }
    }


    // Small utility methods.

    /**
     * Marks the given UTF-8 constant pool entry of the given class.
     */
    private void markNameAndTypeConstant(Clazz clazz, int index)
    {
         markAsUsed((NameAndTypeConstant)((ProgramClass)clazz).getConstant(index));
    }


    /**
     * Marks the given VisitorAccepter as being used.
     * In this context, the VisitorAccepter will be a NameAndTypeConstant object.
     */
    private void markAsUsed(VisitorAccepter visitorAccepter)
    {
        visitorAccepter.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given VisitorAccepter has been marked as being used.
     * In this context, the VisitorAccepter will be a NameAndTypeConstant object.
     */
    private boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }


    /**
     * Removes all NameAndType entries that are not marked as being used
     * from the given constant pool.
     * @return the new number of entries.
     */
    private int shrinkConstantPool(Constant[] constantPool, int length)
    {
        // Create a new index map, if necessary.
        if (constantIndexMap == null ||
            constantIndexMap.length < length)
        {
            constantIndexMap = new int[length];
        }

        int     counter = 1;
        boolean isUsed  = false;

        // Shift the used constant pool entries together.
        for (int index = 1; index < length; index++)
        {
            Constant constant = constantPool[index];

            // Is the constant being used? Don't update the flag if this is the
            // second half of a long entry.
            if (constant != null)
            {
                isUsed = constant.getTag() != ClassConstants.CONSTANT_NameAndType ||
                         isUsed(constant);
            }

            if (isUsed)
            {
                // Remember the new index.
                constantIndexMap[index] = counter;

                // Shift the constant pool entry.
                constantPool[counter++] = constant;
            }
            else
            {
                // Remember an invalid index.
                constantIndexMap[index] = -1;
            }
        }

        // Clear the remaining constant pool elements.
        Arrays.fill(constantPool, counter, length, null);

        return counter;
    }
}
