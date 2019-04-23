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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.ClassConstant;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

import java.util.Arrays;

/**
 * This ClassVisitor removes InnerClasses and EnclosingMethod attributes in
 * classes that are retargeted or that refer to classes that are retargeted.
 *
 * @see ClassMerger
 * @author Eric Lafortune
 */
public class RetargetedInnerClassAttributeRemover
extends      SimplifiedVisitor
implements   ClassVisitor,
             AttributeVisitor,
             InnerClassesInfoVisitor,
             ConstantVisitor
{
    private boolean retargeted;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        int         attributesCount = programClass.u2attributesCount;
        Attribute[] attributes      = programClass.attributes;

        int newAtributesCount = 0;

        // Copy over all non-retargeted attributes.
        for (int index = 0; index < attributesCount; index++)
        {
            Attribute attribute = attributes[index];

            // Check if it's an InnerClasses or EnclosingMethod attribute in
            // a retargeted class or referring to a retargeted class.
            retargeted = false;
            attribute.accept(programClass, this);
            if (!retargeted)
            {
                attributes[newAtributesCount++] = attribute;
            }
        }

        // Clean up any remaining array elements.
        Arrays.fill(attributes, newAtributesCount, attributesCount, null);

        // Update the number of attributes.
        programClass.u2attributesCount = newAtributesCount;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
    {
        // Check whether the class itself is retargeted.
        checkTarget(clazz);

        if (!retargeted)
        {
            // Check whether the referenced classes are retargeted.
            innerClassesAttribute.innerClassEntriesAccept(clazz, this);
            int                classesCount = innerClassesAttribute.u2classesCount;
            InnerClassesInfo[] classes      = innerClassesAttribute.classes;

            int newClassesCount = 0;

            // Copy over all non-retargeted attributes.
            for (int index = 0; index < classesCount; index++)
            {
                InnerClassesInfo classInfo = classes[index];

                // Check if the outer class or inner class is a retargeted class.
                retargeted = false;
                classInfo.outerClassConstantAccept(clazz, this);
                classInfo.innerClassConstantAccept(clazz, this);
                if (!retargeted)
                {
                    classes[newClassesCount++] = classInfo;
                }
            }

            // Clean up any remaining array elements.
            Arrays.fill(classes, newClassesCount, classesCount, null);

            // Update the number of classes.
            innerClassesAttribute.u2classesCount = newClassesCount;

            // Remove the attribute altogether if it's empty.
            retargeted = newClassesCount == 0;
        }
    }


    public void visitEnclosingMethodAttribute(Clazz clazz, EnclosingMethodAttribute enclosingMethodAttribute)
    {
        // Check whether the class itself is retargeted.
        checkTarget(clazz);

        // Check whether the referenced class is retargeted.
        checkTarget(enclosingMethodAttribute.referencedClass);
    }


    // Implementations for InnerClassesInfoVisitor.

    public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
    {
        // Check whether the inner class or the outer class are retargeted.
        innerClassesInfo.innerClassConstantAccept(clazz, this);
        innerClassesInfo.outerClassConstantAccept(clazz, this);
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Check whether the referenced class is retargeted.
        checkTarget(classConstant.referencedClass);
    }


    // Small utility methods.

    /**
     * Sets the global return value to true if the given class is retargeted.
     */
    private void checkTarget(Clazz clazz)
    {
        if (clazz != null &&
            ClassMerger.getTargetClass(clazz) != null)
        {
            retargeted = true;
        }
    }
}