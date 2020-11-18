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
package proguard.classfile.attribute.annotation;

import proguard.classfile.*;
import proguard.classfile.attribute.annotation.visitor.ElementValueVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ElementValue represents a class element value.
 *
 * @author Eric Lafortune
 */
public class ClassElementValue extends ElementValue
{
    public int u2classInfoIndex;

    /**
     * An extra field pointing to the Clazz objects referenced in the
     * type name string. This field is filled out by the <code>{@link
     * proguard.classfile.util.ClassReferenceInitializer ClassReferenceInitializer}</code>.
     * References to primitive types are ignored.
     */
    public Clazz[] referencedClasses;


    /**
     * Creates an uninitialized ClassElementValue.
     */
    public ClassElementValue()
    {
    }


    /**
     * Creates an initialized ClassElementValue.
     */
    public ClassElementValue(int u2elementNameIndex,
                             int u2classInfoIndex)
    {
        super(u2elementNameIndex);

        this.u2classInfoIndex = u2classInfoIndex;
    }


    /**
     * Returns the class info name.
     */
    public String getClassName(Clazz clazz)
    {
        return clazz.getString(u2classInfoIndex);
    }


    /**
     * Applies the given visitor to all referenced classes.
     */
    public void referencedClassesAccept(ClassVisitor classVisitor)
    {
        if (referencedClasses != null)
        {
            for (int index = 0; index < referencedClasses.length; index++)
            {
                Clazz referencedClass = referencedClasses[index];
                if (referencedClass != null)
                {
                    referencedClass.accept(classVisitor);
                }
            }
        }
    }


    // Implementations for ElementValue.

    public char getTag()
    {
        return ClassConstants.ELEMENT_VALUE_CLASS;
    }

    public void accept(Clazz clazz, Annotation annotation, ElementValueVisitor elementValueVisitor)
    {
        elementValueVisitor.visitClassElementValue(clazz, annotation, this);
    }
}
