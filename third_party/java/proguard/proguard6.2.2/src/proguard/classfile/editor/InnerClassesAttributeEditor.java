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

import proguard.classfile.attribute.*;
import proguard.util.ArrayUtil;

/**
 * This class can add/remove bootstrap methods to/from a given inner classes
 * attribute. Inner classes to be added must have been filled out beforehand.
 *
 * @author Thomas Neidhart
 */
public class InnerClassesAttributeEditor
{
    private InnerClassesAttribute targetInnerClassesAttribute;


    /**
     * Creates a new InnerClassesAttributeEditor that will edit inner
     * classes in the given inner classes attribute.
     */
    public InnerClassesAttributeEditor(InnerClassesAttribute targetInnerClassesAttribute)
    {
        this.targetInnerClassesAttribute = targetInnerClassesAttribute;
    }


    /**
     * Adds a given inner class to the inner classes attribute.
     * @return the index of the inner class.
     */
    public int addInnerClassesInfo(InnerClassesInfo innerClassesInfo)
    {
        targetInnerClassesAttribute.classes =
            ArrayUtil.add(targetInnerClassesAttribute.classes,
                          targetInnerClassesAttribute.u2classesCount,
                          innerClassesInfo);

        return targetInnerClassesAttribute.u2classesCount++;
    }


    /**
     * Removes the given inner class from the inner classes attribute.
     */
    public void removeInnerClassesInfo(InnerClassesInfo innerClassesInfo)
    {
        ArrayUtil.remove(targetInnerClassesAttribute.classes,
                         targetInnerClassesAttribute.u2classesCount--,
                         findInnerClassesInfoIndex(innerClassesInfo));
    }


    /**
     * Finds the index of the given bootstrap method info in the target attribute.
     */
    private int findInnerClassesInfoIndex(InnerClassesInfo innerClassesInfo)
    {
        int                innerClassesCount = targetInnerClassesAttribute.u2classesCount;
        InnerClassesInfo[] innerClassesInfos = targetInnerClassesAttribute.classes;

        for (int index = 0; index < innerClassesCount; index++)
        {
            if (innerClassesInfos[index].equals(innerClassesInfo))
            {
                return index;
            }
        }

        return innerClassesCount;
    }

}