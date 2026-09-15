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

import proguard.classfile.ProgramClass;
import proguard.util.ArrayUtil;

/**
 * This class can add and delete interfaces to and from classes. References to
 * the constant pool must be filled out beforehand.
 *
 * @author Eric Lafortune
 */
public class InterfacesEditor
{
    private final ProgramClass targetClass;


    /**
     * Creates a new InterfacesEditor that will edit interfaces in the given
     * target class.
     */
    public InterfacesEditor(ProgramClass targetClass)
    {
        this.targetClass = targetClass;
    }


    /**
     * Adds the specified interface to the target class, if it isn't present yet.
     */
    public void addInterface(int interfaceConstantIndex)
    {
        // Is the interface not yet present?
        if (findInterfaceIndex(interfaceConstantIndex) < 0)
        {
            // Append the interface.
            targetClass.u2interfaces =
                ArrayUtil.add(targetClass.u2interfaces,
                              targetClass.u2interfacesCount++,
                              interfaceConstantIndex);
        }
    }


    /**
     * Deletes the given interface from the target class, if it is present.
     */
    public void deleteInterface(int interfaceConstantIndex)
    {
        // Is the interface already present?
        int interfaceIndex = findInterfaceIndex(interfaceConstantIndex);
        if (interfaceIndex >= 0)
        {
            int   interfacesCount = --targetClass.u2interfacesCount;
            int[] interfaces      = targetClass.u2interfaces;

            // Shift the other interfaces in the array.
            for (int index = interfaceIndex; index < interfacesCount; index++)
            {
                interfaces[index] = interfaces[index + 1];
            }

            // Clear the remaining entry in the array.
            interfaces[interfacesCount] = 0;
        }
    }


    // Small utility methods.

    /**
     * Finds the index of the specified interface in the list of interfaces of
     * the target class.
     */
    private int findInterfaceIndex(int interfaceConstantIndex)
    {
        int   interfacesCount = targetClass.u2interfacesCount;
        int[] interfaces      = targetClass.u2interfaces;

        for (int index = 0; index < interfacesCount; index++)
        {
            if (interfaces[index] == interfaceConstantIndex)
            {
                return index;
            }
        }

        return -1;
    }
}