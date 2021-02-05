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
import proguard.classfile.attribute.annotation.*;

/**
 * This class can add and delete element values to and from a given target
 * annotation default attribute, annotation, or array element value. Element
 * values to be added must be filled out beforehand, including their references
 * to the constant pool.
 *
 * @author Eric Lafortune
 */
public class ElementValuesEditor
{
    private final ProgramClass      targetClass;
    private final Annotation        targetAnnotation;
    private final ArrayElementValue targetArrayElementValue;
    private final boolean           replaceElementValues;


    /**
     * Creates a new ElementValuesEditor that will edit element values in the
     * given target annotation.
     */
    public ElementValuesEditor(ProgramClass targetClass,
                               Annotation   targetAnnotation,
                               boolean      replaceElementValues)
    {
        this.targetClass             = targetClass;
        this.targetAnnotation        = targetAnnotation;
        this.targetArrayElementValue = null;
        this.replaceElementValues    = replaceElementValues;
    }


    /**
     * Creates a new ElementValuesEditor that will edit element values in the
     * given target array element value.
     */
    public ElementValuesEditor(ProgramClass      targetClass,
                               ArrayElementValue targetArrayElementValue,
                               boolean           replaceElementValues)
    {
        this.targetClass             = targetClass;
        this.targetAnnotation        = null;
        this.targetArrayElementValue = targetArrayElementValue;
        this.replaceElementValues    = replaceElementValues;
    }


    /**
     * Adds the given elementValue to the target.
     */
    public void addElementValue(ElementValue elementValue)
    {
        // What's the target?
        if (targetAnnotation != null)
        {
            // Try to replace an existing element value.
            if (!replaceElementValues ||
                !replaceElementValue(targetAnnotation.u2elementValuesCount,
                                     targetAnnotation.elementValues,
                                     elementValue))
            {
                // Otherwise append the element value.
                targetAnnotation.elementValues =
                    addElementValue(targetAnnotation.u2elementValuesCount,
                                    targetAnnotation.elementValues,
                                    elementValue);

                targetAnnotation.u2elementValuesCount++;
            }
        }
        else
        {
            // Try to replace an existing element value.
            if (!replaceElementValues ||
                !replaceElementValue(targetArrayElementValue.u2elementValuesCount,
                                     targetArrayElementValue.elementValues,
                                     elementValue))
            {
                // Otherwise append the element value.
                targetArrayElementValue.elementValues =
                    addElementValue(targetArrayElementValue.u2elementValuesCount,
                                    targetArrayElementValue.elementValues,
                                    elementValue);

                targetArrayElementValue.u2elementValuesCount++;
            }
        }
    }


    /**
     * Deletes the given elementValue to the target.
     */
    public void deleteElementValue(String elementValueMethodName)
    {
        // What's the target?
        if (targetAnnotation != null)
        {
            // Delete the element value to the target annotation.
            targetAnnotation.u2elementValuesCount =
                deleteElementValue(targetAnnotation.u2elementValuesCount,
                                   targetAnnotation.elementValues,
                                   elementValueMethodName);
        }
        else
        {
            // Delete the element value to the target array element value.
            targetArrayElementValue.u2elementValuesCount =
                deleteElementValue(targetArrayElementValue.u2elementValuesCount,
                                   targetArrayElementValue.elementValues,
                                   elementValueMethodName);
        }
    }


    // Small utility methods.

    /**
     * Tries put the given element value in place of an existing element value
     * of the same name, returning whether it was present.
     */
    private boolean replaceElementValue(int            elementValuesCount,
                                        ElementValue[] elementValues,
                                        ElementValue   elementValue)
    {
        // Find the element value with the same name.
        int index = findElementValue(elementValuesCount,
                                     elementValues,
                                     elementValue.getMethodName(targetClass));
        if (index < 0)
        {
            return false;
        }

        elementValues[index] = elementValue;

        return true;
    }


    /**
     * Appends the given element value to the given array of element values,
     * creating a new array if necessary.
     */
    private ElementValue[] addElementValue(int            elementValuesCount,
                                           ElementValue[] elementValues,
                                           ElementValue   elementValue)
    {
        // Is the array too small to contain the additional elementValue?
        if (elementValues.length <= elementValuesCount)
        {
            // Create a new array and copy the elementValues into it.
            ElementValue[] newElementValues = new ElementValue[elementValuesCount + 1];
            System.arraycopy(elementValues, 0,
                             newElementValues, 0,
                             elementValuesCount);
            elementValues = newElementValues;
        }

        // Append the elementValue.
        elementValues[elementValuesCount] = elementValue;

        return elementValues;
    }


    /**
     * Deletes the element values with the given name from the given array of
     * element values, returning the new number of element values.
     */
    private int deleteElementValue(int            elementValuesCount,
                                   ElementValue[] elementValues,
                                   String         elementValueMethodName)
    {
        // Find the element value.
        int index = findElementValue(elementValuesCount,
                                     elementValues,
                                     elementValueMethodName);
        if (index < 0)
        {
            return elementValuesCount;
        }

        // Shift the other element values in the array.
        System.arraycopy(elementValues, index + 1,
                         elementValues, index,
                         elementValuesCount - index - 1);

        // Clear the last entry in the array.
        elementValues[--elementValuesCount] = null;

        return elementValuesCount;
    }


    /**
     * Finds the index of the element value with the given name in the given
     * array of element values.
     */
    private int findElementValue(int            elementValuesCount,
                                 ElementValue[] elementValues,
                                 String         elementValueName)
    {
        for (int index = 0; index < elementValuesCount; index++)
        {
            if (elementValues[index].getMethodName(targetClass).equals(elementValueName))
            {
                return index;
            }
        }

        return -1;
    }
}