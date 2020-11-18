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

/**
 * This class can add and delete attributes to and from classes, fields,
 * methods, and code attributes. Attributes to be added must be filled out
 * beforehand, including their references to the constant pool. Existing
 * attributes of the same type are always replaced.
 *
 * @author Eric Lafortune
 */
public class AttributesEditor
{
    private final ProgramClass  targetClass;
    private final ProgramMember targetMember;
    private final CodeAttribute targetAttribute;
    private final boolean       replaceAttributes;


    /**
     * Creates a new AttributeAdder that will edit attributes in the given
     * target class.
     */
    public AttributesEditor(ProgramClass targetClass,
                            boolean      replaceAttributes)
    {
        this(targetClass, null, null, replaceAttributes);
    }


    /**
     * Creates a new AttributeAdder that will edit attributes in the given
     * target class member.
     */
    public AttributesEditor(ProgramClass  targetClass,
                            ProgramMember targetMember,
                            boolean       replaceAttributes)
    {
        this(targetClass, targetMember, null, replaceAttributes);
    }


    /**
     * Creates a new AttributeAdder that will edit attributes in the given
     * target code attribute.
     */
    public AttributesEditor(ProgramClass  targetClass,
                            ProgramMember targetMember,
                            CodeAttribute targetAttribute,
                            boolean       replaceAttributes)
    {
        this.targetClass       = targetClass;
        this.targetMember      = targetMember;
        this.targetAttribute   = targetAttribute;
        this.replaceAttributes = replaceAttributes;
    }


    /**
     * Finds the specified attribute in the target.
     */
    public Attribute findAttribute(String attributeName)
    {
        // What's the target?
        return
            targetAttribute != null ?
                findAttribute(targetAttribute.u2attributesCount,
                              targetAttribute.attributes,
                              attributeName) :
            targetMember != null ?
                findAttribute(targetMember.u2attributesCount,
                              targetMember.attributes,
                              attributeName) :
                findAttribute(targetClass.u2attributesCount,
                              targetClass.attributes,
                              attributeName);
    }


    /**
     * Adds the given attribute to the target.
     */
    public void addAttribute(Attribute attribute)
    {
        // What's the target?
        if (targetAttribute != null)
        {
            // Try to replace an existing attribute.
            if (!replaceAttributes ||
                !replaceAttribute(targetAttribute.u2attributesCount,
                                  targetAttribute.attributes,
                                  attribute))
            {
                // Otherwise append the attribute.
                targetAttribute.attributes =
                    addAttribute(targetAttribute.u2attributesCount,
                                 targetAttribute.attributes,
                                 attribute);

                targetAttribute.u2attributesCount++;
            }
        }
        else if (targetMember != null)
        {
            // Try to replace an existing attribute.
            if (!replaceAttributes ||
                !replaceAttribute(targetMember.u2attributesCount,
                                  targetMember.attributes,
                                  attribute))
            {
                // Otherwise append the attribute.
                targetMember.attributes =
                    addAttribute(targetMember.u2attributesCount,
                                 targetMember.attributes,
                                 attribute);

                targetMember.u2attributesCount++;
            }
        }
        else
        {
            // Try to replace an existing attribute.
            if (!replaceAttributes ||
                !replaceAttribute(targetClass.u2attributesCount,
                                  targetClass.attributes,
                                  attribute))
            {
                // Otherwise append the attribute.
                targetClass.attributes =
                    addAttribute(targetClass.u2attributesCount,
                                 targetClass.attributes,
                                 attribute);

                targetClass.u2attributesCount++;
            }
        }
    }


    /**
     * Deletes the specified attribute from the target.
     */
    public void deleteAttribute(String attributeName)
    {
        // What's the target?
        if (targetAttribute != null)
        {
            targetAttribute.u2attributesCount =
                deleteAttribute(targetAttribute.u2attributesCount,
                                targetAttribute.attributes,
                                attributeName);
        }
        else if (targetMember != null)
        {
            targetMember.u2attributesCount =
                deleteAttribute(targetMember.u2attributesCount,
                                targetMember.attributes,
                                attributeName);
        }
        else
        {
            targetClass.u2attributesCount =
                deleteAttribute(targetClass.u2attributesCount,
                                targetClass.attributes,
                                attributeName);
        }
    }


    // Small utility methods.

    /**
     * Tries to put the given attribute in place of an existing attribute of
     * the same name, returning whether it was present.
     */
    private boolean replaceAttribute(int         attributesCount,
                                     Attribute[] attributes,
                                     Attribute   attribute)
    {
        // Find the attribute with the same name.
        int index = findAttributeIndex(attributesCount,
                                       attributes,
                                       attribute.getAttributeName(targetClass));
        if (index < 0)
        {
            return false;
        }

        attributes[index] = attribute;

        return true;
    }


    /**
     * Appends the given attribute to the given array of attributes, creating a
     * new array if necessary.
     */
    private Attribute[] addAttribute(int         attributesCount,
                                     Attribute[] attributes,
                                     Attribute   attribute)
    {
        // Is the array too small to contain the additional attribute?
        if (attributes.length <= attributesCount)
        {
            // Create a new array and copy the attributes into it.
            Attribute[] newAttributes = new Attribute[attributesCount + 1];
            System.arraycopy(attributes, 0,
                             newAttributes, 0,
                             attributesCount);
            attributes = newAttributes;
        }

        // Append the attribute.
        attributes[attributesCount] = attribute;

        return attributes;
    }


    /**
     * Deletes the attributes with the given name from the given array of
     * attributes, returning the new number of attributes.
     */
    private int deleteAttribute(int         attributesCount,
                                Attribute[] attributes,
                                String      attributeName)
    {
        // Find the attribute.
        int index = findAttributeIndex(attributesCount,
                                       attributes,
                                       attributeName);
        if (index < 0)
        {
            return attributesCount;
        }

        // Shift the other attributes in the array.
        System.arraycopy(attributes, index + 1,
                         attributes, index,
                         attributesCount - index - 1);

        // Clear the last entry in the array.
        attributes[--attributesCount] = null;

        return attributesCount;
    }


    /**
     * Finds the index of the attribute with the given name in the given
     * array of attributes.
     */
    private int findAttributeIndex(int         attributesCount,
                                   Attribute[] attributes,
                                   String      attributeName)
    {
        for (int index = 0; index < attributesCount; index++)
        {
            Attribute attribute = attributes[index];

            if (attribute.getAttributeName(targetClass).equals(attributeName))
            {
                return index;
            }
        }

        return -1;
    }


    /**
     * Finds the attribute with the given name in the given
     * array of attributes.
     */
    private Attribute findAttribute(int         attributesCount,
                                    Attribute[] attributes,
                                    String      attributeName)
    {
        for (int index = 0; index < attributesCount; index++)
        {
            Attribute attribute = attributes[index];

            if (attribute.getAttributeName(targetClass).equals(attributeName))
            {
                return attribute;
            }
        }

        return null;
    }
}
