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
package proguard.classfile.util;

import proguard.classfile.*;

import java.util.Stack;

/**
 * A <code>DescriptorClassEnumeration</code> provides an enumeration of all
 * classes mentioned in a given descriptor or signature.
 *
 * @author Eric Lafortune
 */
public class DescriptorClassEnumeration
{
    private String  descriptor;

    private int     index;
    private int     nestingLevel;
    private boolean isInnerClassName;
    private String  accumulatedClassName;
    private Stack   accumulatedClassNames;


    /**
     * Creates a new DescriptorClassEnumeration for the given descriptor.
     */
    public DescriptorClassEnumeration(String descriptor)
    {
        this.descriptor = descriptor;
    }


    /**
     * Returns the number of classes contained in the descriptor. This
     * is the number of class names that the enumeration will return.
     */
    public int classCount()
    {
        int count = 0;

        reset();

        nextFluff();
        while (hasMoreClassNames())
        {
            count++;

            nextClassName();
            nextFluff();
        }

        reset();

        return count;
    }


    /**
     * Resets the enumeration.
     */
    private void reset()
    {
        index                 = 0;
        nestingLevel          = 0;
        isInnerClassName      = false;
        accumulatedClassName  = null;
        accumulatedClassNames = null;
    }


    /**
     * Returns whether the enumeration can provide more class names from the
     * descriptor.
     */
    public boolean hasMoreClassNames()
    {
        return index < descriptor.length();
    }


    /**
     * Returns the next fluff (surrounding class names) from the descriptor.
     */
    public String nextFluff()
    {
        int fluffStartIndex = index;

        // Find the first token marking the start of a class name 'L' or '.'.
        loop: while (index < descriptor.length())
        {
            switch (descriptor.charAt(index++))
            {
                case ClassConstants.TYPE_GENERIC_START:
                {
                    nestingLevel++;

                    // Make sure we have a stack.
                    if (accumulatedClassNames == null)
                    {
                        accumulatedClassNames = new Stack();
                    }

                    // Remember the accumulated class name.
                    accumulatedClassNames.push(accumulatedClassName);

                    break;
                }
                case ClassConstants.TYPE_GENERIC_END:
                {
                    nestingLevel--;

                    // Return to the accumulated class name outside the
                    // generic block.
                    accumulatedClassName = (String)accumulatedClassNames.pop();

                    continue loop;
                }
                case ClassConstants.TYPE_GENERIC_BOUND:
                case ClassConstants.TYPE_ARRAY:
                {
                    continue loop;
                }
                case ClassConstants.TYPE_CLASS_START:
                {
                    // We've found the start of an ordinary class name.
                    nestingLevel += 2;
                    isInnerClassName = false;
                    break loop;
                }
                case ClassConstants.TYPE_CLASS_END:
                {
                    nestingLevel -= 2;
                    break;
                }
                case JavaConstants.INNER_CLASS_SEPARATOR:
                {
                    // We've found the start of an inner class name in a signature.
                    isInnerClassName = true;
                    break loop;
                }
                case ClassConstants.TYPE_GENERIC_VARIABLE_START:
                {
                    // We've found the start of a type identifier. Skip to the end.
                    while (descriptor.charAt(index++) != ClassConstants.TYPE_CLASS_END);
                    break;
                }
            }

            if (nestingLevel == 1 &&
                descriptor.charAt(index) != ClassConstants.TYPE_GENERIC_END)
            {
                // We're at the start of a type parameter. Skip to the start
                // of the bounds.
                while (descriptor.charAt(index++) != ClassConstants.TYPE_GENERIC_BOUND);
            }
        }

        return descriptor.substring(fluffStartIndex, index);
    }


    /**
     * Returns the next class name from the descriptor.
     */
    public String nextClassName()
    {
        int classNameStartIndex = index;

        // Find the first token marking the end of a class name '<' or ';'.
        loop: while (true)
        {
            switch (descriptor.charAt(index))
            {
                case ClassConstants.TYPE_GENERIC_START:
                case ClassConstants.TYPE_CLASS_END:
                case JavaConstants.INNER_CLASS_SEPARATOR:
                {
                    break loop;
                }
            }

            index++;
        }

        String className = descriptor.substring(classNameStartIndex, index);

        // Recompose the inner class name if necessary.
        accumulatedClassName = isInnerClassName ?
            accumulatedClassName + ClassConstants.INNER_CLASS_SEPARATOR + className :
            className;

        return accumulatedClassName;
    }


    /**
     * Returns whether the most recently returned class name was a recomposed
     * inner class name from a signature.
     */
    public boolean isInnerClassName()
    {
        return isInnerClassName;
    }


    /**
     * A main method for testing the class name enumeration.
     */
    public static void main(String[] args)
    {
        try
        {
            for (int index = 0; index < args.length; index++)
            {
                String descriptor = args[index];

                System.out.println("Descriptor ["+descriptor+"]");
                DescriptorClassEnumeration enumeration = new DescriptorClassEnumeration(descriptor);
                System.out.println("  Fluff: ["+enumeration.nextFluff()+"]");
                while (enumeration.hasMoreClassNames())
                {
                    System.out.println("  Name:  ["+enumeration.nextClassName()+"]");
                    System.out.println("  Fluff: ["+enumeration.nextFluff()+"]");
                }
            }
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
