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


/**
 * An <code>ExternalTypeEnumeration</code> provides an enumeration of all
 * types listed in a given external descriptor string. The method name can
 * be retrieved separately.
 * <p>
 * A <code>ExternalTypeEnumeration</code> object can be reused for processing
 * different subsequent descriptors, by means of the <code>setDescriptor</code>
 * method.
 *
 * @author Eric Lafortune
 */
public class ExternalTypeEnumeration
{
    private String descriptor;
    private int    index;


    public ExternalTypeEnumeration(String descriptor)
    {
        setDescriptor(descriptor);
    }


    ExternalTypeEnumeration()
    {
    }


    void setDescriptor(String descriptor)
    {
        this.descriptor = descriptor;

        reset();
    }


    public void reset()
    {
        index = descriptor.indexOf(JavaConstants.METHOD_ARGUMENTS_OPEN) + 1;

        if (index < 1)
        {
            throw new IllegalArgumentException("Missing opening parenthesis in descriptor ["+descriptor+"]");
        }
    }


    public boolean hasMoreTypes()
    {
        return index < descriptor.length() - 1;
    }


    public String nextType()
    {
        int startIndex = index;

        // Find the next separating comma.
        index = descriptor.indexOf(JavaConstants.METHOD_ARGUMENTS_SEPARATOR,
                                   startIndex);

        // Otherwise find the closing parenthesis.
        if (index < 0)
        {
            index = descriptor.indexOf(JavaConstants.METHOD_ARGUMENTS_CLOSE,
                                       startIndex);
            if (index < 0)
            {
                throw new IllegalArgumentException("Missing closing parenthesis in descriptor ["+descriptor+"]");
            }
        }

        return descriptor.substring(startIndex, index++).trim();
    }


    public String methodName()
    {
        return descriptor.substring(0, descriptor.indexOf(JavaConstants.METHOD_ARGUMENTS_OPEN)).trim();
    }
}
