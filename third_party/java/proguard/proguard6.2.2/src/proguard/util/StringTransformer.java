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
package proguard.util;

/**
 * Provides a method for transforming a string into another string.
 *
 * @author Johan Leys
 */
public interface StringTransformer
{
    /**
     * A StringTransformer that doesn't do any transformation.
     */
    StringTransformer IDENTITY_TRANSFORMER = new StringTransformer()
    {
        public String transform(String string)
        {
            return string;
        }
    };


    /**
     * Transforms the given string and returns the transformed result.
     *
     * @param string a string to be transformed.
     * @return the transformed string.
     */
    String transform(String string);
}
