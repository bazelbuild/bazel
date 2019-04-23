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
 * This class contains utility methods operating on objects.
 */
public class ObjectUtil
{
    /**
     * Returns whether the given objects are the same.
     * @param object1 the first object, may be null.
     * @param object2 the second object, may be null.
     * @return whether the objects are the same.
     */
    public static boolean equal(Object object1, Object object2)
    {
        return object1 == null ?
            object2 == null :
            object1.equals(object2);
    }


    /**
     * Returns the hash code of the given object, or 0 if it is null.
     * @param object the object, may be null.
     * @return the hash code.
     */
    public static int hashCode(Object object)
    {
        return object == null ? 0 : object.hashCode();
    }


    /**
     * Returns a comparison of the two given objects.
     * @param object1 the first object, may be null.
     * @param object2 the second object, may be null.
     * @return -1, 0, or 1.
     * @see Comparable#compareTo(Object)
     */
    public static int compare(Comparable object1, Comparable object2)
    {
        return object1 == null ?
            object2 == null ? 0 : -1 :
            object2 == null ? 1 : object1.compareTo(object2);
    }
}
