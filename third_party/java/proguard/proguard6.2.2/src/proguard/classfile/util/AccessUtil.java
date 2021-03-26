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

import proguard.classfile.ClassConstants;


/**
 * Utility methods for working with access flags. For convenience, this class
 * defines access levels, in ascending order: <code>PRIVATE</code>,
 * <code>PACKAGE_VISIBLE</code>, <code>PROTECTED</code>, and <code>PUBLIC</code>.
 *
 * @author Eric Lafortune
 */
public class AccessUtil
{
    public static final int PRIVATE         = 0;
    public static final int PACKAGE_VISIBLE = 1;
    public static final int PROTECTED       = 2;
    public static final int PUBLIC          = 3;


    // The mask of access flags.
    private static final int ACCESS_MASK =
        ClassConstants.ACC_PUBLIC  |
        ClassConstants.ACC_PRIVATE |
        ClassConstants.ACC_PROTECTED;


    /**
     * Returns the corresponding access level of the given access flags.
     * @param accessFlags the internal access flags.
     * @return the corresponding access level: <code>PRIVATE</code>,
     *         <code>PACKAGE_VISIBLE</code>, <code>PROTECTED</code>, or
     *         <code>PUBLIC</code>.
     */
    public static int accessLevel(int accessFlags)
    {
        switch (accessFlags & ACCESS_MASK)
        {
            case ClassConstants.ACC_PRIVATE:   return PRIVATE;
            default:                           return PACKAGE_VISIBLE;
            case ClassConstants.ACC_PROTECTED: return PROTECTED;
            case ClassConstants.ACC_PUBLIC:    return PUBLIC;
        }
    }


    /**
     * Returns the corresponding access flags of the given access level.
     * @param accessLevel the access level: <code>PRIVATE</code>,
     *                    <code>PACKAGE_VISIBLE</code>, <code>PROTECTED</code>,
     *                    or <code>PUBLIC</code>.
     * @return the corresponding internal access flags,  the internal access
     *         flags as a logical bit mask of <code>INTERNAL_ACC_PRIVATE</code>,
     *         <code>INTERNAL_ACC_PROTECTED</code>, and
     *         <code>INTERNAL_ACC_PUBLIC</code>.
     */
    public static int accessFlags(int accessLevel)
    {
        switch (accessLevel)
        {
            case PRIVATE:   return ClassConstants.ACC_PRIVATE;
            default:        return 0;
            case PROTECTED: return ClassConstants.ACC_PROTECTED;
            case PUBLIC:    return ClassConstants.ACC_PUBLIC;
        }
    }


    /**
     * Replaces the access part of the given access flags.
     * @param accessFlags    the internal access flags.
     * @param newAccessFlags the new internal access flags.
     */
    public static int replaceAccessFlags(int accessFlags, int newAccessFlags)
    {
        // A private class member should not be explicitly final.
        if (newAccessFlags == ClassConstants.ACC_PRIVATE)
        {
            accessFlags &= ~ClassConstants.ACC_FINAL;
        }

        return (accessFlags    & ~ACCESS_MASK) |
               (newAccessFlags &  ACCESS_MASK);
    }
}
