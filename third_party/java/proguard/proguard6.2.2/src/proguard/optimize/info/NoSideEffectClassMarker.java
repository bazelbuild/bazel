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

package proguard.optimize.info;

import proguard.classfile.Clazz;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;

/**
 * This ClassVisitor marks all classes that it visits as not having any side
 * effects. It will make the SideEffectClassMarker consider them as such
 * without further analysis.
 *
 * @see SideEffectMethodMarker
 * @author Eric Lafortune
 */
public class NoSideEffectClassMarker
extends      SimplifiedVisitor
implements   ClassVisitor
{
    // Implementations for MemberVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        markNoSideEffects(clazz);
    }


    // Small utility methods.

    private static void markNoSideEffects(Clazz Clazz)
    {
        ClassOptimizationInfo.getClassOptimizationInfo(Clazz).setNoSideEffects();
    }


    public static boolean hasNoSideEffects(Clazz Clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(Clazz).hasNoSideEffects();
    }
}
