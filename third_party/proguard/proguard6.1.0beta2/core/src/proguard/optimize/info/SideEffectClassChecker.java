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

import proguard.classfile.*;
import proguard.classfile.visitor.ClassCollector;

import java.util.*;

/**
 * This utility class contains methods to check whether referencing classes
 * may have side effects due to them being loaded and initialized.
 *
 * @see NoSideEffectClassMarker
 * @see SideEffectClassMarker
 * @author Eric Lafortune
 */
public class SideEffectClassChecker
{
    /**
     * Returns whether accessing the given class member from the given class may
     * have side effects when they are initialized.
     */
    public static boolean mayHaveSideEffects(Clazz  referencingClass,
                                             Clazz  referencedClass,
                                             Member referencedMember)
    {
        // Is the referenced class member static or an initializer method?
        // Does accessing the referenced class then have side effects?
        return
            ((referencedMember.getAccessFlags() & ClassConstants.ACC_STATIC) != 0 ||
             referencedMember.getName(referencedClass).equals(ClassConstants.METHOD_NAME_INIT)) &&
            mayHaveSideEffects(referencingClass, referencedClass);
    }


    /**
     * Returns whether accessing the given class from another given class may
     * have side effects when they are initialized.
     */
    public static boolean mayHaveSideEffects(Clazz referencingClass,
                                             Clazz referencedClass)
    {
        return
            !NoSideEffectClassMarker.hasNoSideEffects(referencedClass) &&
            !referencingClass.extendsOrImplements(referencedClass)     &&
            !sideEffectSuperClasses(referencingClass).containsAll(sideEffectSuperClasses(referencedClass));
    }


    /**
     * Returns the set of superclasses and interfaces that are initialized.
     */
    private static Set sideEffectSuperClasses(Clazz clazz)
    {
        Set set = new HashSet();

        // Visit all superclasses and interfaces, collecting the ones that have
        // side effects when they are initialized.
        clazz.hierarchyAccept(true, true, true, false,
                              new SideEffectClassFilter(
                              new ClassCollector(set)));

        return set;
    }
}