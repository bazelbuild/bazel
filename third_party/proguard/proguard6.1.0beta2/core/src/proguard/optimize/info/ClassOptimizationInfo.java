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

/**
 * This class stores some optimization information that can be attached to
 * a class.
 *
 * @author Eric Lafortune
 */
public class ClassOptimizationInfo
{
    protected boolean hasNoSideEffects = false;


    public void setNoSideEffects()
    {
        hasNoSideEffects = true;
    }


    public boolean hasNoSideEffects()
    {
        return hasNoSideEffects;
    }


    public boolean isKept()
    {
        return true;
    }


    public boolean isInstantiated()
    {
        return true;
    }


    public boolean isInstanceofed()
    {
        // We're relaxing the strict assumption of "true".
        return !hasNoSideEffects;
    }


    public boolean isDotClassed()
    {
        // We're relaxing the strict assumption of "true".
        return !hasNoSideEffects;
    }


    public boolean isCaught()
    {
        return true;
    }


    public boolean isSimpleEnum()
    {
        return false;
    }


    public boolean isWrapper()
    {
        return false;
    }


    public boolean isEscaping()
    {
        return true;
    }


    public boolean hasSideEffects()
    {
        return !hasNoSideEffects;
    }


    public boolean containsPackageVisibleMembers()
    {
        return true;
    }


    public boolean invokesPackageVisibleMembers()
    {
        return true;
    }


    public boolean mayBeMerged()
    {
        return false;
    }


    public Clazz getWrappedClass()
    {
        return null;
    }


    public Clazz getTargetClass()
    {
        return null;
    }


    public static void setClassOptimizationInfo(Clazz clazz)
    {
        clazz.setVisitorInfo(new ClassOptimizationInfo());
    }


    public static ClassOptimizationInfo getClassOptimizationInfo(Clazz clazz)
    {
        return (ClassOptimizationInfo)clazz.getVisitorInfo();
    }
}
