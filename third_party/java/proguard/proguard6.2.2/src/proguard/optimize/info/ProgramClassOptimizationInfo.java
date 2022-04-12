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
public class ProgramClassOptimizationInfo
extends      ClassOptimizationInfo
{
    private volatile boolean isInstantiated                = false;
    private volatile boolean isInstanceofed                = false;
    private volatile boolean isDotClassed                  = false;
    private volatile boolean isCaught                      = false;
    private volatile boolean isSimpleEnum                  = false;
    private volatile boolean isEscaping                    = false;
    private volatile boolean hasSideEffects                = false;
    private volatile boolean containsPackageVisibleMembers = false;
    private volatile boolean invokesPackageVisibleMembers  = false;
    private volatile boolean mayBeMerged                   = true;
    private volatile Clazz   wrappedClass;
    private volatile Clazz   targetClass;


    public boolean isKept()
    {
        return false;
    }


    public void setInstantiated()
    {
        isInstantiated = true;
    }


    public boolean isInstantiated()
    {
        return isInstantiated;
    }


    public void setInstanceofed()
    {
        isInstanceofed = true;
    }


    public boolean isInstanceofed()
    {
        return isInstanceofed;
    }


    public void setDotClassed()
    {
        isDotClassed = true;
    }


    public boolean isDotClassed()
    {
        return isDotClassed;
    }


    public void setCaught()
    {
        isCaught = true;
    }


    public boolean isCaught()
    {
        return isCaught;
    }


    public void setSimpleEnum(boolean simple)
    {
        isSimpleEnum = simple;
    }


    public boolean isSimpleEnum()
    {
        return isSimpleEnum;
    }


    public void setEscaping()
    {
        isEscaping = true;
    }


    public boolean isEscaping()
    {
        return isEscaping;
    }


    public void setSideEffects()
    {
        hasSideEffects = true;
    }


    public boolean hasSideEffects()
    {
        return !hasNoSideEffects && hasSideEffects;
    }


    public void setContainsPackageVisibleMembers()
    {
        containsPackageVisibleMembers = true;
    }


    public boolean containsPackageVisibleMembers()
    {
        return containsPackageVisibleMembers;
    }


    public void setInvokesPackageVisibleMembers()
    {
        invokesPackageVisibleMembers = true;
    }


    public boolean invokesPackageVisibleMembers()
    {
        return invokesPackageVisibleMembers;
    }


    public void setMayNotBeMerged()
    {
        mayBeMerged = false;
    }


    public boolean mayBeMerged()
    {
        return mayBeMerged;
    }


    public void setWrappedClass(Clazz wrappedClass)
    {
        this.wrappedClass = wrappedClass;
    }


    public Clazz getWrappedClass()
    {
        return wrappedClass;
    }


    public void setTargetClass(Clazz targetClass)
    {
        this.targetClass = targetClass;
    }


    public Clazz getTargetClass()
    {
        return targetClass;
    }


    public void merge(ClassOptimizationInfo other)
    {
        this.isInstantiated                |= other.isInstantiated();
        this.isInstanceofed                |= other.isInstanceofed();
        this.isDotClassed                  |= other.isDotClassed();
        this.isCaught                      |= other.isCaught();
        this.isSimpleEnum                  |= other.isSimpleEnum();
        this.isEscaping                    |= other.isEscaping();
        this.hasSideEffects                |= other.hasSideEffects();
        this.containsPackageVisibleMembers |= other.containsPackageVisibleMembers();
        this.invokesPackageVisibleMembers  |= other.invokesPackageVisibleMembers();
    }


    public static void setProgramClassOptimizationInfo(Clazz clazz)
    {
        clazz.setVisitorInfo(new ProgramClassOptimizationInfo());
    }


    public static ProgramClassOptimizationInfo getProgramClassOptimizationInfo(Clazz clazz)
    {
        return (ProgramClassOptimizationInfo)clazz.getVisitorInfo();
    }
}
