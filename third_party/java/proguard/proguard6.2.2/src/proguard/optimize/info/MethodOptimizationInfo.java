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
import proguard.classfile.util.MethodLinker;
import proguard.evaluation.value.Value;

/**
 * This class stores some optimization information that can be attached to
 * a method.
 *
 * @author Eric Lafortune
 */
public class MethodOptimizationInfo
{
    protected boolean hasNoSideEffects          = false;
    protected boolean hasNoExternalSideEffects  = false;
    protected boolean hasNoEscapingParameters   = false;
    protected boolean hasNoExternalReturnValues = false;
    protected Value   returnValue               = null;


    public boolean isKept()
    {
        return true;
    }


    public void setNoSideEffects()
    {
        hasNoSideEffects         = true;
        hasNoExternalSideEffects = true;
        hasNoEscapingParameters  = true;
    }


    public boolean hasNoSideEffects()
    {
        return hasNoSideEffects;
    }


    public void setNoExternalSideEffects()
    {
        hasNoExternalSideEffects = true;
        hasNoEscapingParameters  = true;
    }


    public boolean hasNoExternalSideEffects()
    {
        return hasNoExternalSideEffects;
    }


    public void setNoEscapingParameters()
    {
        hasNoEscapingParameters = true;
    }


    public boolean hasNoEscapingParameters()
    {
        return hasNoEscapingParameters;
    }


    public void setNoExternalReturnValues()
    {
        hasNoExternalReturnValues = true;
    }


    public boolean hasNoExternalReturnValues()
    {
        return hasNoExternalReturnValues;
    }


    public void setReturnValue(Value returnValue)
    {
        this.returnValue = returnValue;
    }


    public Value getReturnValue()
    {
        return returnValue;
    }


    // Methods that may be specialized.

    public boolean hasSideEffects()
    {
        return !hasNoSideEffects;
    }


    public boolean canBeMadePrivate()
    {
        return false;
    }


    public boolean catchesExceptions()
    {
        return true;
    }


    public boolean branchesBackward()
    {
        return true;
    }


    public boolean invokesSuperMethods()
    {
        return true;
    }


    public boolean invokesDynamically()
    {
        return true;
    }


    public boolean accessesPrivateCode()
    {
        return true;
    }


    public boolean accessesPackageCode()
    {
        return true;
    }


    public boolean accessesProtectedCode()
    {
        return true;
    }


    public boolean hasSynchronizedBlock()
    {
        return true;
    }


    public boolean assignsFinalField()
    {
        return true;
    }


    public boolean returnsWithNonEmptyStack()
    {
        return false;
    }


    public int getInvocationCount()
    {
        return Integer.MAX_VALUE;
    }


    public int getParameterSize()
    {
        return 0;
    }


    public boolean hasUnusedParameters()
    {
        return false;
    }


    public boolean isParameterUsed(int variableIndex)
    {
        return true;
    }


    public long getUsedParameters()
    {
        return -1L;
    }


    public boolean hasParameterEscaped(int parameterIndex)
    {
        return true;
    }


    public long getEscapedParameters()
    {
        return -1L;
    }


    public boolean isParameterEscaping(int parameterIndex)
    {
        return !hasNoEscapingParameters;
    }


    public long getEscapingParameters()
    {
        return hasNoEscapingParameters ? 0L : -1L;
    }


    public boolean isParameterModified(int parameterIndex)
    {
        // TODO: Refine for static methods.
        return
            !hasNoSideEffects &&
            (!hasNoExternalSideEffects || parameterIndex == 0);
    }


    public long getModifiedParameters()
    {
        // TODO: Refine for static methods.
        return
            hasNoSideEffects         ? 0L :
            hasNoExternalSideEffects ? 1L :
                                       -1L;
    }


    public boolean modifiesAnything()
    {
        return !hasNoExternalSideEffects;
    }


    public Value getParameterValue(int parameterIndex)
    {
        return null;
    }


    public boolean returnsParameter(int parameterIndex)
    {
        return true;
    }


    public long getReturnedParameters()
    {
        return -1L;
    }


    public boolean returnsNewInstances()
    {
        return true;
    }


    public boolean returnsExternalValues()
    {
        return !hasNoExternalReturnValues;
    }


    public static void setMethodOptimizationInfo(Clazz clazz, Method method)
    {
        MethodLinker.lastMember(method).setVisitorInfo(new MethodOptimizationInfo());
    }


    public static MethodOptimizationInfo getMethodOptimizationInfo(Method method)
    {
        return (MethodOptimizationInfo)MethodLinker.lastMember(method).getVisitorInfo();
    }
}
